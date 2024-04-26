import torch, glob, os, random, czifile, tifffile, psutil, warnings
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from .crappifiers import Crappifier, Poisson

class ImageDataset(Dataset):
    def __init__(self, path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = Poisson(), n_frames : int = None, mode : str = "L", extension : str = "tif", val_split : float = 0.1, rotation : bool = True, split_seed : int = 0, transforms : list[torch.nn.Module] = None):
        r"""Training dataset for loading high-resolution images from individual files and returning high-low-resolution pairs, the latter receiving crappification.

        Dataset used for pre-tiled image files. For image sheets (e.g. .czi files), use :class:`SlidingDataset`.

        LR mode (dataset loads only unmodified low-resolution images for prediction) can be enabled by
        either inputting images less than or equal to LR size (``hr_res``/``lr_scale``) or by setting ``lr_scale`` = 1 and ``hr_res`` = LR resolution.

        Args:
            path (Path) : Path to folder containing high-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            crappifier (Crappifier) : Crappifier for degrading low-resolution images to simulate undersampling. Not used in LR mode. Default is :class:`Poisson`.

            n_frames (int) : Amount of stacked frames per image, disregarding color channels. Can also be list of low-resolution and high-resolution stack amounts respectively. A value of None uses all stacked image frames. Default is None.

            mode (str) : Color mode for loading images, e.g. "L" for grayscale, "RGB" for color. Default is "L".

            extension (str) : File extension of images. Default is "tif".

            val_split (float) : Proportion of images to be held out for evaluation/prediction. Default is 0.1.

            rotation (bool) : Whether to randomly rotate and/or flip images when loading data. Only used during training if applicable. Default is True.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is 0.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.path = Path(path) if type(path) is str else path
        if not self.path.exists(): raise FileNotFoundError(f'Path "{self.path}" does not exist.')

        self.hr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.path))
        if not len(self.hr_files) > 0: raise FileNotFoundError(f'No .{extension} files exist in path "{self.path}".')

        self.mode = mode.upper()
        self.n_frames = _get_n_frames(n_frames, self.mode)

        self.slices, max_size = [], 0
        for image_idx in range(len(self.hr_files)):
            image = Image.open(Path(self.path, self.hr_files[image_idx]))
            self.slices.append(1 if self.n_frames is None else image.n_frames // self.n_frames[0])
            max_size = max(max(image.size), max_size)

        self.val_idx = _get_val_idx(self.slices, val_split, split_seed)
        self.is_lr = max_size <= hr_res//lr_scale or (lr_scale == 1)
        if self.is_lr: print("LR mode is enabled, dataset will load only unmodified low-resolution images.")
        self.crop_res = min(hr_res, max_size) * (lr_scale if self.is_lr else 1)

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.crappifier = crappifier
        self.rotation = rotation
        self.transforms = transforms

    def __getitem__(self, idx, pp=False):
        if idx >= len(self): raise IndexError(f"Tried to retrieve invalid image. Index {idx} is not less than {len(self)} total image frame slices.")

        is_val = idx in self.val_idx or pp
        image_idx, idx = _get_image_idx(idx, self.slices)

        hr = _load_image(self.path, self.hr_files[image_idx], self.mode, self.n_frames[0] if self.n_frames is not None else None, self.slices[image_idx], idx)
        
        if self.is_lr:
            # Rotation and crappifier is disabled in lr mode
            return _ready_lr(hr, self.hr_res//self.lr_scale, self.transforms)

        return _gen_pair(hr, self.hr_res, self.lr_scale, False if is_val else self.rotation, self.crappifier, self.transforms, self.n_frames)
    
    def __len__(self):
        return sum(self.slices)
    
    def __repr__(self):
        return f'ImageDataset from path "{self.path}"\n{len(self.hr_files)} files with {len(self)} total frame slices\nLR mode {"enabled" if self.is_lr else "disabled"}'

    def _get_name(self, idx):
        image_idx, idx = _get_image_idx(idx, self.slices)
        return self.hr_files[image_idx].split('.')[0] + (f"_{idx}" if self.n_frames is not None else "")

class SlidingDataset(Dataset):
    def __init__(self, path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = Poisson(), overlap : int = 128, n_frames : int = None, stack : str = "TZ", mode : str = "L", extension : str = "czi", preload : bool = True, val_split : float = 0.1, rotation : bool = True, split_seed : int = 0, transforms : list[torch.nn.Module] = None):
        r"""Training dataset for loading high-resolution image tiles from image sheets and returning high-low-resolution pairs, the latter receiving crappification.

        Dataset used for image sheets (e.g. .czi files). For pre-tiled image files, use :class:`ImageDataset`.

        LR mode (dataset loads only unmodified low-resolution images for prediction) can be enabled by setting ``lr_scale`` = 1 and ``hr_res`` = LR resolution.

        Args:
            path (Path) : Path to folder containing high-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            crappifier (Crappifier) : Crappifier for degrading low-resolution images to simulate undersampling. Not used in LR mode. Default is :class:`Poisson`.

            overlap (int) : Overlapping pixels between neighboring tiles to increase effective dataset size. Default is 128.

            n_frames (int) : Amount of stacked frames per image, disregarding color channels. Can also be list of low-resolution and high-resolution stack amounts respectively. A value of None uses all stacked image frames. Default is None.

            stack (str) : Multiframe stack handling mode, e.g "T" for time stack, "Z" for z dimension stack, "TZ" or "ZT" for both, determining flattenting order. Only applicable if loading from czi. Default is "TZ".

            mode (str) : Color mode for loading images, e.g. "L" for grayscale, "RGB" for color. Default is "L".

            extension (str) : File extension of images. Default is "czi".

            preload (bool) : Whether to preload images in memory (not VRAM) for faster dataloading. Default is True.

            val_split (float) : Proportion of images to be held out for evaluation/prediction. Default is 0.1.

            rotation (bool) : Whether to randomly rotate and/or flip images when loading data. Only used during training if applicable. Default is True.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is 0.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.path = Path(path) if type(path) is str else path
        if not self.path.exists(): raise FileNotFoundError(f'Path "{self.path}" does not exist.')
        
        self.hr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.path))
        if not len(self.hr_files) > 0: raise FileNotFoundError(f'No .{extension} files exist in path "{self.path}".')

        if not hr_res > overlap: raise ValueError(f"hr_res must be greater than overlap. Given values are {hr_res} and {overlap} respectively.")
        self.stride = hr_res - overlap
        self.stack = stack.upper()
        self.mode = mode.upper()
        self.n_frames = _get_n_frames(n_frames, self.mode)
        
        self.preload = _preload(preload, [self.path], [self.hr_files], self.mode, self.stack)

        self.tiles, self.slices = [], []
        for image_idx in range(len(self.hr_files)):
            image = self.preload[image_idx] if self.preload else _load_sheet(self.path, self.hr_files[image_idx], self.stack, self.mode)
            tiles_x, tiles_y = _n_tiles(image, hr_res, self.stride)
            self.tiles.append(tiles_x * tiles_y)
            self.slices.append(1 if self.n_frames is None else image.shape[0] // self.n_frames[0])

        self.val_idx = _get_val_idx(self.slices, val_split, split_seed, self.tiles)
        self.is_lr = (lr_scale == 1)
        if self.is_lr: print("LR mode is enabled, dataset will load only unmodified low-resolution images.")
        self.crop_res = hr_res * (lr_scale if self.is_lr else 1)

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.crappifier = crappifier
        self.rotation = rotation
        self.transforms = transforms
    
    def __getitem__(self, idx, pp=False):
        if idx >= len(self): raise IndexError(f"Tried to retrieve invalid image. Index {idx} is not less than {len(self)} total image frame slices.")

        is_val = idx in self.val_idx or pp
        image_idx, idx = _get_image_idx(idx, self.slices, self.tiles)

        hr = _sliding_window(self.preload[image_idx] if self.preload else _load_sheet(self.path, self.hr_files[image_idx], self.stack, self.mode), self.hr_res, self.stride, self.n_frames[0], self.slices[image_idx], idx)

        if self.is_lr:
            return _ready_lr(hr, self.hr_res, self.transforms)

        return _gen_pair(hr, self.hr_res, self.lr_scale, False if is_val else self.rotation, self.crappifier, self.transforms, self.n_frames)
    
    def __len__(self):
        return sum([self.tiles[idx] * self.slices[idx] for idx in range(len(self.hr_files))])
    
    def __repr__(self):
        return f'SlidingDataset from path "{self.path}"\n{len(self.hr_files)} files with {len(self)} total frame slices\nLR mode {"enabled" if self.is_lr else "disabled"}'
    
    def _get_name(self, idx):
        image_idx, idx = _get_image_idx(idx, self.slices, self.tiles)
        return f"{self.hr_files[image_idx].split('.')[0]}_{idx}"

class PairedImageDataset(Dataset):
    def __init__(self, hr_path : Path, lr_path : Path, hr_res : int = 512, lr_scale : int = 4, n_frames : int = None, mode : str = "L", extension : str = "tif", val_split : float = 1, rotation : bool = True, split_seed : int = None, transforms : list[torch.nn.Module] = None):
        r"""Testing dataset for loading paired high-low-resolution images without using crappification. Can also be used for approximating :class:`Crappifier` parameters.

        Args:
            hr_path (Path) : Path to folder containing high-resolution images. Can also be a str.

            lr_path (Path) : Path to folder containing low-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            n_frames (int) : Amount of stacked frames per image, disregarding color channels. Can also be list of low-resolution and high-resolution stack amounts respectively. A value of None uses all stacked image frames. Default is None.

            mode (str) : Color mode for loading images, e.g. "L" for grayscale, "RGB" for color. Default is "L".

            extension (str) : File extension of images. Default is "tif".

            val_split (float) : Proportion of images to be held out for evaluation/prediction. Default is 1.

            rotation (bool) : Whether to randomly rotate and/or flip images when loading data. Only used during training if applicable. Default is True.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is None.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.hr_path = Path(hr_path) if type(hr_path) is str else hr_path
        self.lr_path = Path(lr_path) if type(lr_path) is str else lr_path
        for path in [self.hr_path, self.lr_path]:
            if not path.exists(): raise FileNotFoundError(f'Path "{path}" does not exist.')
        if self.hr_path == self.lr_path: warnings.warn("hr_path is equal to lr_path! Consider using ImageDataset instead.", stacklevel=2)

        self.hr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.hr_path))
        self.lr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.lr_path))
        for files, path in zip([self.hr_files, self.lr_files], [self.hr_path, self.lr_path]):
            if not len(files) > 0: raise FileNotFoundError(f'No .{extension} files exist in path "{path}".')
        if len(self.hr_files) != len(self.lr_files): raise FileNotFoundError(f"Mismatch between amounts of high-low-resolution images. Found {len(self.hr_files)} high-resolution and {len(self.lr_files)} low-resolution images.")

        self.mode = mode.upper()
        self.n_frames = _get_n_frames(n_frames, self.mode)

        self.slices, max_size = [], 0
        for image_idx in range(len(self.hr_files)):
            image = Image.open(Path(self.hr_path, self.hr_files[image_idx]))
            self.slices.append(1 if self.n_frames is None else image.n_frames // self.n_frames[0])
            max_size = max(max(image.size), max_size)

        self.val_idx = _get_val_idx(self.slices, val_split, split_seed)
        self.is_lr = False
        self.crop_res = min(hr_res, max_size)

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.rotation = rotation
        self.transforms = transforms
    
    def __getitem__(self, idx, pp=False):
        if idx >= len(self): raise IndexError(f"Tried to retrieve invalid image. Index {idx} is not less than {len(self)} total image frame slices.")

        is_val = idx in self.val_idx or pp
        image_idx, idx = _get_image_idx(idx, self.slices)

        hr = _load_image(self.hr_path, self.hr_files[image_idx], self.mode, self.n_frames[0] if self.n_frames is not None else None, self.slices[image_idx], idx)
        lr = _load_image(self.lr_path, self.lr_files[image_idx], self.mode, self.n_frames[0] if self.n_frames is not None else None, self.slices[image_idx], idx)

        return _transform_pair(hr, lr, self.hr_res, self.hr_res//self.lr_scale, False if is_val else self.rotation, self.transforms, self.n_frames)
    
    def __len__(self):
        return sum(self.slices)
    
    def __repr__(self):
        return f'PairedImageDataset from paths "{self.hr_path}" and "{self.lr_path}"\n{len(self.hr_files)+len(self.lr_files)} files with {len(self)} total frame slices'

    def _get_name(self, idx):
        image_idx, idx = _get_image_idx(idx, self.slices)
        return self.lr_files[image_idx].split('.')[0] + (f"_{idx}" if self.n_frames is not None else "")

class PairedSlidingDataset(Dataset):
    def __init__(self, hr_path : Path, lr_path : Path, hr_res : int = 512, lr_scale : int = 4, overlap : int = 128, n_frames : int = None, stack : str = "TZ", mode : str = "L", extension : str = "czi", preload : bool = True, val_split : float = 1, rotation : bool = True, split_seed : int = None, transforms : list[torch.nn.Module] = None):
        r"""Testing dataset for loading high-low-resolution image tiles from image sheets without crappification. Can also be used for approximating :class:`Crappifier` parameters.

        Dataset used for image sheets (e.g. .czi files). For pre-tiled image files, use :class:`ImageDataset`.

        Args:
            hr_path (Path) : Path to folder containing high-resolution images. Can also be a str.

            lr_path (Path) : Path to folder containing low-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            overlap (int) : Overlapping pixels between neighboring tiles to increase effective dataset size. Default is 128.

            n_frames (int) : Amount of stacked frames per image, disregarding color channels. Can also be list of low-resolution and high-resolution stack amounts respectively. A value of None uses all stacked image frames. Default is None.

            stack (str) : Multiframe stack handling mode, e.g "T" for time stack, "Z" for z dimension stack, "TZ" or "ZT" for both, determining flattenting order. Only applicable if loading from czi. Default is "TZ".

            mode (str) : Color mode for loading images, e.g. "L" for grayscale, "RGB" for color. Default is "L".

            extension (str) : File extension of images. Default is "czi".

            preload (bool) : Whether to preload images in memory (not VRAM) for faster dataloading. Default is True.

            val_split (float) : Proportion of images to be held out for evaluation/prediction. Default is 1.

            rotation (bool) : Whether to randomly rotate and/or flip images when loading data. Only used during training if applicable. Default is True.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is None.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.hr_path = Path(hr_path) if type(hr_path) is str else hr_path
        self.lr_path = Path(lr_path) if type(lr_path) is str else lr_path
        for path in [self.hr_path, self.lr_path]:
            if not path.exists(): raise FileNotFoundError(f'Path "{path}" does not exist.')
        if self.hr_path == self.lr_path: warnings.warn("hr_path is equal to lr_path! Consider using SlidingDataset instead.", stacklevel=2)
        
        self.hr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.hr_path))
        self.lr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.lr_path))
        for files, path in zip([self.hr_files, self.lr_files], [self.hr_path, self.lr_path]):
            if not len(files) > 0: raise FileNotFoundError(f'No .{extension} files exist in path "{path}".')
        if len(self.hr_files) != len(self.lr_files): raise FileNotFoundError(f"Mismatch between amounts of high-low-resolution images. Found {len(self.hr_files)} high-resolution and {len(self.lr_files)} low-resolution images.")

        if not hr_res > overlap: raise ValueError(f"hr_res must be greater than overlap. Given values are {hr_res} and {overlap} respectively.")
        self.stride = hr_res - overlap
        self.stack = stack.upper()
        self.mode = mode.upper()
        self.n_frames = _get_n_frames(n_frames, self.mode)

        self.preload = _preload(preload, [self.hr_path, self.lr_path], [self.hr_files, self.lr_files], self.mode, self.stack)

        self.tiles, self.slices = [], []
        for image_idx in range(len(self.hr_files)):
            image = self.preload[0][image_idx] if self.preload else _load_sheet(self.hr_path, self.hr_files[image_idx], self.stack, self.mode)
            tiles_x, tiles_y = _n_tiles(image, hr_res, self.stride)
            self.tiles.append(tiles_x * tiles_y)
            self.slices.append(1 if self.n_frames is None else image.shape[0] // self.n_frames[0])

        self.val_idx = _get_val_idx(self.slices, val_split, split_seed, self.tiles)
        self.is_lr = False
        self.crop_res = hr_res

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.rotation = rotation
        self.transforms = transforms
    
    def __getitem__(self, idx, pp=False):
        if idx >= len(self): raise IndexError(f"Tried to retrieve invalid image. Index {idx} is not less than {len(self)} total image frame slices.")

        is_val = idx in self.val_idx or pp
        image_idx, idx = _get_image_idx(idx, self.slices, self.tiles)

        hr = _sliding_window(self.preload[0][image_idx] if self.preload else _load_sheet(self.hr_path, self.hr_files[image_idx], self.stack, self.mode), self.hr_res, self.stride, self.n_frames[0], self.slices[image_idx], idx)
        lr = _sliding_window(self.preload[1][image_idx] if self.preload else _load_sheet(self.lr_path, self.lr_files[image_idx], self.stack, self.mode), self.hr_res//self.lr_scale, self.stride//self.lr_scale, self.n_frames[0], self.slices[image_idx], idx)

        return _transform_pair(hr, lr, self.hr_res, self.hr_res//self.lr_scale, False if is_val else self.rotation, self.transforms, self.n_frames)
    
    def __len__(self):
        return sum([self.tiles[idx] * self.slices[idx] for idx in range(len(self.hr_files))])
    
    def __repr__(self):
        return f'PairedSlidingDataset from paths "{self.hr_path}" and "{self.lr_path}"\n{len(self.hr_files)+len(self.lr_files)} files with {len(self)} total frame slices'
    
    def _get_name(self, idx):
        image_idx, idx = _get_image_idx(idx, self.slices, self.tiles)
        return f"{self.lr_files[image_idx].split('.')[0]}_{idx}"

def preprocess_dataset(dataset : Dataset, preprocess_hr : bool = False, out_dir : str = "preprocess"):
    r"""Saves processed frame slices from a given dataset, including processes such as crappification or cropping/padding.
    
    Args:
        dataset (Dataset) : Dataset to load images from. Preprocessing is determined by the dataloading procedure as specified in dataset arguments. Rotation is disabled.

        preprocess_hr (bool) : Whether to save preprocessed high-resolution images in addition to preprocessed low-resolution images. Default is False.

        out_dir (str) : Directory to save preprocessed images. Default is "preprocess".
    """
    os.makedirs(f"{out_dir}/lr", exist_ok=True)
    if preprocess_hr:
        os.makedirs(f"{out_dir}/hr", exist_ok=True)

    progress = tqdm(range(len(dataset)))
    for idx in progress:
        hr, lr = dataset.__getitem__(idx, pp=True)
        hr, lr = np.asarray(hr, dtype=np.uint8), np.asarray(lr, dtype=np.uint8)
        
        tifffile.imwrite(f"{out_dir}/lr/{dataset._get_name(idx)}.tif", lr)
        if preprocess_hr:
            tifffile.imwrite(f"{out_dir}/hr/{dataset._get_name(idx)}.tif", hr)

# TODO: crappify_images

def _gen_pair(hr, hr_res, lr_scale, rotation, crappifier, transforms, n_frames):
    r"""Creates training ready pair of images from a single high-resolution image.
    """
    hr = _square_crop(hr, hr_res)
    hr = _pad_image(hr, hr_res)
    
    # Set random rotation and flip in xy axis
    if rotation:
        hr = np.rot90(hr, axes=(1,2)) if bool(random.getrandbits(1)) else hr
        hr = np.flip(hr, axis=random.choice((1,2,(1,2))))

    # Crappification
    lr = np.stack([Image.fromarray(channel).resize(([hr_res//lr_scale]*2), Image.Resampling.BILINEAR) for channel in hr])
    if crappifier is not None:
        # Allows either Crappifier or nn.Module (or any callable function) to be used as a crappifier
        lr = crappifier.crappify(lr) if issubclass(type(crappifier), Crappifier) else crappifier(lr)
        lr = np.clip(lr.round(), 0, 255)

    hr = _slice_center(hr, n_frames)

    return _tensor_ready(hr, transforms), _tensor_ready(lr, transforms)

def _transform_pair(hr, lr, hr_res, lr_res, rotation, transforms, n_frames):
    r"""Same as _gen_pair, but uses paired high-low-resolution images with no crappifier.
    """
    hr = _square_crop(hr, hr_res)
    lr = _square_crop(lr, lr_res)

    hr = _pad_image(hr, hr_res)
    lr = _pad_image(lr, lr_res)

    if rotation:
        choice = bool(random.getrandbits(1)), random.choice((0,1,(0,1)))

        hr, lr = np.rot90(hr, axes=(1,2)) if choice[0] else hr, np.rot90(lr, axes=(1,2)) if choice[0] else lr
        hr, lr = np.flip(hr, axis=choice[1]), np.flip(lr, axis=choice[1])
    
    hr = _slice_center(hr, n_frames)

    return _tensor_ready(hr, transforms), _tensor_ready(lr, transforms)

def _ready_lr(lr, lr_res, transforms):
    r"""Processes lr images for prediction. Same as transform pair but without rotation.
    """
    lr = _square_crop(lr, lr_res)
    lr = _pad_image(lr, lr_res)

    return _tensor_ready(lr, transforms)

def _tensor_ready(image, transforms):
    image = torch.tensor(image.copy(), dtype=torch.float)

    # Additional nn.Module user transforms
    if transforms is not None:
        for transform in transforms:
            image = transform(image)
    
    return image

def _square_crop(image, max_res):
    height, width = image.shape[-2:]

    if [height, width] == [max_res]*2:
        return image
    
    size = min(height, width, max_res)
    start_x = (height - size) // 2
    start_y = (width - size) // 2
    
    return image[:, start_x:start_x + size, start_y:start_y + size]

def _pad_image(image, res):
    if image.shape[-1] < res:
        return np.stack([np.pad(channel, pad_width=[[0,res-image.shape[-1]]]*2, mode="reflect") for channel in image])
    return image

def _preload(preload, path, files, mode, stack):
    if not preload:
        return False
    
    size = sum([sum([os.stat(Path(idx_path, file)).st_size for file in idx_files]) for idx_path, idx_files in zip(path, files)]) / 10**9
    memory = psutil.virtual_memory().available / 10**9
    if size > memory:
        warnings.warn(f"Total dataset size {size:.2f}GB is greater than available memory of {memory:.2f}GB. Consider disabling preloading to avoid potential slowdowns.", stacklevel=2)

    print(f"Preloading {sum([len(item) for item in files])} images into memory...")
    preload = [[_load_sheet(idx_path, file, stack, mode) for file in idx_files] for idx_path, idx_files in zip(path, files)]
    return preload[0] if len(preload) == 1 else preload

def _load_image(path, file, mode, n_frames, slices, idx):
    hr = Image.open(Path(path, file))

    hr = _frame_channel(hr, mode)
    return _slice_image(hr, n_frames, slices, idx)

def _load_sheet(path, file, stack, mode):
    extension = file.split(".")[-1]
    if extension == "czi":
        # Retrieve same channel information from czi regardless of channel order
        image = czifile.CziFile(Path(path, file))
        out_axes = "TZCXY"

        # Get info only from out_axes axes
        slice_idx, slice_axes = [], []
        for axis in image.axes:
            if axis not in out_axes:
                slice_idx.append(0)
            else:
                slice_idx.append(slice(None))
                slice_axes.append(axis)
        image = image.asarray()[tuple(slice_idx)]

        # Rearrange axes to out_axes order
        axes_idx = [out_axes.rfind(axis) for axis in slice_axes]
        image = np.moveaxis(image, range(len(image.shape)), axes_idx)

        # TODO: Assumes images must have all out_axes?
        # Disregard additional channels and/or reorder
        if mode == "L":
            image = np.mean(image, axis=2)
        match stack:
            case "T":
                image = image[:,0]
            case "Z":
                image = image[0]
            case "ZT":
                image = np.moveaxis(image, 0, 1)
            case "TZ":
                pass
            case _:
                raise ValueError(f"Stack type {stack} is not valid.")
        
        # Flatten channel dimensions
        image = np.reshape(image, [-1, image.shape[-2], image.shape[-1]])
        if image.max() != 0:
            image = image / (image.max() / 255)
        return image.astype(np.uint8)
    else:
        image = Image.open(Path(path, file))
        return _frame_channel(image, mode)

def _sliding_window(image, size, stride, n_frames, n_slices, idx):
    tiles_x, tiles_y = _n_tiles(image, size, stride)
    tile_idx = idx // n_slices

    start_x = tile_idx // tiles_y * stride
    start_y = tile_idx % tiles_y * stride

    image = image[..., start_x:start_x + size, start_y:start_y + size]

    return _slice_image(image, n_frames, n_slices, idx)

def _frame_channel(image, mode):
    # Create frame dimension
    if image.n_frames > 1:
        image = np.stack([np.asarray(_seek_channel(image, frame).convert(mode), dtype=np.uint8) for frame in range(image.n_frames)])
    else:
        image = np.asarray(image.convert(mode), dtype=np.uint8)[np.newaxis, :, :]
    
    return image

def _slice_image(image, n_frames, n_slices, idx):
    if n_frames == None:
        return image
    
    # Only need to calculate residual idx of specific tile
    idx = idx % n_slices
    idx *= n_frames
    return image[idx:idx+n_frames]

def _slice_center(image, n_frames):
    if n_frames is None or n_frames[0] == n_frames[1] or n_frames[1] > image.shape[-3]:
        return image

    center = image.shape[-3] // 2
    half = n_frames[1] // 2
    if n_frames[1] % 2 == 0:
        return image[...,center - half:center + half,:,:]
    else:
        return image[...,center - half:center + half + 1,:,:]

def _seek_channel(image, idx):
    # Because we can't have nice things...
    image.seek(idx)
    return image

def _n_tiles(image, size, stride):
    x, y = image.shape[-2:]

    tiles_x = max(0, (x - size) // stride + 1)
    tiles_y = max(0, (y - size) // stride + 1)
    return tiles_x, tiles_y

def _get_n_frames(n_frames, mode):
    if n_frames is None:
        return n_frames
    
    n_frames = _force_list(n_frames)
    n_frames = n_frames*2 if len(n_frames) == 1 else n_frames
    n_frames = [item*3 for item in n_frames] if mode == "RGB" else n_frames
    return n_frames # [in (low-resolution), out (high-resolution)]

def _force_list(item):
    # Because we still can't have nice things...
    if type(item) is int:
        return [item]
    elif type(item) is not list:
        return list(item)
    return item

def _get_image_idx(idx, slices, tiles=None):
    tiles = [1]*len(slices) if tiles is None else tiles

    image_idx = 0
    for slice, tile in zip(slices, tiles):
        if idx < slice * tile:
            return image_idx, idx
        else:
            idx -= slice * tile
            image_idx += 1

def _get_val_idx(slices, split, seed, tiles=None):
    # Tiles are individual images
    if tiles is not None:
        tile_slices = []
        for slice, tile in zip(slices, tiles):
            tile_slices.extend([slice]*tile)
        slices = tile_slices

    # Select random image indexes as validation
    val_slices = list(range(len(slices)))
    if seed is not None and split < 1:
        np.random.seed(seed)
        np.random.shuffle(val_slices)
    val_slices = val_slices[-max(1,int(split*len(slices))):]
    
    # Convert image indexes to frame indexes for dataloading
    val_idx, idx = [], 0
    for slice_idx, slice in enumerate(slices):
        if slice_idx in val_slices:
            val_idx.extend(range(idx,idx+slice))
        idx += slice

    return val_idx

def _invert_idx(idx, idx_len):
    idx_range = np.arange(idx_len)
    inverse = np.logical_not(np.isin(idx_range, idx))
    return idx_range[inverse]

class _RandomIterIdx:
    def __init__(self, idx, seed=False):
        self.idx = idx
        self.seed = seed
        
    def __iter__(self):
        random_idx = self.idx.copy()
        if self.seed:
            np.random.seed(0)
            np.random.shuffle(random_idx)
        else:
            random.shuffle(random_idx)
        yield from random_idx
    
    def __len__(self):
        return len(self.idx)
