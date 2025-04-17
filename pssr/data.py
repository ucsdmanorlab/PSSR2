import torch, glob, os, random, czifile, tifffile, psutil, warnings
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from .crappifiers import Crappifier, Poisson
from .util import _force_list

# TODO: dataset call: return (hr, lr), extra or hr, lr

class ImageDataset(Dataset):
    def __init__(self, path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = Poisson(), n_frames : list[int] = -1, extension : str = "tif", val_split : float = 0.1, rotation : bool = True, split_seed : int = 0, extra_path : Path = None, extra_scale : int = 1, transforms : list[torch.nn.Module] = None):
        r"""Training dataset for loading high-resolution images from individual files and returning high-low-resolution pairs, the latter receiving crappification.

        Dataset used for pre-tiled image files. For image sheets (e.g. .czi files), use :class:`SlidingDataset`.

        LR mode (dataset loads only unmodified low-resolution images for prediction) can be enabled by
        either inputting images less than or equal to LR size (``hr_res``/``lr_scale``) or by setting ``lr_scale`` = -1 and ``hr_res`` = LR resolution.

        Args:
            path (Path) : Path to folder containing high-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            crappifier (Crappifier) : Crappifier for degrading low-resolution images to simulate undersampling. Not used in LR mode. Default is :class:`Poisson`.

            n_frames (list[int]) : Amount of stacked frames per image. Can also be list of low-resolution and high-resolution stack amounts respectively. A value of -1 uses all stacked image frames. Default is -1.

            extension (str) : File extension of images. Default is "tif".

            val_split (float) : Proportion of images to be held out for evaluation/prediction. Default is 0.1.

            rotation (bool) : Whether to randomly rotate and/or flip images when loading data. Only applicable during training. Default is True.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is 0.

            extra_path (Path) : Optional path to folder containing images with additional information to be used in training loss functions. Each image in `path` must have a corresponding image of the same shape with a scale factor of extra_scale. Default is None.

            extra_scale (int) : Scale factor for extra images. Default is 1.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.path = Path(path) if type(path) is str else path
        if not path or not self.path.exists(): raise FileNotFoundError(f'Path "{self.path}" does not exist.')

        self.hr_files = _root_glob(f"*.{extension}", root_dir=self.path)
        if not len(self.hr_files) > 0: raise FileNotFoundError(f'No .{extension} files exist in path "{self.path}".')

        if extra_path is not None:
            self.extra_path = Path(extra_path) if type(extra_path) is str else extra_path
            if not extra_path or not self.extra_path.exists(): raise FileNotFoundError(f'Extra path "{self.extra_path}" does not exist.')

            self.extra_hr_files = _root_glob(f"*.{extension}", root_dir=self.extra_path)
            if not len(self.extra_hr_files) > 0: raise FileNotFoundError(f'No .{extension} files exist in extra path "{self.extra_path}".')

            if len(self.hr_files) != len(self.extra_hr_files): raise FileNotFoundError(f'Number of files in "path" and "extra_path" are not equal. Found {len(self.hr_files)} files and {len(self.extra_hr_files)} files respectively.')
        else:
            self.extra_path = None
            self.extra_hr_files = None

        lr_scale = None if lr_scale == -1 else lr_scale
        self.mode = "L"
        self.n_frames = _get_n_frames(n_frames)

        # TODO: Decrease loading times for large datasets
        self.slices, max_size = [], 0
        for image_idx in range(len(self.hr_files)):
            image = Image.open(Path(self.path, self.hr_files[image_idx]))
            self.slices.append(1 if self.n_frames is None else image.n_frames // max(self.n_frames))
            max_size = max(max(image.size), max_size)

            if self.extra_hr_files is not None:
                extra_image = Image.open(Path(self.extra_path, self.extra_hr_files[image_idx]))
                needed_extra = tuple([size * extra_scale for size in image.shape[1:]])
                if extra_image.shape[1:] != needed_extra: raise ValueError(f'The corresponding image to "{self.hr_files[image_idx]}" does not have the correct shape. From image shape of {image.shape[1:]} and "extra_scale" of {extra_scale}, expected extra image shape of {needed_extra}, but got {extra_image.shape[1:]}.')
                if image.shape[0] != extra_image.shape[0] and self.n_frames is not None: raise ValueError(f'The corresponding image to "{self.hr_files[image_idx]}" does not have the correct number of frames. n_frames must be -1 if number of image and extra_image frames are not equal. Respective number of frames are {image.shape[0]} and {extra_image.shape[0]}')

        self.val_idx = _get_val_idx(self.slices, val_split, split_seed)
        self.crop_res = min(hr_res, max_size)

        self.is_lr = lr_scale == None or max_size <= hr_res//lr_scale
        if self.is_lr:
            print("LR mode is enabled, dataset will load only unmodified low-resolution images.")
            if val_split < 1:
                warnings.warn("val_split is less than 1, not all low-resolution images will be used in prediciton.", stacklevel=2)

            # if self.extra_path is not None: raise ValueError("extra_path cannot be provided when LR mode is enabled.")

        self.hr_res = hr_res
        self.lr_scale = lr_scale if lr_scale is not None else 1
        self.crappifier = crappifier
        self.rotation = rotation
        self.extra_scale = extra_scale
        self.transforms = transforms

    def __getitem__(self, idx, pp=False):
        if idx >= len(self): raise IndexError(f"Tried to retrieve invalid image. Index {idx} is not less than {len(self)} total image frame slices.")

        is_val = idx in self.val_idx or pp
        image_idx, idx = _get_image_idx(idx, self.slices)

        hr = _load_image(self.path, self.hr_files[image_idx], self.mode, max(self.n_frames) if self.n_frames is not None else None, self.slices[image_idx], idx)
        
        cur_rot = [bool(random.getrandbits(1)), random.choice((1,2,(1,2)))] if self.rotation and not is_val else False

        out =  _gen_pair(hr, self.hr_res, self.lr_scale, cur_rot, self.crappifier, self.transforms, self.n_frames) if not self.is_lr else _ready_lr(hr, self.hr_res//self.lr_scale, self.transforms)

        if self.extra_hr_files is not None:
            extra = _load_image(self.extra_path, self.extra_hr_files[image_idx], self.mode, max(self.n_frames) if self.n_frames is not None else None, self.slices[image_idx], idx)
            if cur_rot:
                extra = np.rot90(extra, axes=(1,2)) if cur_rot[0] else extra
                extra = np.flip(extra, axis=cur_rot[1])
            extra = _tensor_ready(extra, self.transforms)
            return out, extra
        else:
            return out
    
    def __len__(self):
        return sum(self.slices)
    
    def __repr__(self):
        return f'ImageDataset from path "{self.path}"\n{len(self.hr_files)} files with {len(self)} total frame slices\n{f"low-res: {self.hr_res//self.lr_scale}" if self.is_lr else f"high-res: {self.hr_res}, low-res: {self.hr_res//self.lr_scale}"}'

    def _get_name(self, idx):
        image_idx, idx = _get_image_idx(idx, self.slices)
        return self.hr_files[image_idx].split('.')[0] + (f"_{idx}" if self.n_frames is not None else "")

class SlidingDataset(Dataset):
    def __init__(self, path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = Poisson(), overlap : int = 128, n_frames : list[int] = -1, slide : bool = False, stack : str = "TZ", extension : str = "czi", preload : bool = True, val_split : float = 0.1, rotation : bool = True, split_seed : int = 0, extra_path : Path = None, extra_scale : int = 1, transforms : list[torch.nn.Module] = None):
        r"""Training dataset for loading high-resolution image tiles from image sheets and returning high-low-resolution pairs, the latter receiving crappification.

        Dataset used for image sheets (e.g. .czi files). For pre-tiled image files, use :class:`ImageDataset`.

        LR mode (dataset loads only unmodified low-resolution images for prediction) can be enabled by setting ``lr_scale`` = -1 and ``hr_res`` = LR resolution.

        Args:
            path (Path) : Path to folder containing high-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            crappifier (Crappifier) : Crappifier for degrading low-resolution images to simulate undersampling. Not used in LR mode. Default is :class:`Poisson`.

            overlap (int) : Overlapping pixels between neighboring tiles to increase effective dataset size. Default is 128.

            n_frames (list[int]) : Amount of stacked frames per image tile. Can also be list of low-resolution and high-resolution stack amounts respectively. A value of -1 uses all stacked image frames. Default is -1.

            slide (bool) : Whether to slide over stack dimensions rather than taking discrete non-overlapping slices, increasing the effective size of the dataset. Should not be used if more than one dimension is stacked. Default is False.

            stack (str) : Multiframe stack handling mode, e.g "T" for time stack, "Z" for z dimension stack, "TZ" or "ZT" for both, determining flattenting order. Only applicable if loading from czi. Default is "TZ".

            extension (str) : File extension of images. Default is "czi".

            preload (bool) : Whether to preload images in memory (not VRAM) for faster dataloading. Default is True.

            val_split (float) : Proportion of images to be held out for evaluation/prediction. Default is 0.1.

            rotation (bool) : Whether to randomly rotate and/or flip images when loading data. Only applicable during training. Default is True.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is 0.

            extra_path (Path) : Optional path to folder containing images with additional information to be used in training loss functions. Each image in `path` must have a corresponding image of the same shape with a scale factor of extra_scale. Default is None.

            extra_scale (int) : Scale factor for extra images. Default is 1.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.path = Path(path) if type(path) is str else path
        if not path or not self.path.exists(): raise FileNotFoundError(f'Path "{self.path}" does not exist.')

        self.hr_files = _root_glob(f"*.{extension}", root_dir=self.path)
        if not len(self.hr_files) > 0: raise FileNotFoundError(f'No .{extension} files exist in path "{self.path}".')

        if extra_path is not None:
            self.extra_path = Path(extra_path) if type(extra_path) is str else extra_path
            if not extra_path or not self.extra_path.exists(): raise FileNotFoundError(f'Extra path "{self.extra_path}" does not exist.')

            self.extra_hr_files = _root_glob(f"*.{extension}", root_dir=self.extra_path)
            if not len(self.extra_hr_files) > 0: raise FileNotFoundError(f'No .{extension} files exist in extra path "{self.extra_path}".')

            if len(self.hr_files) != len(self.extra_hr_files): raise FileNotFoundError(f'Number of files in "path" and "extra_path" are not equal. Found {len(self.hr_files)} files and {len(self.extra_hr_files)} files respectively.')
        else:
            self.extra_path = None
            self.extra_hr_files = None

        overlap = 0 if overlap is None else overlap
        if not hr_res > overlap: raise ValueError(f"hr_res must be greater than overlap. Given values are {hr_res} and {overlap} respectively.")
        self.stride = hr_res - overlap
        self.stack = stack.upper()
        
        lr_scale = None if lr_scale == -1 else lr_scale
        self.mode = "L"
        self.n_frames = _get_n_frames(n_frames)
        self.slide = slide
        
        self.preload = _preload(preload, [self.path], [self.hr_files], self.mode, self.stack)
        self.extra_preload = _preload(preload, [self.extra_path], [self.extra_hr_files], self.mode, self.stack) if self.extra_hr_files is not None else None

        self.tiles, self.slices = [], []
        for image_idx in range(len(self.hr_files)):
            image = self.preload[image_idx] if self.preload else _load_sheet(self.path, self.hr_files[image_idx], self.stack, self.mode)
            tiles_x, tiles_y = _n_tiles(image, hr_res, self.stride)
            self.tiles.append(tiles_x * tiles_y)
            self.slices.append(1 if self.n_frames is None else ((image.shape[0] - max(self.n_frames) + 1) if slide else (image.shape[0] // max(self.n_frames))))

            if self.extra_hr_files is not None:
                extra_image = self.extra_preload[image_idx] if self.extra_preload else _load_sheet(self.extra_path, self.extra_hr_files[image_idx], self.stack, self.mode)
                needed_extra = tuple([size * extra_scale for size in image.shape[1:]])
                if extra_image.shape[1:] != needed_extra: raise ValueError(f'The corresponding image to "{self.hr_files[image_idx]}" does not have the correct shape. From image shape of {image.shape[1:]} and "extra_scale" of {extra_scale}, expected extra image shape of {needed_extra}, but got {extra_image.shape[1:]}.')
                if image.shape[0] != extra_image.shape[0] and self.n_frames is not None: raise ValueError(f'The corresponding image to "{self.hr_files[image_idx]}" does not have the correct number of frames. n_frames must be -1 if number of image and extra_image frames are not equal. Respective number of frames are {image.shape[0]} and {extra_image.shape[0]}')

        self.val_idx = _get_val_idx(self.slices, val_split, split_seed, self.tiles)
        self.crop_res = hr_res

        self.is_lr = (lr_scale == None)
        if self.is_lr:
            print("LR mode is enabled, dataset will load only unmodified low-resolution images.")
            if val_split < 1:
                warnings.warn("val_split is less than 1, not all low-resolution images will be used in prediciton.", stacklevel=2)

            # if self.extra_path is not None: raise ValueError("extra_path cannot be provided when LR mode is enabled.")

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.crappifier = crappifier
        self.rotation = rotation
        self.extra_scale = extra_scale
        self.transforms = transforms
    
    def __getitem__(self, idx, pp=False):
        if idx >= len(self): raise IndexError(f"Tried to retrieve invalid image. Index {idx} is not less than {len(self)} total image frame slices.")

        is_val = idx in self.val_idx or pp
        image_idx, idx = _get_image_idx(idx, self.slices, self.tiles)

        hr = _sliding_window(self.preload[image_idx] if self.preload else _load_sheet(self.path, self.hr_files[image_idx], self.stack, self.mode), self.hr_res, self.stride, max(self.n_frames) if self.n_frames is not None else None, self.slices[image_idx], idx, self.slide)

        cur_rot = [bool(random.getrandbits(1)), random.choice((1,2,(1,2)))] if self.rotation and not is_val else False

        out = _gen_pair(hr, self.hr_res, self.lr_scale, cur_rot, self.crappifier, self.transforms, self.n_frames) if not self.is_lr else _ready_lr(hr, self.hr_res, self.transforms)

        if self.extra_hr_files is not None:
            extra = _sliding_window(self.extra_preload[image_idx] if self.extra_preload else _load_sheet(self.extra_path, self.extra_hr_files[image_idx], self.stack, self.mode), self.hr_res*self.extra_scale, self.stride*self.extra_scale, max(self.n_frames) if self.n_frames is not None else None, self.slices[image_idx], idx, self.slide)
            if cur_rot:
                extra = np.rot90(extra, axes=(1,2)) if cur_rot[0] else extra
                extra = np.flip(extra, axis=cur_rot[1])
            extra = _tensor_ready(extra, self.transforms)
            return out, extra
        else:
            return out
    
    def __len__(self):
        return sum([self.tiles[idx] * self.slices[idx] for idx in range(len(self.hr_files))])
    
    def __repr__(self):
        return f'SlidingDataset from path "{self.path}"\n{len(self.hr_files)} files with {len(self)} total frame slices\n{f"low-res: {self.hr_res}" if self.is_lr else f"high-res: {self.hr_res}, low-res: {self.hr_res//self.lr_scale}"}'
    
    def _get_name(self, idx):
        image_idx, idx = _get_image_idx(idx, self.slices, self.tiles)
        return f"{self.hr_files[image_idx].split('.')[0]}_{idx//self.slices[image_idx]}_{idx%self.slices[image_idx]}"

class PairedImageDataset(Dataset):
    def __init__(self, hr_path : Path, lr_path : Path, hr_res : int = 512, lr_scale : int = 4, n_frames : list[int] = -1, extension : str = "tif", val_split : float = 1, rotation : bool = True, split_seed : int = None, transforms : list[torch.nn.Module] = None):
        r"""Testing dataset for loading paired high-low-resolution images without using crappification. Can also be used for approximating :class:`Crappifier` parameters.

        Args:
            hr_path (Path) : Path to folder containing high-resolution images. Can also be a str.

            lr_path (Path) : Path to folder containing low-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            n_frames (list[int]) : Amount of stacked frames per image. Can also be list of low-resolution and high-resolution stack amounts respectively. A value of -1 uses all stacked image frames. Default is -1.

            extension (str) : File extension of images. Default is "tif".

            val_split (float) : Proportion of images to be held out for evaluation/prediction. Default is 1.

            rotation (bool) : Whether to randomly rotate and/or flip images when loading data. Only applicable during training. Default is True.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is None.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.hr_path = Path(hr_path) if type(hr_path) is str else hr_path
        self.lr_path = Path(lr_path) if type(lr_path) is str else lr_path
        for path in [self.hr_path, self.lr_path]:
            if not path or not path.exists(): raise FileNotFoundError(f'Path "{path}" does not exist.')
        if self.hr_path == self.lr_path: warnings.warn("hr_path is equal to lr_path! Consider using ImageDataset instead.", stacklevel=2)

        self.hr_files = _root_glob(f"*.{extension}", root_dir=self.hr_path)
        self.lr_files = _root_glob(f"*.{extension}", root_dir=self.lr_path)
        for files, path in zip([self.hr_files, self.lr_files], [self.hr_path, self.lr_path]):
            if not len(files) > 0: raise FileNotFoundError(f'No .{extension} files exist in path "{path}".')
        if len(self.hr_files) != len(self.lr_files): raise FileNotFoundError(f"Mismatch between amounts of high-low-resolution images. Found {len(self.hr_files)} high-resolution and {len(self.lr_files)} low-resolution images.")

        self.mode = "L"
        self.n_frames = _get_n_frames(n_frames)

        self.slices, max_size = [], 0
        for image_idx in range(len(self.hr_files)):
            image = Image.open(Path(self.hr_path, self.hr_files[image_idx]))
            self.slices.append(1 if self.n_frames is None else image.n_frames // max(self.n_frames))
            max_size = max(max(image.size), max_size)

        self.val_idx = _get_val_idx(self.slices, val_split, split_seed)
        self.is_lr = False
        self.crop_res = min(hr_res, max_size)
        self.extra_hr_files = None

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.rotation = rotation
        self.transforms = transforms
    
    def __getitem__(self, idx, pp=False):
        if idx >= len(self): raise IndexError(f"Tried to retrieve invalid image. Index {idx} is not less than {len(self)} total image frame slices.")

        is_val = idx in self.val_idx or pp
        image_idx, idx = _get_image_idx(idx, self.slices)

        hr = _load_image(self.hr_path, self.hr_files[image_idx], self.mode, self.n_frames[1] if self.n_frames is not None else None, self.slices[image_idx], idx)
        lr = _load_image(self.lr_path, self.lr_files[image_idx], self.mode, self.n_frames[0] if self.n_frames is not None else None, self.slices[image_idx], idx)

        cur_rot = [bool(random.getrandbits(1)), random.choice((1,2,(1,2)))] if self.rotation and not is_val else False

        return _transform_pair(hr, lr, self.hr_res, self.hr_res//self.lr_scale, cur_rot, self.transforms, self.n_frames)
    
    def __len__(self):
        return sum(self.slices)
    
    def __repr__(self):
        return f'PairedImageDataset from paths "{self.hr_path}" and "{self.lr_path}"\n{len(self.hr_files)} paired files with {len(self)} total frame slices\nhigh-res: {self.hr_res}, low-res: {self.hr_res//self.lr_scale}'

    def _get_name(self, idx):
        image_idx, idx = _get_image_idx(idx, self.slices)
        return self.lr_files[image_idx].split('.')[0] + (f"_{idx}" if self.n_frames is not None else "")

class PairedSlidingDataset(Dataset):
    def __init__(self, hr_path : Path, lr_path : Path, hr_res : int = 512, lr_scale : int = 4, overlap : int = 128, n_frames : list[int] = -1, slide : bool = False, stack : str = "TZ", extension : str = "czi", preload : bool = True, val_split : float = 1, rotation : bool = True, split_seed : int = None, transforms : list[torch.nn.Module] = None):
        r"""Testing dataset for loading high-low-resolution image tiles from image sheets without crappification. Can also be used for approximating :class:`Crappifier` parameters.

        Dataset used for image sheets (e.g. .czi files). For pre-tiled image files, use :class:`ImageDataset`.

        Args:
            hr_path (Path) : Path to folder containing high-resolution images. Can also be a str.

            lr_path (Path) : Path to folder containing low-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            overlap (int) : Overlapping pixels between neighboring tiles to increase effective dataset size. Default is 128.

            n_frames (list[int]) : Amount of stacked frames per image tile. Can also be list of low-resolution and high-resolution stack amounts respectively. A value of -1 uses all stacked image frames. Default is -1.

            slide (bool) : Whether to slide over stack dimensions rather than taking discrete non-overlapping slices, increasing the effective size of the dataset. Should not be used if more than one dimension is stacked. Default is False.

            stack (str) : Multiframe stack handling mode, e.g "T" for time stack, "Z" for z dimension stack, "TZ" or "ZT" for both, determining flattenting order. Only applicable if loading from czi. Default is "TZ".

            extension (str) : File extension of images. Default is "czi".

            preload (bool) : Whether to preload images in memory (not VRAM) for faster dataloading. Default is True.

            val_split (float) : Proportion of images to be held out for evaluation/prediction. Default is 1.

            rotation (bool) : Whether to randomly rotate and/or flip images when loading data. Only applicable during training. Default is True.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is None.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.hr_path = Path(hr_path) if type(hr_path) is str else hr_path
        self.lr_path = Path(lr_path) if type(lr_path) is str else lr_path
        for path in [self.hr_path, self.lr_path]:
            if not path or not path.exists(): raise FileNotFoundError(f'Path "{path}" does not exist.')
        if self.hr_path == self.lr_path: warnings.warn("hr_path is equal to lr_path! Consider using SlidingDataset instead.", stacklevel=2)
        
        self.hr_files = _root_glob(f"*.{extension}", root_dir=self.hr_path)
        self.lr_files = _root_glob(f"*.{extension}", root_dir=self.lr_path)
        for files, path in zip([self.hr_files, self.lr_files], [self.hr_path, self.lr_path]):
            if not len(files) > 0: raise FileNotFoundError(f'No .{extension} files exist in path "{path}".')
        if len(self.hr_files) != len(self.lr_files): raise FileNotFoundError(f"Mismatch between amounts of high-low-resolution images. Found {len(self.hr_files)} high-resolution and {len(self.lr_files)} low-resolution images.")

        overlap = 0 if overlap is None else overlap
        if not hr_res > overlap: raise ValueError(f"hr_res must be greater than overlap. Given values are {hr_res} and {overlap} respectively.")
        self.stride = hr_res - overlap
        self.stack = stack.upper()
        self.mode = "L"
        self.n_frames = _get_n_frames(n_frames)
        self.slide = slide

        self.preload = _preload(preload, [self.hr_path, self.lr_path], [self.hr_files, self.lr_files], self.mode, self.stack)

        self.tiles, self.slices = [], []
        for image_idx in range(len(self.hr_files)):
            image = self.preload[0][image_idx] if self.preload else _load_sheet(self.hr_path, self.hr_files[image_idx], self.stack, self.mode)
            tiles_x, tiles_y = _n_tiles(image, hr_res, self.stride)
            self.tiles.append(tiles_x * tiles_y)
            self.slices.append(1 if self.n_frames is None else ((image.shape[0] - max(self.n_frames) + 1) if slide else (image.shape[0] // max(self.n_frames))))

        self.val_idx = _get_val_idx(self.slices, val_split, split_seed, self.tiles)
        self.is_lr = False
        self.crop_res = hr_res
        self.extra_hr_files = None

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.rotation = rotation
        self.transforms = transforms
    
    def __getitem__(self, idx, pp=False):
        if idx >= len(self): raise IndexError(f"Tried to retrieve invalid image. Index {idx} is not less than {len(self)} total image frame slices.")

        is_val = idx in self.val_idx or pp
        image_idx, idx = _get_image_idx(idx, self.slices, self.tiles)

        hr = _sliding_window(self.preload[0][image_idx] if self.preload else _load_sheet(self.hr_path, self.hr_files[image_idx], self.stack, self.mode), self.hr_res, self.stride, self.n_frames[1] if self.n_frames is not None else None, self.slices[image_idx], idx, self.slide)
        lr = _sliding_window(self.preload[1][image_idx] if self.preload else _load_sheet(self.lr_path, self.lr_files[image_idx], self.stack, self.mode), self.hr_res//self.lr_scale, self.stride//self.lr_scale, self.n_frames[0] if self.n_frames is not None else None, self.slices[image_idx], idx, self.slide)

        cur_rot = [bool(random.getrandbits(1)), random.choice((1,2,(1,2)))] if self.rotation and not is_val else False

        return _transform_pair(hr, lr, self.hr_res, self.hr_res//self.lr_scale, cur_rot, self.transforms, self.n_frames)
    
    def __len__(self):
        return sum([self.tiles[idx] * self.slices[idx] for idx in range(len(self.hr_files))])
    
    def __repr__(self):
        return f'PairedSlidingDataset from paths "{self.hr_path}" and "{self.lr_path}"\n{len(self.hr_files)} paired files with {len(self)} total frame slices\nhigh-res: {self.hr_res}, low-res: {self.hr_res//self.lr_scale}'
    
    def _get_name(self, idx):
        image_idx, idx = _get_image_idx(idx, self.slices, self.tiles)
        return f"{self.lr_files[image_idx].split('.')[0]}_{idx//self.slices[image_idx]}_{idx%self.slices[image_idx]}"

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
        hr = np.rot90(hr, axes=(1,2)) if rotation[0] else hr
        hr = np.flip(hr, axis=rotation[1])

    # Crappification
    lr = np.stack([Image.fromarray(channel).resize(([hr_res//lr_scale]*2), Image.Resampling.BILINEAR) for channel in hr])
    if crappifier is not None:
        # Allows either Crappifier or nn.Module (or any callable function) to be used as a crappifier
        lr = crappifier.crappify(lr) if issubclass(type(crappifier), Crappifier) else crappifier(lr)
        lr = np.clip(lr.round(), 0, 255)

    if n_frames is not None and n_frames[0] != n_frames[1]:
        if not n_frames[1] > hr.shape[-3]:
            hr = _slice_center(hr, n_frames[1])
        if not n_frames[0] > lr.shape[-3]:
            lr = _slice_center(lr, n_frames[0])

    return _tensor_ready(hr, transforms), _tensor_ready(lr, transforms)

def _transform_pair(hr, lr, hr_res, lr_res, rotation, transforms, n_frames):
    r"""Same as _gen_pair, but uses paired high-low-resolution images with no crappifier.
    """
    hr = _square_crop(hr, hr_res)
    lr = _square_crop(lr, lr_res)

    hr = _pad_image(hr, hr_res)
    lr = _pad_image(lr, lr_res)

    if rotation:
        hr, lr = np.rot90(hr, axes=(1,2)) if rotation[0] else hr, np.rot90(lr, axes=(1,2)) if rotation[0] else lr
        hr, lr = np.flip(hr, axis=rotation[1]), np.flip(lr, axis=rotation[1])
    
    if n_frames is not None and n_frames[0] != n_frames[1]:
        if not n_frames[1] > hr.shape[-3]:
            hr = _slice_center(hr, n_frames[1])
        if not n_frames[0] > lr.shape[-3]:
            lr = _slice_center(lr, n_frames[0])

    return _tensor_ready(hr, transforms), _tensor_ready(lr, transforms)

def _ready_lr(lr, lr_res, transforms):
    r"""Processes lr images for prediction. Same as transform pair but without rotation.
    """
    lr = _square_crop(lr, lr_res)
    lr = _pad_image(lr, lr_res)

    return _tensor_ready(lr, transforms)

def _tensor_ready(image, transforms):
    image = torch.tensor(image.copy().astype(np.float32), dtype=torch.float)

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

    print(f"Preloading {sum([len(item) for item in files])} image sheets into memory...")
    preload = [[_load_sheet(idx_path, file, stack, mode) for file in idx_files] for idx_path, idx_files in zip(path, files)]
    return preload[0] if len(preload) == 1 else preload

def _load_image(path, file, mode, n_frames, slices, idx):
    extension = file.split(".")[-1].lower()
    if extension in ("tif", "tiff"):
        image = tifffile.imread(Path(path, file))
        if len(image.shape) < 3:
            image = image[np.newaxis]
    else:
        image = Image.open(Path(path, file))
        image = _frame_channel(image, mode)

    # TODO: Allow slide for ImageDataset?
    return _slice_image(image, n_frames, slices, idx, slide=False)

def _load_sheet(path, file, stack, mode):
    extension = file.split(".")[-1].lower()
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
        if stack == "T":
            image = image[:,0]
        elif stack == "Z":
            image = image[0]
        elif stack == "ZT":
            image = np.moveaxis(image, 0, 1)
        elif stack == "TZ":
            pass
        else:
            raise ValueError(f"Stack type {stack} is not valid.")
        
        # Flatten channel dimensions
        image = np.reshape(image, [-1, image.shape[-2], image.shape[-1]])
        if image.max() != 0:
            image = image / (image.max() / 255)
        return image.astype(np.uint8)
    elif extension in ("tif", "tiff"):
        image = tifffile.imread(Path(path, file))
        if len(image.shape) < 3:
            image = image[np.newaxis]
        return image
    else:
        image = Image.open(Path(path, file))
        return _frame_channel(image, mode)

def _sliding_window(image, size, stride, n_frames, n_slices, idx, slide):
    tiles_x, tiles_y = _n_tiles(image, size, stride)
    tile_idx = idx // n_slices

    start_x = tile_idx // tiles_y * stride
    start_y = tile_idx % tiles_y * stride

    image = image[..., start_x:start_x + size, start_y:start_y + size]

    return _slice_image(image, n_frames, n_slices, idx, slide)

def _frame_channel(image, mode = "L"):
    # Create frame dimension
    if image.n_frames > 1:
        image = np.stack([np.asarray(_seek_channel(image, frame).convert(mode), dtype=np.uint8) for frame in range(image.n_frames)])
    else:
        image = np.asarray(image.convert(mode), dtype=np.uint8)[np.newaxis, :, :]
    
    return image

def _slice_image(image, n_frames, n_slices, idx, slide):
    if n_frames == None:
        return image
    
    # Only need to calculate residual idx of specific tile
    if slide:
        idx = idx % n_slices
    else:
        idx = idx % n_slices
        idx *= n_frames

    return image[idx:idx+n_frames]

def _slice_center(image, n_frames):
    center = image.shape[-3] // 2
    half = n_frames // 2
    if n_frames % 2 == 0:
        return image[...,center - half:center + half,:,:]
    else:
        return image[...,center - half:center + half + 1,:,:]

def _seek_channel(image, idx):
    # Because we can't have nice things...
    image.seek(idx)
    return image

def _root_glob(search, root_dir, recursive : bool = True):
    if recursive:
        files = glob.glob(f"{root_dir}/**/{search}", recursive=True)
    else:
        files = glob.glob(f"{root_dir}/{search}")
    return sorted([item.split(str(root_dir))[-1].strip("/") for item in files])

def _n_tiles(image, size, stride):
    x, y = image.shape[-2:]

    tiles_x = max(0, (x - size) // stride + 1)
    tiles_y = max(0, (y - size) // stride + 1)
    return tiles_x, tiles_y

def _get_n_frames(n_frames):
    if n_frames in [None, -1, [-1]]:
        return None
    
    n_frames = _force_list(n_frames)
    n_frames = n_frames*2 if len(n_frames) == 1 else n_frames
    return n_frames # [in (low-resolution), out (high-resolution)]

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
