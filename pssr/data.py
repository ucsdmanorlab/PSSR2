import torch, glob, os, random, czifile, warnings
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from .crappifiers import Crappifier, Poisson

class ImageDataset(Dataset):
    def __init__(self, path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = Poisson(), val_split : float = 0.1, extension : str = "tif", mode : str = "L", rotation : bool = True, split_seed : int = 0, transforms : list[torch.nn.Module] = None):
        r"""Training dataset for loading high-resolution images from individual files and returning high-low-resolution pairs, the latter receiving crappification.

        Dataset used for pre-tiled image files. For image sheets (e.g. .czi files), use :class:`SlidingDataset`.

        LR mode for predictions (dataset loads only unmodified low-resolution images) can be enabled by
        either inputting images less than or equal to LR size (`hr_res`/`lr_scale`) or by setting `lr_scale` = 1 and `hr_res` = LR resolution.

        Args:
            path (Path) : Path to folder containing high-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            crappifier (Crappifier) : Crappifier for degrading low-resolution images to simulate undersampling. Not used in LR mode. Default is :class:`Poisson`.

            val_split (float) : Fraction of images to be held out for evaluation. Default is 0.1.

            extension (str) : File extension of images. Default is "tif".

            mode (str) : PIL image mode for loading images, e.g. "L" for grayscale, "RGB" for color. Default is "L".

            rotation (bool) : Whether to randomly rotate images when loading data. Automatically set to False during evaluation if `val_split` is not less than 1, or in LR mode. Default is True.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is 0.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.path = Path(path) if type(path) is str else path
        assert self.path.exists(), f"Path {self.path} does not exist."

        self.hr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.path))
        assert len(self.hr_files) > 0, f"No {extension} files exist in {self.path}."

        self.val_idx = _get_val_idx(len(self.hr_files), val_split, split_seed)

        self.is_lr = all([size <= hr_res//lr_scale for size in Image.open(Path(self.path, self.hr_files[0])).size]) or (lr_scale == 1)
        self.crop_res = min(hr_res, max(Image.open(Path(self.path, self.hr_files[0])).size)) * (lr_scale if self.is_lr else 1)

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.crappifier = crappifier
        self.mode = mode
        self.rotation = rotation if val_split < 1 else False
        self.transforms = transforms

    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        assert idx < len(self), "Tried to retrieve invalid image."

        hr = Image.open(Path(self.path, self.hr_files[idx]))

        hr = _frame_channel(hr, self.mode)
        
        if self.is_lr:
            # Rotation and crappifier is disabled in lr mode
            return _ready_lr(hr, self.hr_res//self.lr_scale, self.transforms)

        return _gen_pair(hr, self.hr_res, self.lr_scale, self.rotation, self.crappifier, self.transforms)
    
    def _get_name(self, idx):
        return self.hr_files[idx].split('.')[0]

# TODO: Optimize dataloading
class SlidingDataset(Dataset):
    def __init__(self, path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = Poisson(), overlap : int = 128, val_split : float = 0.1, extension : str = "czi", mode : str = "L", rotation : bool = True, split_seed : int = 0, transforms : list[torch.nn.Module] = None):
        r"""Training dataset for loading high-resolution image tiles from image sheets and returning high-low-resolution pairs, the latter receiving crappification.

        Dataset used for image sheets (e.g. .czi files). For pre-tiled image files, use :class:`ImageDataset`.

        LR mode for predictions (dataset loads only unmodified low-resolution images) can be enabled by setting `lr_scale` = 1 and `hr_res` = LR resolution.

        Args:
            path (Path) : Path to folder containing high-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            crappifier (Crappifier) : Crappifier for degrading low-resolution images to simulate undersampling. Not used in LR mode. Default is :class:`Poisson`.

            overlap (int) : Overlapping pixels between neighboring tiles to increase effective dataset size. Default is 128.

            val_split (float) : Fraction of images to be held out for evaluation. Default is 0.1.

            extension (str) : File extension of images. Default is "czi".

            mode (str) : PIL image mode for loading images, e.g. "L" for grayscale, "RGB" for color. Not applicable if loading from czi. Default is "L".

            rotation (bool) : Whether to randomly rotate images when loading data. Automatically set to False during evaluation if `val_split` is not less than 1, or in LR mode. Default is True.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is 0.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.path = Path(path) if type(path) is str else path
        assert self.path.exists(), f"Path {self.path} does not exist."
        
        self.hr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.path))
        assert len(self.hr_files) > 0, f"No {extension} files exist in {self.path}."

        self.stride = hr_res - overlap
        self.tiles = []
        for file in self.hr_files:
            image = _load_sheet(self.path, file, mode)
            tiles_x, tiles_y = _n_tiles(image, hr_res, self.stride)
            self.tiles.append(tiles_x * tiles_y)

        self.val_idx = _get_val_idx(sum(self.tiles), val_split, split_seed)
        
        self.is_lr = (lr_scale == 1)
        self.crop_res = hr_res * (lr_scale if self.is_lr else 1)

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.crappifier = crappifier
        self.mode = mode
        self.rotation = rotation if val_split < 1 else False
        self.transforms = transforms
        
    def __len__(self):
        return sum(self.tiles)
    
    def __getitem__(self, idx):
        assert idx < len(self), "Tried to retrieve invalid tile."

        # Find idx tile across all tiles
        image_idx = 0
        for size in self.tiles:
            if idx < size:
                image = _sliding_window(_load_sheet(self.path, self.hr_files[image_idx], self.mode), self.hr_res, self.stride, idx)
                break
            else:
                idx -= size
                image_idx += 1

        if self.is_lr:
            return _ready_lr(image, self.hr_res, self.transforms)

        return _gen_pair(image, self.hr_res, self.lr_scale, self.rotation, self.crappifier, self.transforms)
    
    def _get_name(self, idx):
        image_idx = 0
        for size in self.tiles:
            if idx < size:
                break
            else:
                idx -= size
                image_idx += 1

        return f"{self.hr_files[image_idx].split('.')[0]}_{idx}"

class PairedImageDataset(Dataset):
    def __init__(self, hr_path : Path, lr_path : Path, hr_res : int = 512, lr_scale : int = 4, val_split : float = 1, extension : str = "tif", mode : str = "L", rotation : bool = False, split_seed : int = None, transforms : list[torch.nn.Module] = None):
        r"""Testing dataset for loading paired high-low-resolution images without using crappification. Can also be used for approximating :class:`Crappifier` parameters.

        Args:
            hr_path (Path) : Path to folder containing high-resolution images. Can also be a str.

            lr_path (Path) : Path to folder containing low-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            val_split (float) : Fraction of images to be held out for evaluation. Default is 1.

            extension (str) : File extension of images. Default is "tif".

            mode (str) : PIL image mode for loading images, e.g. "L" for grayscale, "RGB" for color. Default is "L".

            rotation (bool) : Whether to randomly rotate images when loading data. Default is False.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is None.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.hr_path = Path(hr_path) if type(hr_path) is str else hr_path
        self.lr_path = Path(lr_path) if type(lr_path) is str else lr_path
        assert self.hr_path.exists() and self.lr_path.exists(), f"One path {self.hr_path} or {self.lr_path} does not exist."
        if self.hr_path == self.lr_path:
            warnings.warn("hr_path is equal to lr_path, this should not be done except for testing purposes. Consider using ImageDataset instead.")

        self.hr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.hr_path))
        self.lr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.lr_path))
        for files, path in zip([self.hr_files, self.lr_files], [self.hr_path, self.lr_path]):
            assert len(files) > 0, f"No {extension} files exist in {path}."
        assert len(self.hr_files) == len(self.lr_files), "Length mismatch between high-low-resolution images."

        self.val_idx = _get_val_idx(len(self.hr_files), val_split, split_seed)

        self.is_lr = False
        self.crop_res = min(hr_res, max(Image.open(Path(self.hr_path, self.hr_files[0])).size))

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.mode = mode
        self.rotation = rotation
        self.transforms = transforms

    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        assert idx < len(self), "Tried to retrieve invalid image."

        hr = Image.open(Path(self.hr_path, self.hr_files[idx]))
        lr = Image.open(Path(self.lr_path, self.lr_files[idx]))

        hr = _frame_channel(hr, self.mode)
        lr = _frame_channel(lr, self.mode)

        return _transform_pair(hr, lr, self.hr_res, self.hr_res//self.lr_scale, self.rotation, self.transforms)

    def _get_name(self, idx):
        return self.lr_files[idx].split('.')[0]

class PairedSlidingDataset(Dataset):
    def __init__(self, hr_path : Path, lr_path : Path, hr_res : int = 512, lr_scale : int = 4, overlap : int = 128, val_split : float = 1, extension : str = "czi", mode : str = "L", rotation : bool = False, split_seed : int = None, transforms : list[torch.nn.Module] = None):
        r"""Testing dataset for loading high-low-resolution image tiles from image sheets without crappification. Can also be used for approximating :class:`Crappifier` parameters.

        Dataset used for image sheets (e.g. .czi files). For pre-tiled image files, use :class:`ImageDataset`.

        Args:
            hr_path (Path) : Path to folder containing high-resolution images. Can also be a str.

            lr_path (Path) : Path to folder containing low-resolution images. Can also be a str.

            hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Images smaller will be padded. Default is 512.

            lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            overlap (int) : Overlapping pixels between neighboring tiles to increase effective dataset size. Default is 128.

            val_split (float) : Fraction of images to be held out for evaluation. Default is 1.

            extension (str) : File extension of images. Default is "czi".

            mode (str) : PIL image mode for loading images, e.g. "L" for grayscale, "RGB" for color. Not applicable if loading from czi. Default is "L".

            rotation (bool) : Whether to randomly rotate images when loading data. Default is False.

            split_seed (int) : Seed for random train/evaluation data splitting. A value of None splits the last images as evaluation. Default is None.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        self.hr_path = Path(hr_path) if type(hr_path) is str else hr_path
        self.lr_path = Path(lr_path) if type(lr_path) is str else lr_path
        assert self.hr_path.exists() and self.lr_path.exists(), f"One path {self.hr_path} or {self.lr_path} does not exist."
        if self.hr_path == self.lr_path:
            warnings.warn("hr_path is equal to lr_path, this should not be done except for testing purposes. Consider using SlidingDataset instead.")
        
        self.hr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.hr_path))
        self.lr_files = sorted(glob.glob(f"*.{extension}", root_dir=self.lr_path))
        for files, path in zip([self.hr_files, self.lr_files], [self.hr_path, self.lr_path]):
            assert len(files) > 0, f"No {extension} files exist in {path}."
        assert len(self.hr_files) == len(self.lr_files), "Length mismatch between high-low-resolution images."

        self.stride = hr_res - overlap

        self.tiles = []
        for file in self.hr_files:
            image = _load_sheet(self.hr_path, file, mode)
            tiles_x, tiles_y = _n_tiles(image, hr_res, self.stride)
            self.tiles.append(tiles_x * tiles_y)

        self.val_idx = _get_val_idx(sum(self.tiles), val_split, split_seed)
        
        self.is_lr = False
        self.crop_res = hr_res

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.mode = mode
        self.rotation = rotation
        self.transforms = transforms

    def __len__(self):
        return sum(self.tiles)
    
    def __getitem__(self, idx):
        assert idx < len(self), "Tried to retrieve invalid tile."

        # Find idx tile across all tiles
        image_idx = 0
        for size in self.tiles:
            if idx < size:
                hr = _sliding_window(_load_sheet(self.hr_path, self.hr_files[image_idx], self.mode), self.hr_res, self.stride, idx)
                lr = _sliding_window(_load_sheet(self.lr_path, self.lr_files[image_idx], self.mode), self.hr_res//self.lr_scale, self.stride//self.lr_scale, idx)
                break
            else:
                idx -= size
                image_idx += 1

        return _transform_pair(hr, lr, self.hr_res, self.hr_res//self.lr_scale, self.rotation, self.transforms)
    
    def _get_name(self, idx):
        image_idx = 0
        for size in self.tiles:
            if idx < size:
                break
            else:
                idx -= size
                image_idx += 1

        return f"{self.lr_files[image_idx].split('.')[0]}_{idx}"

# TODO: preprocess_dataset
def preprocess_hr(path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = Poisson(), extension : str = "tif", mode : str = "L"):
    r"""Preprocesses and crappifies low-resolution images from high-resolution images.
    However, it is better for most use cases to process data at runtime using a dataset such as :class:`ImageDataset` to save disk space at the cost of negligible CPU usage.

    Args:
        path (Path) : Path to folder containing high-resolution images. Can also be a str.

        hr_res (int) : Resolution of high-resolution images. Images larger than this will be downscaled to this resolution. Default is 512.

        lr_scale (int) : Downscaling factor for low-resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

        crappifier (Crappifier) : Crappifier for degrading low-resolution images to simulate undersampling. Default is :class:`Poisson`.

        extension (str) : File extension of images. Default is "tif".

        mode (str) : PIL image mode for loading images, e.g. "L" for grayscale, "RGB" for color. Default is "L".
    """
    path = Path(path) if type(path) is str else path
    assert path.exists(), f"Path {path} does not exist."

    lr_path = Path(path.parent, "pp/lr")
    os.makedirs(lr_path, exist_ok=True)

    hr_files = sorted(glob.glob(f"*.{extension}", root_dir=path))

    if Image.open(Path(path, hr_files[0])).size != (hr_res, hr_res):
        print("High-resolution image dimensions are not square and/or do not match hr_res, cropped hr files will be saved in addition to lr.")

        differing = True
        hr_path = Path(path.parent, "pp/hr")
        os.makedirs(hr_path, exist_ok=True)
    else:
        differing = False

    for file in hr_files:
        hr = Image.open(Path(path, file))

        hr = _frame_channel(hr, mode)

        hr, lr = _gen_pair(hr, hr_res, lr_scale, False, crappifier, None)

        # TODO: Save multiframe images
        Image.fromarray(np.moveaxis(np.asarray(lr, dtype=np.uint8), 0, -1).squeeze()).save(Path(lr_path, file))
        if differing:
            Image.fromarray(np.moveaxis(np.asarray(hr, dtype=np.uint8), 0, -1).squeeze()).save(Path(hr_path, file))

def _gen_pair(hr, hr_res, lr_scale, rotation, crappifier, transforms):
    r"""Creates training ready pair of images from a single high-resolution image.
    """
    # Square crop
    hr = np.stack([_square_crop(channel, hr_res) for channel in hr])

    # Ensure image has hr_res
    if hr.shape[1] < hr_res:
        hr = np.stack([np.pad(channel, pad_width=[[0,hr_res-hr.shape[1]]]*2, mode="reflect") for channel in hr])
    # if hr.shape[1] > hr_res:
    #     hr = np.stack([Image.fromarray(channel).resize(([hr_res]*2), Image.Resampling.BILINEAR) for channel in hr])
    
    # Set random rotation and flip in xy axis
    if rotation:
        hr = np.rot90(hr, axes=(1,2)) if bool(random.getrandbits(1)) else hr
        hr = np.flip(hr, axis=random.choice((1,2,(1,2))))

    # Crappification
    lr = np.stack([Image.fromarray(channel).resize(([hr_res//lr_scale]*2), Image.Resampling.BILINEAR) for channel in hr])
    if crappifier is not None:
        # Allows either Crappifier or nn.Module to be used as a crappifier
        lr = crappifier.crappify(lr) if issubclass(type(crappifier), Crappifier) else crappifier(lr)
        lr = np.clip(lr.round(), 0, 255)

    return _tensor_ready(hr, transforms), _tensor_ready(lr, transforms)

def _transform_pair(hr, lr, hr_res, lr_res, rotation, transforms):
    r"""Same as _gen_pair, but uses paired high-low-resolution images with no crappifier.
    """
    hr = np.stack([_square_crop(channel, hr_res) for channel in hr])
    lr = np.stack([_square_crop(channel, lr_res) for channel in lr])

    if hr.shape[1] < hr_res:
        hr = np.stack([np.pad(channel, pad_width=[[0,hr_res-hr.shape[1]]]*2, mode="reflect") for channel in hr])
        lr = np.stack([np.pad(channel, pad_width=[[0,lr_res-lr.shape[1]]]*2, mode="reflect") for channel in lr])

    if rotation:
        choice = bool(random.getrandbits(1)), random.choice((0,1,(0,1)))

        hr, lr = np.rot90(hr, axes=(1,2)) if choice[0] else hr, np.rot90(lr, axes=(1,2)) if choice[0] else lr
        hr, lr = np.flip(hr, axis=choice[1]), np.flip(lr, axis=choice[1])

    return _tensor_ready(hr, transforms), _tensor_ready(lr, transforms)

def _ready_lr(lr, lr_res, transforms):
    r"""Processes lr images for prediction. Same as transform pair but without rotation.
    """
    lr = np.stack([_square_crop(channel, lr_res) for channel in lr])

    if lr.shape[1] < lr_res:
        lr = np.stack([np.pad(channel, pad_width=[[0,lr_res-lr.shape[1]]]*2, mode="reflect") for channel in lr])

    return _tensor_ready(lr, transforms)

def _frame_channel(image, mode):
    # Create frame dimension
    if image.n_frames > 1:
        image = np.stack([np.asarray(image.seek(frame).convert(mode), dtype=np.uint8) for frame in range(image.n_frames)])
    else:
        image = np.asarray(image.convert(mode), dtype=np.uint8)[np.newaxis, :, :]

    # Merge channel into frame dimension
    if len(image.shape) > 3:
        image = np.reshape(np.moveaxis(image, -1, 0), [-1, *image.shape[1:3]])
    
    return image

def _tensor_ready(image, transforms):
    r"""nn.Module ready axes.
    """
    image = torch.tensor(image.copy(), dtype=torch.float)

    # Additional nn.Module user transforms
    if transforms is not None:
        for transform in transforms:
            image = transform(image)
    
    return image

def _square_crop(image, max_res):
    if list(image.shape) == [max_res]*2:
        return image

    height, width = image.shape
    size = min(height, width, max_res)
    
    start_row = (height - size) // 2
    start_col = (width - size) // 2
    
    return image[start_row:start_row + size, start_col:start_col + size]

# TODO: T stacking option
def _load_sheet(path, file, mode):
    extension = file.split(".")[-1]
    if extension == "czi":
        image = czifile.imread(Path(path, file))[0,0,0,:,:,:,:,0] # Only CZYX, disregard T?
        image = np.reshape(np.moveaxis(image, [0,2], [1,3]), [-1, image.shape[3], image.shape[2]]) # CZYX to ZCXY to CXY
        if image.max() != 0:
            image = image / image.max() * 255
        return image.astype(np.uint8)
    else:
        image = Image.open(Path(path, file))
        return _frame_channel(image, mode)

def _sliding_window(image, size, stride, idx):
    tiles_x, tiles_y = _n_tiles(image, size, stride)
    if idx > tiles_x * tiles_y:
        return tiles_x * tiles_y

    start_x = idx // tiles_y * stride
    start_y = idx % tiles_y * stride

    return image[:, start_x:start_x + size, start_y:start_y + size]

def _n_tiles(image, size, stride):
    x, y = image.shape[1:]

    tiles_x = max(0, (x - size) // stride + 1)
    tiles_y = max(0, (y - size) // stride + 1)
    return tiles_x, tiles_y

def _get_val_idx(idx_len, split, seed):
    val_idx = list(range(idx_len))

    if seed is not None and split < 1:
        np.random.seed(seed)
        np.random.shuffle(val_idx)
    
    return val_idx[-int(split*idx_len):]

def _invert_idx(idx, idx_len):
    idx_range = np.arange(idx_len)
    inverse = np.logical_not(np.isin(idx_range, idx))
    return idx_range[inverse]

class _RandomIterIdx:
    def __init__(self, idx):
        self.idx = idx
        
    def __iter__(self):
        random_idx = self.idx.copy()
        random.shuffle(random_idx)
        yield from random_idx
    
    def __len__(self):
        return len(self.idx)
