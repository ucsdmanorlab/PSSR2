import torch, glob, os, random
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from aicsimageio import AICSImage
from .crappifiers import Crappifier, Poisson

# TODO: aicsimageio is large, substitute?

class ImageDataset(Dataset):
    def __init__(self, path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = Poisson(), val_split : float = 0.1, extension : str = "tif", mode : str = "L", rotation : bool = True, transforms : list[torch.nn.Module] = None):
        r"""Training dataset for loading high-resolution images from individual files and returning high-low-resolution pairs, the latter receiving crappification.

        Dataset used for pre-tiled image files. For image sheets (e.g. .czi files), use :class:`SlidingDataset`.

        LR mode for predictions can be enabled by either inputting images of LR size or by setting `lr_scale` = 1 and `hr_res` = LR resolution.

        Args:
            path (Path) : Path to folder containing high resolution images. Can also be a str.

            hr_res (int) : Resolution of high resolution images. Images larger than this will be downscaled to this resolution. Default is 512.

            lr_scale (int) : Downscaling factor for low resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            crappifier (Crappifier) : Crappifier for degrading low resolution images to simulate undersampling. Default is :class:`Poisson`.

            val_split (float) : Fraction of images to be held out for validation. Default is 0.1.

            extension (str) : File extension of images. Default is "tif".

            mode (str) : PIL image mode for loading images, e.g. "L" for grayscale, "RGB" for color. Default is "L".

            rotation (bool) : Whether to randomly rotate images when loading data. Default is True.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        path = Path(path) if type(path) is str else path
        assert path.exists(), "Data does not exist at given path."

        self.hr_files = sorted(glob.glob(f"*.{extension}", root_dir=path))
        # if shuffle:
        #     random.shuffle(self.hr_files)

        self.is_lr = (hr_res//lr_scale) in Image.open(Path(path, self.hr_files[0])).size or lr_scale == 1

        self.path = path
        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.crappifier = crappifier
        self.mode = mode
        self.rotation = rotation
        self.transforms = transforms

        self.val_len = len(self.hr_files) if self.is_lr else int(val_split*len(self.hr_files))
        self.train_len = len(self.hr_files)-self.val_len

    def __len__(self):
        return self.train_len
    
    def __getitem__(self, idx):
        hr = Image.open(Path(self.path, self.hr_files[idx]))

        if self.is_lr:
            # rotation and crappifier is disabled in lr mode
            return _ready_lr(hr, self.hr_res//self.lr_scale, self.mode, self.transforms)

        return _gen_pair(hr, self.hr_res, self.lr_scale, self.rotation, self.crappifier, self.mode, self.transforms)

class SlidingDataset(Dataset):
    def __init__(self, path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = Poisson(), val_split : float = 0.1, extension : str = "czi", overlap : int = 64, rotation : bool = True, transforms : list[torch.nn.Module] = None):
        r"""Training dataset for loading high-resolution image tiles from image sheets and returning high-low-resolution pairs, the latter receiving crappification.

        Dataset used for image sheets (e.g. .czi files). For pre-tiled image files, use :class:`ImageDataset`.

        LR mode for predictions can be enabled by setting `lr_scale` = 1 and `hr_res` = LR resolution.

        Args:
            path (Path) : Path to folder containing high resolution images. Can also be a str.

            hr_res (int) : Resolution of high resolution images. Images larger than this will be downscaled to this resolution. Default is 512.

            lr_scale (int) : Downscaling factor for low resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            crappifier (Crappifier) : Crappifier for degrading low resolution images to simulate undersampling. Default is :class:`Poisson`.

            val_split (float) : Fraction of images to be held out for validation. Default is 0.1.

            extension (str) : File extension of images. Default is "czi".

            overlap (int) : Overlapping pixels between neighboring tiles to increase effective dataset size. Default is 64.

            rotation (bool) : Whether to randomly rotate images when loading data. Default is True.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        path = Path(path) if type(path) is str else path
        assert path.exists(), "Data does not exist at given path."
        self.path = path
        
        hr_files = sorted(glob.glob(f"*.{extension}", root_dir=path))

        if extension == "czi":
            self.hr_images = [_remove_empty_z(AICSImage(Path(path, file)).get_image_data("CXYZ", T=0)).astype(np.uint8) for file in hr_files]
        else:
            try:
                # TODO
                self.hr_images = [np.asarray(Image.open(Path(path, file)), dtype=np.uint8) for file in hr_files]
            except:
                raise ValueError(f"File type {extension} not supported.")
        
        self.is_lr = lr_scale == 1

        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.crappifier = crappifier
        self.stride = hr_res - overlap
        self.rotation = rotation
        self.transforms = transforms

        self.total_tiles = 0
        for image in self.hr_images:
            tiles_x, tiles_y = _n_tiles(image, self.hr_res, self.stride)
            self.total_tiles += tiles_x * tiles_y

        self.val_len = self.total_tiles if self.is_lr else int(val_split*self.total_tiles)
        self.train_len = self.total_tiles-self.val_len
        
    def __len__(self):
        return self.train_len
    
    def __getitem__(self, idx):
        assert idx < self.total_tiles, "Tried to retrieve invalid tile."
        
        # TODO: Possible optimization?
        image_idx = 0
        while type(result := _sliding_window(self.hr_images[image_idx], self.hr_res, self.stride, idx)) is int:
            # If idx is beyond current image, try next image
            idx -= result
            image_idx += 1
        
        if self.is_lr:
            # rotation and crappifier are disabled in lr mode
            return _ready_lr(result, self.hr_res, self.mode, self.transforms)

        return _gen_pair(result, self.hr_res, self.lr_scale, self.rotation, self.mode, self.crappifier, self.transforms)
    
class PairedImageDataset(Dataset):
    def __init__(self, hr_path : Path, lr_path : Path, hr_res : int = 512, lr_scale : int = 4, val_split : float = 1, extension : str = "tif", mode : str = "L", rotation : bool = False, transforms : list[torch.nn.Module] = None):
        r"""Training dataset for loading paired high-low-resolution images without using crappification. Can be used for approximating :class:`Crappifier` parameters.

        Args:
            hr_path (Path) : Path to folder containing high resolution images. Can also be a str.

            lr_path (Path) : Path to folder containing low resolution images. Can also be a str.

            hr_res (int) : Resolution of high resolution images. Images larger than this will be downscaled to this resolution. Default is 512.

            lr_scale (int) : Downscaling factor for low resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            val_split (float) : Fraction of images to be held out for validation. Default is 1.

            extension (str) : File extension of images. Default is "tif".

            mode (str) : PIL image mode for loading images, e.g. "L" for grayscale, "RGB" for color. Default is "L".

            rotation (bool) : Whether to randomly rotate images when loading data. Default is False.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        hr_path = Path(hr_path) if type(hr_path) is str else hr_path
        lr_path = Path(lr_path) if type(lr_path) is str else lr_path
        assert hr_path.exists() and lr_path.exists(), "Data does not exist at given path."

        self.hr_files = sorted(glob.glob(f"*.{extension}", root_dir=hr_path))
        self.lr_files = sorted(glob.glob(f"*.{extension}", root_dir=lr_path))
        assert len(self.hr_files) == len(self.lr_files), "Length mismatch between high and low resolution images."

        # if shuffle:
        #     zipped = list(zip(self.hr_files, self.lr_files))
        #     random.shuffle(zipped)
        #     self.hr_files, self.lr_files = zip(*zipped)

        self.hr_path = hr_path
        self.lr_path = lr_path
        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.mode = mode
        self.rotation = rotation
        self.transforms = transforms
        self.is_lr = False

        self.val_len = int(val_split*len(self.hr_files))
        self.train_len = len(self.hr_files)-self.val_len

    def __len__(self):
        return self.train_len
    
    def __getitem__(self, idx):
        hr = Image.open(Path(self.hr_path, self.hr_files[idx]))
        lr = Image.open(Path(self.lr_path, self.lr_files[idx]))

        return _transform_pair(hr, lr, self.hr_res, self.hr_res//self.lr_scale, self.rotation, self.mode, self.transforms)
    
def preprocess_hr(path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = Poisson(), extension : str = "tif", mode : str = "L"):
    r"""Preprocesses and crappifies low resolution images from high resolution images.
    However, it is better for most use cases to process data at runtime using a dataset such as :class:`ImageDataset` to save disk space at the cost of negligible CPU usage.

    Args:
        path (Path) : Path to folder containing high resolution images. Can also be a str.

        hr_res (int) : Resolution of high resolution images. Images larger than this will be downscaled to this resolution. Default is 512.

        lr_scale (int) : Downscaling factor for low resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

        crappifier (Crappifier) : Crappifier for degrading low resolution images to simulate undersampling. Default is :class:`Poisson`.

        extension (str) : File extension of images. Default is "tif".

        mode (str) : PIL image mode for loading images, e.g. "L" for grayscale, "RGB" for color. Default is "L".
    """
    path = Path(path) if type(path) is str else path
    assert path.exists(), "Data does not exist at given path."

    lr_path = Path(path.parent, "pp/lr")
    os.makedirs(lr_path, exist_ok=True)

    hr_files = sorted(glob.glob(f"*.{extension}", root_dir=path))

    differing = False
    if Image.open(Path(path, hr_files[0])).size != (hr_res, hr_res):
        print("High resolution image dimensions are not square and/or do not match hr_res, cropped hr files will be saved in addition to lr.")

        differing = True
        hr_path = Path(path.parent, "pp/hr")
        os.makedirs(hr_path, exist_ok=True)

    for file in hr_files:
        hr = Image.open(Path(path, file))

        hr, lr = _gen_pair(hr, hr_res, lr_scale, False, crappifier, mode, None)

        Image.fromarray(np.moveaxis(np.asarray(lr, dtype=np.uint8), 0, -1).squeeze()).save(Path(lr_path, file))
        if differing:
            Image.fromarray(np.moveaxis(np.asarray(hr, dtype=np.uint8), 0, -1).squeeze()).save(Path(hr_path, file))

def _gen_pair(hr, hr_res, lr_scale, rotation, crappifier, mode, transforms):
    r"""Creates training ready pair of images from a single high resolution image.
    """
    hr = _square_crop(hr).resize(([hr_res]*2), Image.Resampling.BILINEAR)
    
    if rotation:
        # Set random rotation and flip in xy axis
        hr = np.rot90(hr, axes=(0,1)) if bool(random.getrandbits(1)) else hr
        hr = Image.fromarray(np.flip(hr, axis=random.choice((0,1,(0,1)))))

    hr = hr.convert(mode)

    # Crappification
    lr = np.asarray(hr.resize([hr_res//lr_scale]*2, Image.Resampling.BILINEAR))
    if crappifier is not None:
        # Allows either Crappifier or nn.Module to be used as a crappifier
        lr = crappifier.crappify(lr) if issubclass(type(crappifier), Crappifier) else crappifier(lr)

    return _tensor_ready(hr, transforms), _tensor_ready(Image.fromarray(lr.astype(np.uint8)), transforms)
    # return _tensor_ready(hr, Image.fromarray(lr.astype(np.uint8)), transforms)

def _transform_pair(hr, lr, hr_res, lr_res, rotation, mode, transforms):
    r"""Same as _gen_pair, but uses paired high and low resolution images.
    """
    hr = _square_crop(hr).resize(([hr_res]*2), Image.Resampling.BILINEAR)
    lr = _square_crop(lr).resize(([lr_res]*2), Image.Resampling.BILINEAR)

    if rotation:
        choice = bool(random.getrandbits(1)), random.choice((0,1,(0,1)))

        hr, lr = np.rot90(hr, axes=(0,1)) if choice[0] else hr, np.rot90(lr, axes=(0,1)) if choice[0] else lr
        hr, lr = Image.fromarray(np.flip(hr, axis=choice[1])), Image.fromarray(np.flip(lr, axis=choice[1]))

    return _tensor_ready(hr.convert(mode), transforms), _tensor_ready(lr.convert(mode), transforms)
    # return _tensor_ready(hr.convert(mode), lr.convert(mode), transforms)

def _ready_lr(lr, lr_res, mode, transforms):
    r"""Processes lr images for prediction.
    """
    lr = _square_crop(lr).resize(([lr_res]*2), Image.Resampling.BILINEAR)

    return _tensor_ready(lr.convert(mode), transforms)

def _tensor_ready(image, transforms):
    r"""nn.Module ready axes.
    """
    # Channel first
    image = np.squeeze(image)
    if len(image.shape) < 3:
        image = image[np.newaxis, :, :]
    else:
        image = np.moveaxis(image, -1, 0)
    
    image = torch.tensor(image.copy(), dtype=torch.float)

    # Additional nn.Module user transforms
    if transforms is not None:
        for transform in transforms:
            image = transform(image)
    
    return image

# def _tensor_ready(hr, lr, transforms):
#     # nn.Module ready axes
#     hr, lr = np.squeeze(hr), np.squeeze(lr)
#     if len(hr.shape) < 3:
#         hr, lr = hr[np.newaxis, :, :], lr[np.newaxis, :, :]
#     else:
#         hr, lr = np.moveaxis(hr, -1, 0), np.moveaxis(lr, -1, 0)
    
#     hr = torch.tensor(hr.copy(), dtype=torch.float)
#     lr = torch.tensor(lr.copy(), dtype=torch.float)

#     # Additional nn.Module user transforms
#     if transforms is not None:
#         for transform in transforms:
#             hr, lr = transform(hr), transform(lr)
    
#     return hr, lr

def _square_crop(image : Image):
    width, height = image.size
    offset  = int(abs(height-width)/2)
    if width > height:
        image = image.crop([offset, 0, width-offset, height])
    elif width < height:
        image = image.crop([0, offset, width, height-offset])
    return image

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

def _remove_empty_z(image : np.ndarray):
    if image.shape[-1] == 1:
        return np.squeeze(image, axis=-1)
    return image
