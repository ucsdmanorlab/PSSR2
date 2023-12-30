import torch, glob, random
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from aicsimageio import AICSImage
from .crappifiers import Crappifier, AdditiveGaussian
    
class ImageDataset(Dataset):
    def __init__(self, path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = AdditiveGaussian(), extension : str = "tif", mode : str = "L", rotation : bool = True, shuffle : bool = True, transforms : list[torch.nn.Module] = None):
        r"""Training dataset for loading high resolution images from individual files and returning high and low resolution pairs, the latter receiving crappification.

        Args:
            path (Path) : Path to folder containing high resolution images. Can also be a str.

            hr_res (int) : Resolution of high resolution images. Images larger than this will be downscaled to this resolution. Default is 512.

            lr_scale (int) : Downscaling factor for low resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            crappifier (Crappifier) : Crappifier for degrading low resolution images to simulate undersampling. Default is :class:`AdditiveGaussian`.

            extension (str) : File extension of images. Default is "tif".

            mode (str) : PIL image mode for loading images, e.g. "L" for grayscale, "RGB" for color. Default is "L".

            rotation (bool) : Whether to randomly rotate images when loading data. Default is true.

            shuffle (bool) : Whether to shuffle the order of images when loading data. Default is true.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        path = Path(path) if type(path) is str else path
        assert path.exists(), "Data does not exist at given path."

        self.hr_files = sorted(glob.glob(f"*.{extension}", root_dir=path))
        if shuffle:
            random.shuffle(self.hr_files)

        self.path = path
        self.hr_res = hr_res
        self.lr_scale = lr_scale
        self.crappifier = crappifier
        self.mode = mode
        self.rotation = rotation
        self.transforms = transforms

    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr = Image.open(Path(self.path, self.hr_files[idx]))

        hr, lr = _gen_pair(hr, self.hr_res, self.lr_scale, self.rotation, self.crappifier, self.mode, self.transforms)

        return hr, lr

class SlidingDataset(Dataset):
    def __init__(self, path : Path, hr_res : int = 512, lr_scale : int = 4, crappifier : Crappifier = AdditiveGaussian(), extension : str = "czi", overlap : int = 32, rotation : bool = True, shuffle : bool = True, transforms : list[torch.nn.Module] = None):
        r"""Training dataset for loading high resolution image tiles from an image sheet and returning high and low resolution pairs, the latter receiving crappification.

        Args:
            path (Path) : Path to folder containing high resolution images. Can also be a str.

            hr_res (int) : Resolution of high resolution images. Images larger than this will be downscaled to this resolution. Default is 512.

            lr_scale (int) : Downscaling factor for low resolution images to simulate undersampling. Choose a power of 2 for best results. Default is 4.

            crappifier (Crappifier) : Crappifier for degrading low resolution images to simulate undersampling. Default is :class:`AdditiveGaussian`.

            extension (str) : File extension of images. Default is "czi".

            overlap (int) : Overlapping pixels between neighboring tiles to increase effective dataset size. Default is 32.

            rotation (bool) : Whether to randomly rotate images when loading data. Default is true.

            shuffle (bool) : Whether to shuffle the order of images when loading data. Default is true.

            transforms (list[nn.Module]) : Additional final data transforms to apply. Default is None.
        """
        super().__init__()
        path = Path(path) if type(path) is str else path
        assert path.exists(), "Data does not exist at given path."
        self.path = path
        
        hr_files = sorted(glob.glob(f"*.{extension}", root_dir=path))
        if shuffle:
            random.shuffle(hr_files)

        if extension == "czi":
            self.hr_images = [_remove_empty_z(AICSImage(Path(path, file)).get_image_data("CXYZ", T=0)).astype(np.uint8) for file in hr_files]
        else:
            try:
                # TODO
                self.hr_images = []
            except:
                raise ValueError(f"File type {extension} not supported.")

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
        
    def __len__(self):
        return self.total_tiles
    
    def __getitem__(self, idx):
        assert idx < self.total_tiles, "Tried to retrieve invalid tile."
        
        image_idx = 0
        while type(result := _sliding_window(self.hr_images[image_idx], self.hr_res, self.stride, idx)) is int:
            # If idx is beyond current image, try next image
            idx -= result
            image_idx += 1

        return _gen_pair(result, self.hr_res, self.lr_scale, self.rotation, self.mode, self.crappifier, self.transforms)

def _gen_pair(hr, hr_res, lr_scale, rotation, crappifier, mode, transforms):
    hr = _square_crop(hr).resize(([hr_res]*2), Image.Resampling.NEAREST)
    
    if rotation:
        # Set random rotation and flip in xy axis
        hr = np.rot90(hr, axes=(0,1)) if bool(random.getrandbits(1)) else hr
        hr = Image.fromarray(np.flip(hr, axis=random.choice((2,0,1,(0,1)))))

    # Crappification
    lr = np.asarray(hr.resize([hr_res//lr_scale]*2, Image.Resampling.NEAREST))
    if crappifier is not None:
        lr = crappifier.crappify(lr)

    # nn.Module ready axes
    hr, lr = np.squeeze(hr.convert(mode)), np.squeeze(Image.fromarray(lr.astype(np.uint8)).convert(mode))
    if len(hr.shape) < 3:
        hr, lr = hr[np.newaxis, :, :], lr[np.newaxis, :, :]
    else:
        hr, lr = np.moveaxis(hr, -1, 0), np.moveaxis(lr, -1, 0)
    
    hr = torch.tensor(hr.copy(), dtype=torch.float)
    lr = torch.tensor(lr.copy(), dtype=torch.float)

    # Additional nn.Module user transforms
    if transforms is not None:
        for transform in transforms:
            hr, lr = transform(hr), transform(lr)
    
    return hr, lr

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
