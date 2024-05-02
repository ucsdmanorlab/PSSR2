import torch, tifffile, os
import numpy as np
from pathlib import Path

HR_RES = 512
LR_RES = 128
CROP_RES = 500

def get_shape(res : int, channels : int = 1, batch : int = 2):
    return (batch, channels, res, res) if batch > 0 else (channels, res, res)

def get_image(shape : tuple[int], tensor : bool = False):
    image = np.random.rand(*shape) * 255
    return torch.as_tensor(image, dtype=torch.float) if tensor else image

def make_tifs(path : Path, shape : tuple[int]):
    os.makedirs(path, exist_ok=True)

    shape = (1, *shape) if len(shape) <= 3 else shape
    images = np.random.rand(*shape) * 255

    for idx, image in enumerate(images):
        tifffile.imwrite(f"{path}/temp_tif_{idx}.tif", image.squeeze().astype(np.uint8))
