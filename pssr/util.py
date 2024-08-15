import torch, math, inspect, glob, tifffile, os
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pytorch_msssim import SSIM, MS_SSIM
from skimage.transform import resize
from pathlib import Path
from PIL import Image

class SSIMLoss(nn.Module):
    def __init__(self, channels : int = 1, mix : float = .8, win_size : int = 11, win_sigma : float = 1.5, ms : bool = True, kwargs = None):
        r"""SSIM and MS-SSIM loss with Mix as detailed in Zhao et al., 2018.

        Args:
            channels (int) : Number of channels in image. Default is 1.

            mix (float) : Mix of SSIM loss in loss calculation. 1 is entirely SSIM, 0 is entirely L1 with Gaussian correction. Default is 0.8.

            win_size (int) : Size of Gaussian window. Must be odd. Default is 11.

            win_sigma (float) : Sigma for distribution of Gaussian window. Default is 1.5.

            ms (bool) : Whether to use MS-SSIM over basic SSIM. Default is True.

            kwargs (dict[str, Any]) : Keyword arguments for ``pytorch_msssim``.
        """
        super().__init__()

        kwargs = {} if kwargs is None else kwargs
        self.ssim = MS_SSIM(channel=channels, win_size=win_size, win_sigma=win_sigma, data_range=1, **kwargs) if ms else SSIM(channel=channels, win_size=win_size, win_sigma=win_sigma, data_range=1, **kwargs)

        if mix < 1:
            # Generate Gaussian window for L1 loss
            coords = torch.arange(win_size, dtype=torch.float) - win_size // 2

            gaussian = torch.exp(-(coords ** 2) / (2 * win_sigma ** 2))
            gaussian /= gaussian.sum()

            self.gaussian = torch.outer(gaussian, gaussian)[np.newaxis, np.newaxis, ...]

        self.channels = channels
        self.win_size = win_size
        self.mix = mix

    def forward(self, input, target):
        x = 1 - self.ssim(input, target)
        if self.mix < 1:
            # Combine SSIM with L1 loss with applied Gaussian window for elementwise multiplication against non-reduced L1 loss
            device = "cpu" if input.get_device() == -1 else input.get_device()
            l1 = F.conv2d(F.l1_loss(input, target, reduction="none"), self.gaussian.to(device), groups=self.channels, padding=(self.win_size-1)//2).mean()  # F.conv2d with Gaussian filter
            x = self.mix*x + (1-self.mix)*l1
        return x
    
def reassemble_sheets(pred_path : Path, lr_path : Path, lr_scale : int, overlap : int = 0, margin : int = 0, out_dir : str = "preds"):
    r"""Reassembles image sheets from tiled images created during prediction by sliding datasets.
    
    Args:
        pred_path (Path) : Path to predicted image tiles.

        lr_path (Path) : Path to low-resolution image sheets.

        lr_scale (int) : Upscaling factor of the predicted images.

        overlap (int) : Overlap between adjacent low-resolution images tiles. Should be the same value as dataset. Default is 0.

        margin (int) : Size of margins for overlapping inner image tiles. The generated image sheet doesn't include margin pixels.
        It is recommended to increase this value to reduce grid artifacts. Must be smaller than overlap. Default is 0.

        out_dir (str) : Directory to save images. A value of None returns images. Default is "preds".
    
    Returns:
        images (list[np.ndarray]) : Returns predicted images if ``out_dir`` is None.
    """
    # Prevents circular dependency
    from .data import _frame_channel

    if margin > overlap:
        raise ValueError(f"The value of margin cannot be greater than overlap. Given {margin} and {overlap} respectively.")

    sheet_files = glob.glob(f"{lr_path}/*.tif", recursive=True)
    # sheet_files = list(set(["_".join(file.split('/')[-1].split('.')[0].split("_")[:-2]) for file in glob.glob(f"{path}/*.tif", recursive=True)]))

    outs = []
    for sheet in sheet_files:
        files = sorted(glob.glob(f"{pred_path}/{sheet.split('/')[-1].split('.')[0]}*"), key=_sort_tiles)
        # files = sorted(glob.glob(f"{path}/{sheet}*"), key=_sort_tiles)
        batched = np.asarray([_frame_channel(Image.open(file)).squeeze() for file in files])

        lr_shape = _frame_channel(Image.open(sheet)).squeeze().shape
        
        # TODO: Check n_rows and n_cols are in correct order
        n_rows, n_cols = (lr_shape[1] * lr_scale - batched.shape[1]) // (batched.shape[1] - overlap * lr_scale) + 1, (lr_shape[2] * lr_scale - batched.shape[2]) // (batched.shape[2] - overlap * lr_scale) + 1

        outs.append(np.asarray([_patch_images(batched[idx*n_rows*n_cols:idx*n_rows*n_cols+n_rows*n_cols], n_cols, n_rows, overlap * lr_scale, margin) for idx in range(lr_shape[0])], dtype=np.uint8))

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for idx, sheet in enumerate(sheet_files):
            tifffile.imwrite(f"{out_dir}/{sheet.split('/')[-1].split('.')[0]}.tif", outs[idx])
    else:
        return outs

def _sort_tiles(name : str):
    parts = name.replace(".", "_").split("_")
    return int(parts[-2]), int(parts[-3])

def _patch_images(batched, n_cols, n_rows, overlap, margin):
    image_size = batched.shape[-1]
    step = image_size - overlap
    collage_height = n_rows * step + overlap
    collage_width = n_cols * step + overlap

    collage = np.zeros((collage_height, collage_width))
    count = np.zeros((collage_height, collage_width))

    for idx in range(n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        start_row = row * step
        start_col = col * step

        rel_margins = [(margin if row != 0 else 0), (margin if row != n_rows-1 else 0), (margin if col != 0 else 0), (margin if col != n_cols-1 else 0)]

        collage[start_row+rel_margins[0]:start_row+image_size-rel_margins[1], start_col+rel_margins[2]:start_col+image_size-rel_margins[3]] += batched[idx, rel_margins[0]:batched.shape[1]-rel_margins[1], rel_margins[2]:batched.shape[2]-rel_margins[3]]
        count[start_row+rel_margins[0]:start_row+image_size-rel_margins[1], start_col+rel_margins[2]:start_col+image_size-rel_margins[3]] += 1

    count[count == 0] = 1
    return collage / count

def normalize_preds(hr : np.ndarray, hr_hat : np.ndarray, pmin : float = 0.1, pmax : float = 99.9):
    r"""Normalizes prediction image intensities to ground truth for fair benchmarking.

    Args:
        hr (ndarray) : High-resolution ground truth images as array.

        hr_hat (ndarray) : High-resolution prediction images as array.

        pmin (float) : Percentile minimum image intensity. Default is 0.1.

        pmax (float) : Percentile maximum image intensity. Default is 99.9.
    
    Returns:
        hr_norm (ndarray) : Normalized high-resolution ground truth image.

        hr_hat_norm (ndarray) : Normalized high-resolution prediction image.
    """
    hr, hr_hat = np.asarray(hr), np.asarray(hr_hat)
    if len(hr.shape) != len(hr_hat.shape): raise ValueError(f"hr and hr_hat must have the same number of dimensions. Dimension lengths are {hr.shape} and {hr_hat.shape} respectively.")
    hr_shape, hr_hat_shape = hr.shape, hr_hat.shape

    if len(hr.shape) < 3:
        hr, hr_hat = hr[np.newaxis, ...], hr_hat[np.newaxis, ...]
    hr, hr_hat = hr.reshape(-1, *hr.shape[-2:]), hr_hat.reshape(-1, *hr_hat.shape[-2:])
    if len(hr) != len(hr_hat): raise ValueError(f"hr and hr_hat must have the same number of images. Received {len(hr)} and {len(hr_hat)} images respectively.")

    hr_norms, hr_hat_norms = [], []
    for idx in range(len(hr)):
        # Same procedure as in intial PSSR implementation
        hr_norm = hr[idx].astype(np.float32)
        hr_hat_norm = hr_hat[idx].astype(np.float32)

        base_max = np.percentile(hr_norm, pmax)
        base_mean = np.mean(hr_norm)

        hr_norm = _normalize_minmax(hr_norm, pmin, pmax)

        hr_hat_norm = hr_hat_norm - np.mean(hr_hat_norm)
        hr_norm = hr_norm - np.mean(hr_norm)

        scaled = resize(hr_hat_norm, hr_norm.shape) if hr_hat_norm.shape != hr_norm.shape else hr_hat_norm
        amp = np.cov(scaled.flatten(), hr_norm.flatten())[0, 1] / np.var(hr_hat_norm.flatten())
        hr_hat_norm = amp * hr_hat_norm
        
        # Rescale to initial image intensity
        hr_norm, hr_hat_norm = (hr_norm-hr_norm.min())*base_max, (hr_hat_norm-hr_norm.min())*base_max
        hr_norm, hr_hat_norm = hr_norm/(hr_norm.mean()/base_mean), hr_hat_norm/(hr_hat_norm.mean()/base_mean)

        hr_norms.append(hr_norm)
        hr_hat_norms.append(hr_hat_norm)
    
    hr, hr_hat = np.asarray(hr_norms).clip(0, 255), np.asarray(hr_hat_norms).clip(0, 255)
    return hr.reshape(hr_shape).astype(np.uint8), hr_hat.reshape(hr_hat_shape).astype(np.uint8)

def _normalize_minmax(x, pmin=0.1, pmax=99.9, eps=1e-20, dtype=np.float32):
    # From csbdeep
    x_min = np.percentile(x, pmin, keepdims=True)
    x_max = np.percentile(x, pmax, keepdims=True)

    x = x.astype(dtype,copy=False)
    x_min = dtype(x_min) if np.isscalar(x_min) else x_min.astype(dtype,copy=False)
    x_max = dtype(x_max) if np.isscalar(x_max) else x_max.astype(dtype,copy=False)
    eps = dtype(eps)

    x = (x - x_min) / (x_max - x_min + eps)

    return x

def pixel_metric(mse : float, image_range : int = 255):
    r"""Simple metric for calculating average pixel error.
    
    Args:
        mse (float) : Mean squared error between predicted and ground truth images.

        image_range (int) : Value range of image. Default is 255.
    """
    return math.sqrt(mse) * image_range

def _psnr_metric(mse : float):
    return 20 * torch.log10(1 / torch.sqrt(mse))

def _force_list(item):
    if type(item) is not list:
        try:
            return list(item)
        except:
            return [item]
    return item

def _get_callbacks(raw):
    callbacks = [] if raw is None else _force_list(raw)
    callback_locals = [len([arg for arg in inspect.getfullargspec(callback).args if arg != "self"]) == 1 for callback in callbacks]
    return callbacks, callback_locals

def _tab_string(text):
    lines = text.split("\n")
    indented_lines = ["\t" + line for line in lines]
    return "\n".join(indented_lines)
