import torch, os, tifffile
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from PIL import Image
from .data import _RandomIterIdx, _slice_center
from .util import _get_callbacks, pixel_metric, normalize_preds

def predict_images(model : nn.Module, dataset : Dataset, device : str = "cpu", norm : bool = False, prefix : str = None, out_dir : str = "preds", callbacks = None):
    r"""Predicts high-resolution images from low-resolution images using a given model.
    
    Only uses evaluation images if applicable. Set ``val_split=1`` in dataset to use all images.

    Args:
        model (nn.Module) : Model to recieve low-resolution images.

        dataset (Dataset) : Dataset to load low-resolution images from.

        device (str) : Device to train model on. Default is "cpu".

        norm (bool) : Whether to normalize prediction image intensities to ground truth, which must be provided by a paired dataset. Default is False.

        prefix (str) : Prefix to append at the beginning the output file name. Default is None.

        out_dir (str) : Directory to save images. A value of None returns images. Default is "preds".

        callbacks (list[Callable]) : Callbacks after each prediction. Can optionally specify an argument for locals to be passed. Default is None.
    
    Returns:
        images (list[np.ndarray]) : Returns predicted images if ``out_dir`` is None.
    """
    if norm and dataset.is_lr: raise ValueError("Dataset must be paired with high-low-resolution images for normalization.")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    callbacks, callback_locals = _get_callbacks(callbacks)

    model.to(device)
    model.eval()

    progress = tqdm(dataset.val_idx)
    outs = []
    with torch.no_grad():
        for idx in progress:
            lr = dataset[idx] if dataset.is_lr else dataset[idx][1]
            lr = lr.to(device).unsqueeze(0)

            hr_hat = model(lr)
            hr_hat = _pred_array(hr_hat).squeeze(0)
            
            if norm:
                _, hr_hat = normalize_preds(_pred_array(dataset[idx][0]), hr_hat)

            crop_res = dataset.crop_res if not dataset.is_lr else dataset.crop_res * (hr_hat.shape[-1]//lr.shape[-1])
            hr_hat = hr_hat[:,:crop_res,:crop_res]

            if out_dir:
                tifffile.imwrite(f"{out_dir}/{prefix+'_' if prefix else ''}{dataset._get_name(idx)}.tif", np.asarray(hr_hat))
            else:
                outs.append(hr_hat)

            for idx, callback in enumerate(callbacks):
                if callback_locals[idx]:
                    callback(locals())
                else:
                    callback()

    if not out_dir:
        return outs

def predict_collage(model : nn.Module, dataset : Dataset, device : str = "cpu", norm : bool = True, n_images : int = None, prefix : str = None, out_dir : str = "preds", callbacks = None):
    r"""Saves to file an image collage of vertically stacked instances of horizontally aligned low-resolution, PSSR upscaled, and high-resolution images in that order.
    If the dataset is in LR mode, the collage will not have high-resolution images.
    Only the center frame of each slice is displayed.

    Only uses evaluation images if applicable. Set ``val_split=1`` in dataset to use all images.

    Args:
        model (nn.Module) : Model to recieve low-resolution images.

        dataset (Dataset) : Paired image dataset to load data from.

        device (str) : Device to train model on. Default is "cpu".

        norm (bool) : Whether to normalize prediction image intensities to ground truth. Default is True.

        n_images (int) : Number of images to concatenate. Set to None to use all validation images, maximum 50. Default is None.

        prefix (str) : Prefix to append at the beginning the output file name. Default is None.

        out_dir (str) : Directory to save collage. Default is "preds".

        callbacks (list[Callable]) : Callbacks after each prediction. Can optionally specify an argument for locals to be passed. Default is None.
    """
    if norm and dataset.is_lr: raise ValueError("Dataset must be paired with high-low-resolution images for normalization.")

    callbacks, callback_locals = _get_callbacks(callbacks)
    n_images = min(50, len(dataset)) if n_images is None else n_images

    model.to(device)
    model.eval()

    collage = Image.new("L", (dataset.crop_res*(2 if dataset.is_lr else 3), dataset.crop_res*n_images))
    with torch.no_grad():
        # Only shuffle if val_split < 1
        for idx, data_idx in enumerate(_RandomIterIdx(dataset.val_idx, seed=True) if len(dataset.val_idx) < len(dataset) else dataset.val_idx):
            if dataset.is_lr:
                lr = dataset[data_idx]
                lr = lr.to(device).unsqueeze(0)
            else:
                hr, lr = dataset[data_idx]
                hr, lr = hr.to(device).unsqueeze(0), lr.to(device).unsqueeze(0)

            hr_hat = model(lr)

            collage.paste(_collage_preds(lr, hr_hat, None if dataset.is_lr else hr, norm, 1, dataset.crop_res, dataset.lr_scale), (0, dataset.crop_res*idx))

            for idx, callback in enumerate(callbacks):
                if callback_locals[idx]:
                    callback(locals())
                else:
                    callback()

            if idx >= n_images - 1:
                break

    os.makedirs(out_dir, exist_ok=True)
    collage.save(f"{out_dir}/{prefix+'_' if prefix else ''}collage_{n_images}.png")

def test_metrics(model : nn.Module, dataset : Dataset, device : str = "cpu", metrics : list[str] = ["mse", "pixel", "psnr", "ssim"], avg : bool = True, norm : bool = True, callbacks = None):
    r"""Computes image restoration metrics of predicted vs ground truth images.

    Only uses evaluation images if applicable. Set ``val_split=1`` in dataset to use all images.

    Args:
        model (nn.Module) : Model to recieve low-resolution images.

        dataset (Dataset) : Paired image dataset to load data from.

        device (str) : Device to train model on. Default is "cpu".

        metrics (list[str]) : Metrics to calculate out of "mse", "pixel", "psnr", and "ssim". Default is all.

        avg (bool) : Whether to return a single averaged value per metric. Default is True.

        norm (bool) : Whether to normalize prediction image intensities to ground truth. Default is True.

        callbacks (list[Callable]) : Callbacks after each prediction. Can optionally specify an argument for locals to be passed. Default is None.
    
    Returns:
        metrics (dict[str, Any]) : Dictionary of metric names and outputs.
    """
    callbacks, callback_locals = _get_callbacks(callbacks)
    image_range = 255

    metrics = [metrics] if type(metrics) is str else metrics
    metrics = {metric:[] for metric in metrics}
    use_mse = True if any(x in metrics.keys() for x in ["mse", "pixel"]) else False

    model.to(device)
    model.eval()

    progress = tqdm(dataset.val_idx)
    with torch.no_grad():
        for idx in progress:
            hr, lr = dataset[0]
            hr, lr = hr.to(device).unsqueeze(0), lr.to(device).unsqueeze(0)

            hr_hat = model(lr)

            hr, hr_hat = _pred_array(hr), _pred_array(hr_hat)

            crop_res = dataset.crop_res if not dataset.is_lr else dataset.crop_res * (hr_hat.shape[-1]//lr.shape[-1])
            hr, hr_hat = hr[:,:,:crop_res,:crop_res], hr_hat[:,:,:crop_res,:crop_res]

            if norm:
                hr, hr_hat = normalize_preds(hr, hr_hat)

            for idx in range(len(hr)):
                mse = np.mean((hr[idx]/image_range-hr_hat[idx]/image_range)**2) if use_mse else None

                if "mse" in metrics:
                    metrics["mse"].append(mse)
                if "pixel" in metrics:
                    metrics["pixel"].append(pixel_metric(mse, image_range))
                if "psnr" in metrics:
                    metrics["psnr"].append(peak_signal_noise_ratio(hr[idx], hr_hat[idx], data_range=image_range))
                if "ssim" in metrics:
                    metrics["ssim"].append(structural_similarity(hr[idx].squeeze(), hr_hat[idx].squeeze(), data_range=image_range))
            
            for idx, callback in enumerate(callbacks):
                if callback_locals[idx]:
                    callback(locals())
                else:
                    callback()

    return {metric:(sum(values)/len(values) if avg else values) for metric, values in metrics.items()}

def _collage_preds(lr, hr_hat, hr, norm : bool = False, max_images : int = 5, crop_res : int = None, lr_scale : int = 4):
    crop_res = hr_hat.shape[-1] if crop_res is None else crop_res
    lr_scale = int(hr_hat.shape[-1]/lr.shape[-1]) if lr_scale is None else lr_scale

    # hr is None in LR mode
    lr, hr_hat, hr = _pred_array(lr)[:,:,:crop_res//lr_scale,:crop_res//lr_scale], _pred_array(hr_hat)[:,:,:crop_res,:crop_res], None if hr is None else _pred_array(hr)[:,:,:crop_res,:crop_res]

    if norm:
        hr, hr_hat = normalize_preds(hr, hr_hat)
        _, lr = normalize_preds(hr, lr)

    lr = _image_stack(lr, max_images)
    hr_hat = _image_stack(hr_hat, max_images)
    hr = None if hr is None else _image_stack(hr, max_images)

    lr = lr.resize((hr_hat.width, hr_hat.height), Image.Resampling.NEAREST)
    if hr is not None and hr_hat.size != hr.size:
        hr_hat = hr_hat.resize((hr.width, hr.height), Image.Resampling.NEAREST)

    return _image_stack([lr, hr_hat] + ([hr] if hr is not None else []), raw=False)

def _image_stack(data, max_images : int = 5, raw : bool = True):
    images = [Image.fromarray(image.astype(np.uint8), mode="L") for image in data[:min(max_images, len(data)), 0]] if raw else data
    width, height = images[0].width, images[0].height
    stack = Image.new("L", (width, height*len(images))) if raw else Image.new("L", (width*len(images), height))
    for idx, image in enumerate(images):
        if raw:
            stack.paste(image, (0, height*idx))
        else:
            stack.paste(image, (width*idx, 0))
    return stack

def _pred_array(data, n_frames=1):
    return _slice_center(np.clip(data.detach().cpu().numpy(), 0, 255).astype(np.uint8), n_frames)
