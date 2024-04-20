import torch, os, cv2
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
from PIL import Image
from .loss import pixel_metric

def predict_images(model : nn.Module, dataset : Dataset, device : str = "cpu", out_dir : str = None, norm : bool = False):
    r"""Predicts high-resolution images from low-resolution images using a given model.
    
    Only predicts evaluation images if applicable. Set `val_split=1` in dataset to use all images.

    Args:
        model (nn.Module) : Model recieve low-resolution images.

        dataset (Dataset) : Single image dataset to load data from.

        device (str) : Device to train model on. Default is "cpu".

        out_dir (str) : Set to a path to automatically save images, otherwise return images. Default is None.

        norm (bool) : Whether to normalize prediction image intensities to ground truth, which must be provided by a paired dataset. Default is False.
    
    Returns:
        images (list[Image]) : Returns predicted images if `out_dir` is None.
    """
    if norm:
        assert not dataset.is_lr, "Dataset must be paired with high-low-resolution images for normalization."

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
                _, hr_hat = normalize_preds(np.mean(_pred_array(dataset[idx][0]), axis=0), np.mean(hr_hat, axis=0))

            hr_hat = Image.fromarray(hr_hat.astype(np.uint8).squeeze())
            hr_hat = hr_hat.crop([0,0,dataset.crop_res,dataset.crop_res])

            outs.append(hr_hat)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for idx, hr_hat in enumerate(outs):
            cv2.imwrite(out_dir + f"/{dataset._get_name(idx)}_pred.png", np.asarray(hr_hat))
    else:
        return outs

def predict_collage(model : nn.Module, dataset : Dataset, batch_size : int, device : str = "cpu", norm : bool = True, n_images : int = None, prefix : str = None, dataloader_kwargs = None):
    r"""Saves to file an image collage of vertically stacked instances of horizontally aligned low-resolution, PSSR upscaled, and high-resolution images in that order.

    Only predicts evaluation images if applicable. Set `val_split=1` in dataset to use all images.

    Args:
        model (nn.Module) : Model recieve low-resolution images.

        dataset (Dataset) : Paired image dataset to load data from.

        batch_size (int) : Batch size for dataloader.

        device (str) : Device to train model on. Default is "cpu".

        norm (bool) : Whether to normalize prediction image intensities to ground truth. Default is True.

        n_images (int) : Number of images to concatenate. Set to None to use all validation images, maximum 50. Default is None.

        prefix (str) : Prefix to append at the beginning the output file name. Default is None.

        dataloader_kwargs (dict[str, Any]) : Keyword arguments for pytorch Dataloader. Default is None.
    """
    prefix = "" if prefix is None else prefix
    dataloader_kwargs = {} if dataloader_kwargs is None else dataloader_kwargs
    val_dataloader = DataLoader(dataset, batch_size, sampler=dataset.val_idx, **dataloader_kwargs)
    n_images = min(50, len(dataset)) if n_images is None else n_images

    model.to(device)
    model.eval()

    collage = Image.new("L", (dataset.crop_res*3, dataset.crop_res*n_images))
    remaining = n_images
    with torch.no_grad():
        for idx, (hr, lr) in enumerate(val_dataloader):
            hr, lr = hr.to(device), lr.to(device)

            hr_hat = model(lr)

            collage.paste(_collage_preds(lr, hr_hat, hr, norm, min(remaining, batch_size), dataset.crop_res), (0, dataset.crop_res*batch_size*idx))

            remaining -= batch_size
            if remaining <= 0:
                break

    os.makedirs("preds", exist_ok=True)
    collage.save(f"preds/{prefix+'_' if prefix else ''}collage_{n_images}.png")

def test_metrics(model : nn.Module, dataset : Dataset, batch_size : int, device : str = "cpu", metrics : list[str] = ["mse", "pixel", "psnr", "ssim"], avg : bool = True, norm : bool = True, dataloader_kwargs = None):
    r"""Computes image restoration metrics of predicted vs ground truth images.

    Args:
        model (nn.Module) : Model recieve low-resolution images.

        dataset (Dataset) : Paired image dataset to load data from.

        batch_size (int) : Batch size for dataloader.

        device (str) : Device to train model on. Default is "cpu".

        metrics (list[str]) : Metrics to calculate out of "mse", "pixel", "psnr", and "ssim". Default is all.

        avg (bool) : Whether to return a single averaged value per metric. Default is True.

        norm (bool) : Whether to normalize prediction image intensities to ground truth. Default is True.

        dataloader_kwargs (dict[str, Any]) : Keyword arguments for pytorch Dataloader. Default is None.
    
    Returns:
        metrics (dict[str, Any]) : Dictionary of metric names and outputs.
    """
    dataloader_kwargs = {} if dataloader_kwargs is None else dataloader_kwargs
    val_dataloader = DataLoader(dataset, batch_size, sampler=dataset.val_idx, **dataloader_kwargs)
    image_range = 255

    metrics = [metrics] if type(metrics) is str else metrics
    metrics = {metric:[] for metric in metrics}
    use_mse = True if any(x in metrics.keys() for x in ["mse", "pixel"]) else False

    model.to(device)
    model.eval()

    progress = tqdm(val_dataloader)
    with torch.no_grad():
        for hr, lr in progress:
            hr, lr = hr.to(device), lr.to(device)

            hr_hat = model(lr)

            hr, hr_hat = _pred_array(hr)[:,:,:dataset.crop_res,:dataset.crop_res], _pred_array(hr_hat)[:,:,:dataset.crop_res,:dataset.crop_res]

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

    metrics = {metric:(sum(values)/len(values) if avg else values) for metric, values in metrics.items()}
    return metrics

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
    assert len(hr.shape) == len(hr_hat.shape), "hr and hr_hat must have the same number of dimensions."
    hr_shape = hr.shape
    hr_hat_shape = hr_hat.shape

    if len(hr.shape) < 3:
        hr, hr_hat = hr[np.newaxis, ...], hr_hat[np.newaxis, ...]
    hr, hr_hat = hr.reshape(-1, *hr.shape[-2:]), hr_hat.reshape(-1, *hr_hat.shape[-2:])
    assert len(hr) == len(hr_hat), "hr and hr_hat must have the same number of images."

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

        scale = np.cov(cv2.resize(hr_hat_norm, hr_norm.shape).flatten(), hr_norm.flatten())[0, 1] / np.var(hr_hat_norm.flatten())
        hr_hat_norm = scale * hr_hat_norm
        
        # Rescale to initial image intensity
        hr_norm, hr_hat_norm = (hr_norm-hr_norm.min())*base_max, (hr_hat_norm-hr_norm.min())*base_max
        hr_norm, hr_hat_norm = hr_norm/(hr_norm.mean()/base_mean), hr_hat_norm/(hr_hat_norm.mean()/base_mean)

        hr_norms.append(hr_norm)
        hr_hat_norms.append(hr_hat_norm)
    
    hr, hr_hat = np.asarray(hr_norms).clip(0, 255), np.asarray(hr_hat_norms).clip(0, 255)
    return hr.reshape(hr_shape), hr_hat.reshape(hr_hat_shape)
    

def _collage_preds(lr, hr_hat, hr, norm : bool = True, max_images : int = 5, crop_res : int = None):
    crop_res = hr.shape[-1] if crop_res is None else crop_res
    lr, hr_hat, hr = _pred_array(lr)[:,:,:crop_res//4,:crop_res//4], _pred_array(hr_hat)[:,:,:crop_res,:crop_res], _pred_array(hr)[:,:,:crop_res,:crop_res]

    if norm:
        hr, hr_hat = normalize_preds(hr, hr_hat)
        _, lr = normalize_preds(hr, lr)

    lr = _image_stack(lr, max_images)
    hr_hat = _image_stack(hr_hat, max_images)
    hr = _image_stack(hr, max_images)

    lr = lr.resize((hr.width, hr.height), Image.Resampling.NEAREST)
    if hr_hat.size != hr.size:
        hr_hat = hr_hat.resize((hr.width, hr.height), Image.Resampling.NEAREST)

    return _image_stack([lr, hr_hat, hr], raw=False)

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

def _pred_array(data):
    return np.clip(data.detach().cpu().numpy(), 0, 255).astype(np.uint8)
