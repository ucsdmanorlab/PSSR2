import torch, os, cv2
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pytorch_msssim import ssim
from tqdm import tqdm
from PIL import Image
from .loss import pixel_metric, psnr_metric

def predict_images(model : nn.Module, dataset : Dataset, device : str = "cpu", out_dir : str = "preds", out_res : int = None):
    r"""Predicts and saves high-resolution images from low-resolution using a given model.

    Args:
        model (nn.Module) : Model recieve low-resolution images.

        dataset (Dataset) : Single image dataset to load data from.

        device (str) : Device to train model on. Default is "cpu".

        out_dir (str) : Path to save images. Default is "preds".

        out_res (int) : Resolution to scale images to, if different than that of the model output. Default is None.
    """
    os.makedirs(out_dir, exist_ok=True)

    model.to(device)
    model.eval()

    progress = tqdm(range(dataset.train_len, dataset.train_len + dataset.val_len))
    with torch.no_grad():
        for idx in progress:
            lr = dataset[idx] if dataset.is_lr else dataset[idx][1]
            lr = lr.to(device).unsqueeze(0)

            hr_hat = model(lr)
            
            hr_hat = Image.fromarray(hr_hat.detach().cpu().numpy().squeeze())
            if out_res:
                hr_hat = hr_hat.resize([out_res]*2, Image.Resampling.BILINEAR)
            cv2.imwrite(out_dir + f"/pred_{idx}.png", np.asarray(hr_hat))

def predict_collage(model : nn.Module, dataset : Dataset, batch_size : int, n_images : int, device : str = "cpu", prefix : str = None, dataloader_kwargs = None):
    r"""Saves to file an image collage of vertically stacked instances of horizontally aligned low resolution, upscaled, and high resolution images in that order.

    Args:
        model (nn.Module) : Model recieve low-resolution images.

        dataset (Dataset) : Paired image dataset to load data from.

        batch_size (int) : Batch size for dataloader.

        n_images (int) : Number of images to concatenate.

        device (str) : Device to train model on. Default is "cpu".

        prefix (str) : Prefix to append at the beginning the output file name. Default is None.

        dataloader_kwargs (dict[str, Any]) : Keyword arguments for pytorch Dataloader. Default is None.
    """
    prefix = "" if prefix is None else prefix
    dataloader_kwargs = {} if dataloader_kwargs is None else dataloader_kwargs
    val_dataloader = DataLoader(dataset, batch_size, sampler=range(dataset.train_len, dataset.train_len + dataset.val_len), **{key:dataloader_kwargs[key] for key in dataloader_kwargs if key!="shuffle"})

    model.to(device)
    model.eval()

    collage = Image.new("L", (dataset.hr_res*3, dataset.hr_res*n_images))
    remaining = n_images
    with torch.no_grad():
        for idx, (hr, lr) in enumerate(val_dataloader):
            hr, lr = hr.to(device), lr.to(device)

            hr_hat = model(lr)
            
            collage.paste(_collage_preds(lr, hr_hat, hr, 255, min(remaining, batch_size)), (0, dataset.hr_res*batch_size*idx))

            del hr, lr, hr_hat

            remaining -= batch_size
            if remaining <= 0:
                break

    os.makedirs("preds", exist_ok=True)
    collage.save(f"preds/{prefix}collage{n_images}.png")

def test_metrics(model : nn.Module, dataset : Dataset, batch_size : int = 1, metrics : list[str] = ["mse", "pixel", "psnr", "ssim"], device : str = "cpu", clamp : bool = False, dataloader_kwargs = None):
    r"""Computes image restoration metrics of predicted vs ground truth images.

    Args:
        model (nn.Module) : Model recieve low-resolution images.

        dataset (Dataset) : Paired image dataset to load data from.

        batch_size (int) : Batch size for dataloader. Default is 1.

        metrics (list[str]) : Metrics to calculate out of "mse", "pixel", "psnr", and "ssim". Default is all.

        device (str) : Device to train model on. Default is "cpu".

        clamp (bool) : Whether to clamp model image output before weight calculation. Default is False.

        dataloader_kwargs (dict[str, Any]) : Keyword arguments for pytorch Dataloader. Default is None.
    """
    dataloader_kwargs = {} if dataloader_kwargs is None else dataloader_kwargs
    val_dataloader = DataLoader(dataset, batch_size, sampler=range(dataset.train_len, dataset.train_len + dataset.val_len), **{key:dataloader_kwargs[key] for key in dataloader_kwargs if key!="shuffle"})
    image_range = 255

    metrics = [metrics] if type(metrics) is str else metrics
    metrics = {metric:[] for metric in metrics}
    use_mse = True if any(x in metrics.keys() for x in ["mse", "pixel", "psnr"]) else False

    model.to(device)
    model.eval()

    progress = tqdm(val_dataloader)
    with torch.no_grad():
        for hr, lr in progress:
            hr, lr = hr.to(device), lr.to(device)

            hr_hat = model(lr)
            if clamp:
                hr_hat = torch.clamp(hr_hat, 0, image_range)

            mse = nn.functional.mse_loss(hr_hat/image_range, hr/image_range) if use_mse else None

            if "mse" in metrics:
                metrics["mse"].append(mse.item())
            if "pixel" in metrics:
                metrics["pixel"].append(pixel_metric(mse, image_range))
            if "psnr" in metrics:
                metrics["psnr"].append(psnr_metric(mse, hr.max()/image_range).item())
            if "ssim" in metrics:
                metrics["ssim"].append(ssim(hr_hat, hr, data_range=image_range).item())

    metrics = {metric:sum(values)/len(values) for metric, values in metrics.items()}
    return metrics

def _collage_preds(lr, hr_hat, hr, image_range : int = 255, max_images : int = 5):
    lr, hr_hat, hr = [_image_stack(data, image_range, max_images) for data in (lr, hr_hat, hr)]
    lr = lr.resize((hr.width, hr.height), Image.Resampling.NEAREST)
    if hr_hat.size != hr.size:
        hr_hat = hr_hat.resize((hr.width, hr.height), Image.Resampling.NEAREST)

    return _image_stack([lr, hr_hat, hr], image_range, raw=False)

def _image_stack(data, image_range, max_images : int = 5, raw : bool = True):
    images = [Image.fromarray(image, mode="L") for image in (np.clip(data.detach().cpu().numpy()[:min(max_images, len(data)), 0], 0, image_range)*(255//image_range)).astype(np.uint8)] if raw else data
    width, height = images[0].width, images[0].height
    stack = Image.new("L", (width, height*len(images))) if raw else Image.new("L", (width*len(images), height))
    for idx, image in enumerate(images):
        if raw:
            stack.paste(image, (0, height*idx))
        else:
            stack.paste(image, (width*idx, 0))
    return stack
