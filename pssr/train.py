import torch, os
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pytorch_msssim import ssim
from tqdm import tqdm
from PIL import Image
from .metrics import pixel, psnr
Image.MAX_IMAGE_PIXELS = None

def fit_paired(model : nn.Module, dataset : Dataset, epochs : int, batch_size : int, loss_fn : nn.Module, optim : torch.optim.Optimizer, device : str = "cpu", image_range : int = 255, clamp : bool = False, log_frequency : int = 50, dataloader_kwargs = None):
    r"""Trains model on paired high and low resolution crappified data.

    Args:
        model (nn.Module) : Model to train on paired data.

        dataset (Dataset) : Paired image dataset to load data from.

        epochs (int) : Number of epochs to train model for.

        batch_size (int) : Batch size for dataloader.

        loss_fn (nn.Module) : Loss function for loss calculation.

        optim (Optimizer) : Optimizer for weight calculation.

        device (str) : Device to train model on. Default is "cpu".

        image_range (int) : Value space for images. Default is 255.

        clamp (bool) : Whether to clamp model output before weight calculation. Default is False.

        log_frequency (int) : Frequency to log losses and recalculate metrics in steps. Default is 50.

        dataloader_kwargs (dict[str, Any]) : Keyword arguments for pytorch Dataloader. Default is None.

    Returns:
        losses (list[float]) : List of losses during training.
    """
    dataloader_kwargs = {} if dataloader_kwargs is None else dataloader_kwargs
    train_dataloader = DataLoader(dataset, batch_size, **dataloader_kwargs)

    model.to(device)
    model.train()

    losses = []
    for epoch in range(epochs):
        progress = tqdm(train_dataloader)
        for batch_idx, (hr, lr) in enumerate(progress):
            hr, lr = hr.to(device), lr.to(device)

            hr_hat = model(lr)
            if clamp:
                hr_hat = torch.clamp(hr_hat, 0, image_range)

            loss = loss_fn(hr_hat/image_range, hr/image_range)
            loss.backward()
            optim.step()
            optim.zero_grad()

            if batch_idx % log_frequency == 0:
                losses.append(loss.item())

                mse = nn.functional.mse_loss(hr_hat/image_range, hr/image_range)
                progress.set_description(f"mse[{mse:.5f}], pixel[{pixel(mse, image_range):.2f}], psnr[{psnr(mse, hr.max()/image_range):.2f}], ssim[{ssim(hr_hat, hr, data_range=image_range):.3f}]")
  
        collage = _collage_preds(lr, hr_hat, hr, image_range)
        os.makedirs("preds", exist_ok=True)
        collage.save(f"preds/pred{epoch}.png")
    return losses

def predict_collage(model : nn.Module, dataset, n_images : int, batch_size : int, device : str = "cpu", image_range : int = 255, dataloader_kwargs = None):
    r"""Saves to file an image collage of vertically stacked instances of horizontally aligned low resolution, upscaled, and high resolution images in that order.

    Args:
        model (nn.Module) : Model to train on paired data.

        dataset (Dataset) : Paired image dataset to load data from.

        n_images (int) : Number of images to concatenate.

        batch_size (int) : Batch size for dataloader.

        device (str) : Device to train model on. Default is "cpu".

        image_range (int) : Value space for images. Default is 255.

        dataloader_kwargs (dict[str, Any]) : Keyword arguments for pytorch Dataloader. Default is None.
    """
    dataloader_kwargs = {} if dataloader_kwargs is None else dataloader_kwargs
    train_dataloader = DataLoader(dataset, batch_size, **dataloader_kwargs)

    model.to(device)
    model.eval()

    collage = Image.new("L", (dataset.hr_res*3, dataset.hr_res*n_images))
    remaining = n_images
    for idx, (hr, lr) in enumerate(train_dataloader):
        hr, lr = hr.to(device), lr.to(device)

        hr_hat = model(lr)
        
        collage.paste(_collage_preds(lr, hr_hat, hr, image_range, min(remaining, batch_size)), (0, dataset.hr_res*batch_size*idx))

        del hr, lr, hr_hat

        remaining -= batch_size
        if remaining <= 0:
            break

    os.makedirs("preds", exist_ok=True)
    collage.save(f"preds/collage{n_images}.png")

def _collage_preds(lr, hr_hat, hr, image_range : int = 255, max_images : int = 5):
    lr, hr_hat, hr = [_image_stack(data, image_range, max_images) for data in (lr, hr_hat, hr)]
    sf = hr.width//lr.width
    lr = lr.resize((lr.width*sf, lr.height*sf))

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
