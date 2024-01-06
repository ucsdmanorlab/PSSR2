import torch, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pytorch_msssim import ssim
from skopt import gp_minimize
from skopt.space import Dimension
from tqdm import tqdm
from PIL import Image
from .metrics import pixel, psnr
from .crappifiers import Crappifier
from .loss import SSIMLoss
from .models._blocks import GradHist
Image.MAX_IMAGE_PIXELS = None

def train_paired(model : nn.Module, dataset : Dataset, batch_size : int, loss_fn : nn.Module, optim : torch.optim.Optimizer, epochs : int, device : str = "cpu", image_range : int = 255, clamp : bool = False, log_frequency : int = 50, dataloader_kwargs = None):
    r"""Trains model on paired high and low resolution crappified data.

    Args:
        model (nn.Module) : Model to train on paired data.

        dataset (Dataset) : Paired image dataset to load data from.

        batch_size (int) : Batch size for dataloader.

        loss_fn (nn.Module) : Loss function for loss calculation.

        optim (Optimizer) : Optimizer for weight calculation.

        epochs (int) : Number of epochs to train model for.

        device (str) : Device to train model on. Default is "cpu".

        image_range (int) : Value space for images. Default is 255.

        clamp (bool) : Whether to clamp model image output before weight calculation. Default is False.

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

def train_crappifier(model : nn.Module, dataset : Dataset, batch_size : int, optim : torch.optim.Optimizer, epochs : int, sigma : int = 5, clip : float = 3, device : str = "cpu", image_range : int = 255, clamp : bool = False, log_frequency : int = 50, dataloader_kwargs = None):
    r"""Trains an :class:`nn.Module` model as a crappifier on high and low resolution paired data.
    The model must output an image the same size as the input/have a :var:`scale` value of 1.
    This is not necessary if you are using a :class:`Crappifier` instance as your crappifier.

    Args:
        model (nn.Module) : Model to train on paired data.

        dataset (Dataset) : Paired image dataset to load data from.

        batch_size (int) : Batch size for dataloader.

        optim (Optimizer) : Optimizer for weight calculation.

        epochs (int) : Number of epochs to train model for.

        sigma (int) : Precision of noise distribution. Higher values will yield better results can cause larger gradients that are unstable during training. Default is 5.

        clip : (float) : Max gradient for gradient clipping. Use None for no clipping.

        device (str) : Device to train model on. Default is "cpu".

        image_range (int) : Value space for images. Default is 255.

        clamp (bool) : Whether to clamp model image output before weight calculation. Default is False.

        log_frequency (int) : Frequency to log losses and recalculate metrics in steps. Default is 50.

        dataloader_kwargs (dict[str, Any]) : Keyword arguments for pytorch Dataloader. Default is None.

    Returns:
        losses (list[float]) : List of losses during training.
    """
    dataloader_kwargs = {} if dataloader_kwargs is None else dataloader_kwargs
    train_dataloader = DataLoader(dataset, batch_size, **dataloader_kwargs)

    model.to(device)
    model.train()

    # Leave GradHist on cpu?
    hist_fn = GradHist(sigma=sigma)
    ssim_loss = SSIMLoss(ms=False)

    losses = []
    for epoch in range(epochs):
        progress = tqdm(train_dataloader)
        for batch_idx, (hr, lr) in enumerate(progress):
            # Process outlined in approximate_crappifier
            scale = int(hr.shape[-1]/lr.shape[-1])
            ds_hr = hr[:, :, ::scale, ::scale]
            ds_hr = ds_hr.to(device)

            lr_hat = model(ds_hr)

            pred_profile = lr_hat - ds_hr
            target_profile = lr.to(device) - ds_hr

            pred_dist = hist_fn(pred_profile)
            target_dist = hist_fn(target_profile)

            # We are generating the noise, so a spacial criterion must be present (in a lower order as to optimize purely to the "noiseless" profile)
            dist_error = F.mse_loss(pred_dist, target_dist)
            profile_error = ssim_loss(pred_profile, target_profile)
            # value_error = F.l1_loss(pred_profile.view(pred_profile.shape[0], -1).mean(1), target_profile.view(target_profile.shape[0], -1).mean(1))

            loss = dist_error * profile_error / (image_range**2)
            loss.backward()

            if clip is not None and clip > 0:
                nn.utils.clip_grad_value_(model.parameters(), clip)

            optim.step()
            optim.zero_grad()

            if batch_idx % log_frequency == 0:
                losses.append(loss.item())

                progress.set_description(f"loss[{loss.item():.5f}]")

        collage = _collage_preds(lr, lr_hat, hr, image_range)
        os.makedirs("preds", exist_ok=True)
        collage.save(f"preds/pred{epoch}.png")
    return losses


def predict_collage(model : nn.Module, dataset, batch_size : int, n_images : int, device : str = "cpu", image_range : int = 255, prefix : str = None, dataloader_kwargs = None):
    r"""Saves to file an image collage of vertically stacked instances of horizontally aligned low resolution, upscaled, and high resolution images in that order.

    Args:
        model (nn.Module) : Model to train on paired data.

        dataset (Dataset) : Paired image dataset to load data from.

        batch_size (int) : Batch size for dataloader.

        n_images (int) : Number of images to concatenate.

        device (str) : Device to train model on. Default is "cpu".

        image_range (int) : Value space for images. Default is 255.

        prefix (str) : Prefix to append at the beginning the output file name. Default is None.

        dataloader_kwargs (dict[str, Any]) : Keyword arguments for pytorch Dataloader. Default is None.
    """
    prefix = "" if dataloader_kwargs is None else prefix
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
    collage.save(f"preds/{prefix}collage{n_images}.png")

def approximate_crappifier(crappifier : Crappifier, space : list[Dimension], dataset : Dataset, n_samples : int = 10, opt_kwargs = None):
    r"""Approximates :class:`Crappifier` parameters from ground truth paired images. Uses Bayesian optimization because Crappifier functions are not differentiable.

    Args:
        crappifier (Crappifier) : Crappifier whose parameter space will be optimized.

        space (list[Dimension]) : List of parameter spaces for each crappifier parameter.

        dataset (Dataset) : Paired image dataset to load data from.

        n_samples (int) : Number of image samples to average computations over for each optimization step. Default is 10.

        opt_kwargs (dict[str, Any]) : Keyword arguments for skopt :meth:`gp_minimize`. Default is None
    """
    opt_kwargs = {} if opt_kwargs is None else opt_kwargs
    space = [space] if type(space) is not list else space

    objective = _Crappifier_Objective(crappifier, dataset, n_samples).sample

    result = gp_minimize(objective, space, **opt_kwargs)

    return result

# TODO: Replace PIL with np fancy indexing
class _Crappifier_Objective():
    def __init__(self, crappifier : Crappifier, dataset : Dataset, n_samples : int):
        self.crappifier = crappifier
        self.dataset = dataset
        self.n_samples = n_samples

        self.idx = 0

    def sample(self, params):
        metrics = []
        for sample in range(int(self.n_samples)):
            # Grab gound truth high and low resolution images
            hr, lr = self.dataset[self.idx]
            hr, lr = np.asarray(hr, dtype=np.uint8), np.asarray(lr, dtype=np.uint8)

            self.idx = self.idx + 1 if self.idx < len(self.dataset) - 1 else 0
            
            # Downsampled high resolution image is the baseline for noise profile comparison
            ds_hr = np.asarray(Image.fromarray(np.squeeze(np.moveaxis(hr, 0, -1))).resize(lr.shape[-2:], Image.Resampling.NEAREST))
            if len(ds_hr.shape) > 2:
                ds_hr = np.moveaxis(ds_hr, -1, 0)

            # Generate artificial low resolution image using optimized crappifier parameters
            lr_hat = self.crappifier(*params).crappify(ds_hr)
            if len(lr_hat.shape) < 3:
                lr_hat = lr_hat[np.newaxis, :, :]
 
            # Generate distribution of noise values for both crappified and ground truth low resolution images
            # NOTE: Cant use SSIM or MSE on images as Crappifier noise levels will converge to zero because a noiseless downscaled image will be closer to ground truth than one with correct amount of noise
            pred_profile = lr_hat.astype(np.float32) - ds_hr.astype(np.float32)
            target_profile = lr.astype(np.float32) - ds_hr.astype(np.float32)

            bins = np.arange(-256, 256)
            pred_dist, _ = np.histogram(pred_profile.flatten(), bins)
            target_dist, _ = np.histogram(target_profile.flatten(), bins)
            
            # Aggregate errors of both noise distribution and mean noise profile value
            # We are not generating the noise, so spacial significance is low and mean value is used (spacial loss would underapproximate the correct amount of noise)
            dist_error = (target_dist - pred_dist)**2
            value_error = abs(target_profile.mean() - pred_profile.mean())

            loss = dist_error.mean() * value_error / (255**2)
            metrics.append(loss)
        return sum(metrics) / len(metrics)

def _collage_preds(lr, hr_hat, hr, image_range : int = 255, max_images : int = 5):
    lr, hr_hat, hr = [_image_stack(data, image_range, max_images) for data in (lr, hr_hat, hr)]
    lr = lr.resize((hr.width, hr.height))
    if hr_hat.size != hr.size:
         hr_hat = hr_hat.resize((hr.width, hr.height))

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
