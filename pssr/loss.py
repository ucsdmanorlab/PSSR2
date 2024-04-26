import torch, math
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pytorch_msssim import SSIM, MS_SSIM

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
            l1 = F.conv2d(F.l1_loss(input, target, reduction="none"), self.gaussian.to(input.get_device()), groups=self.channels, padding=(self.win_size-1)//2).mean()  # F.conv2d with Gaussian filter
            x = self.mix*x + (1-self.mix)*l1
        return x

def pixel_metric(mse : float, image_range : int = 255):
    r"""Simple metric for calculating average pixel error.
    
    Args:
        mse (float) : Mean squared error between predicted and ground truth images.

        image_range (int) : Value range of image. Default is 255.
    """
    return math.sqrt(mse) * image_range

def _psnr_metric(mse : float, max : float):
    return 20 * torch.log10(max / torch.sqrt(mse))
