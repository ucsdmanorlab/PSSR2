import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Reconstruction(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, hidden : int, scale : int = 4):
        super().__init__()

        self.pre = nn.Conv2d(in_channels=hidden + in_channels, out_channels=scale**2 * hidden, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_channels=hidden, out_channels=out_channels, kernel_size=3, padding=1)

        self.scale = scale

    def forward(self, x):
        x = F.relu(self.pre(x), inplace=True)
        x = self.conv(F.pixel_shuffle(x, self.scale))
        return x
    
class GradHist(nn.Module):
    # https://discuss.pytorch.org/t/differentiable-torch-histc/25865/38
    def __init__(self, bins : int = 512, range : list[int, int] = (-256, 256), sigma : int = 5):
        super().__init__()
        assert range[1] > range[0]
        
        self.delta = float(range[1]-range[0]) / float(bins)
        self.centers = float(range[0]) + self.delta * (torch.arange(bins).float() + 0.5)

        self.sigma = sigma

    def forward(self, x):
        batch, size = x.shape[0], np.prod(x.shape[2:])
        x = x.flatten(start_dim=1)[:, np.newaxis, :] - self.centers[:, np.newaxis].to(device=x.device)
        x = torch.sigmoid(x * self.sigma)
        diff = torch.cat([torch.ones((batch, 1, size), device=x.device), x], dim=1) - torch.cat([x, torch.zeros((batch, 1, size), device=x.device)], dim=1)

        diff = diff.sum(dim=-1)
        return diff[:, :-1]
