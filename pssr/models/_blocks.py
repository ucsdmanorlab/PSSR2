import torch
import torch.nn as nn
import torch.nn.functional as F

class Reconstruction(nn.Module):
    def __init__(self, channels : int, hidden : int, scale : int = 4):
        super().__init__()

        self.pre = nn.Conv2d(in_channels=hidden + channels, out_channels=scale**2 * hidden, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(in_channels=hidden, out_channels=channels, kernel_size=3, padding=1)

        self.scale = scale

    def forward(self, x):
        x = F.relu(self.pre(x), inplace=True)
        x = self.conv(F.pixel_shuffle(x, self.scale))
        return x
