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
    
class ResBlock(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, depth : int, norm : bool = True):
        super().__init__()

        self.conv = nn.Sequential()
        
        n_layers = max(depth, 0) + 1
        for layer_idx in range(n_layers):
            self.conv.append(nn.Conv2d(in_channels if layer_idx == 0 else out_channels, out_channels, kernel_size=3, padding=1))
            
            if norm:
                self.conv.append(nn.BatchNorm2d(out_channels))
            if layer_idx + 1 < n_layers:
                self.conv.append(nn.ReLU(inplace=True))

        self.respass = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.depth = depth

    def forward(self, x):
        x = F.relu(self.conv(x) + self.respass(x))
        return x

class ResBlockA(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, dilations : list[int], depth : int, norm : bool = True):
        super().__init__()

        self.dilations = nn.ModuleList()
        for dilation in dilations:
            conv = nn.Sequential()

            n_layers = max(depth, 0) + 1
            for layer_idx in range(n_layers):
                if norm:
                    conv.append(nn.BatchNorm2d(in_channels if layer_idx == 0 else out_channels))
                conv.append(nn.ReLU(inplace=True))

                conv.append(nn.Conv2d(in_channels if layer_idx == 0 else out_channels, out_channels, kernel_size=3, padding="same", dilation=dilation))
            self.dilations.append(conv)

        self.respass = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.min_size = max(dilations) * 2 + 1
        self.depth = depth

    def forward(self, x):
        if x.shape[-1] < self.min_size: raise ValueError(f"Tensor size {x.shape} is smaller than than dilation kernel size {self.min_size}.")
        x = F.relu(sum([conv(x) for conv in self.dilations]) + self.respass(x))
        return x

class PSP_Pooling(nn.Module):
    def __init__(self, channels, sizes):
        super().__init__()

        small = channels//len(sizes)
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv2d(small, small, kernel_size=1), nn.BatchNorm2d(small)) for size in sizes])

        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm_out = nn.BatchNorm2d(channels)

        self.sizes = sizes

    def forward(self, x):
        size = x.shape[-2:]

        # Split x along sizes and apply poolings
        x = torch.chunk(x, chunks=len(self.sizes), dim=1)
        x = [F.interpolate(input=F.max_pool2d(x_chunk, kernel_size=self.sizes[idx]), size=size, mode="bilinear") for idx, x_chunk in enumerate(x)]
        x = [F.relu(self.convs[idx](x_chunk)) for idx, x_chunk in enumerate(x)]
        x = torch.concat(x, dim=1)

        x = F.relu(self.norm_out(self.conv_out(x)))
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
    
def get_resblock(in_channels : int, out_channels : int, dilations : list[int], depth : int, norm : bool = True):
    if dilations:
        return ResBlockA(in_channels, out_channels, dilations, depth, norm)
    return ResBlock(in_channels, out_channels, depth, norm)
