import torch
import torch.nn as nn
import torch.nn.functional as F
from ._blocks import Reconstruction
from ..data import _force_list

class ResUNet(nn.Module):
    def __init__(self, channels : int = 1, hidden : list[int] = [64, 128, 256, 512, 1024], scale : int = 4, depth : int = 3):
        r"""A modified Residual UNet as detailed in Zhang et al., 2017 with an additional image upscaling block.

        Args:
            channels (int) : Number of channels in image data. Can also be a list of in channels and out channels respectively.

            hidden (list[int]) : Elementwise list of hidden layer channels controlling width and length of model.

            scale (int) : Upscaling factor for predictions. Choose a power of 2 for best results. Default is 4.

            depth (int) : Number of hidden layers per residual block. Default is 3.
        """
        super().__init__()
        channels = _force_list(channels)
        channels = channels*2 if len(channels) == 1 else channels
        
        self.norm = nn.BatchNorm2d(channels[0])
        
        self.encoder, self.decoder, self.upscale = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        layers = [channels[0], *hidden]
        n_layers = len(layers) - 1
        for layer_idx in range(n_layers):
            self.encoder.append(_ResBlock(in_channels=layers[layer_idx], out_channels=layers[layer_idx+1], depth=depth))
            if layer_idx + 1 < n_layers:
                self.decoder.append(_ResBlock(in_channels=layers[-layer_idx-1] - int(layers[-layer_idx-2]/2), out_channels=layers[-layer_idx-2], depth=depth))
                self.upscale.append(nn.PixelShuffle(2))

        self.reconstuction = Reconstruction(channels[0], channels[1], hidden[0], scale)

    def forward(self, x):
        x = x / 128 - 1 # Scale input approx from [0, 255] to [-1, 1]
        x = self.norm(x)

        skips = [x]
        for idx, layer in enumerate(self.encoder):
            x = layer(x) # ResBlock

            if idx + 1 < len(self.encoder): # Downscale
                skips.append(x)
                x = F.max_pool2d(x, kernel_size=2)

        for idx, layer in enumerate(self.decoder):
            x = self.upscale[idx](x) # Upscale

            x = torch.cat([x, skips.pop()], dim=1) # ResBlock
            x = layer(x)

        x = torch.cat([x, skips.pop()], dim=1) # Final skip connection before reconstruction
        if len(skips) != 0: raise IndexError(f"Skip connection mismatch between encoder and decoder. {len(skips)} skip connections are unused.")

        x = self.reconstuction(x)

        x = x * 128 + 128 # Scale output approx from [-1, 1] to [0, 255]
        return x
    
class _ResBlock(nn.Module):
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

    def forward(self, x):
        x = F.relu(self.conv(x) + self.respass(x))
        return x
