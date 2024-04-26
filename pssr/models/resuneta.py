import torch, warnings
import torch.nn as nn
import torch.nn.functional as F
from ._blocks import Reconstruction
from ..data import _force_list

class ResUNetA(nn.Module):
    def __init__(self, channels : int = 1, hidden : list[int] = [64, 128, 256, 512, 1024], scale : int = 4, depth : int = 3, dilations : list[list[int]] = [[1,3,15,31],[1,3,15],[1,3],[1],[1]], pool_sizes : list[int] = [1, 2, 4, 8], encoder_pool : bool = False):
        r"""A modified Atrous Residual UNet as detailed in Diakogiannis et al., 2019 with an additional image upscaling block.

        Channel sizes hidden[0] (and hidden[-1] if encoder_pool is True) must be divisible by len(pool_sizes).

        Args:
            channels (int) : Number of channels in image data. Can also be a list of in channels and out channels respectively.

            hidden (list[int]) : Elementwise list of hidden layer channels controlling width and length of model.

            scale (int) : Upscaling factor for predictions. Choose a power of 2 for best results. Default is 4.

            depth (int) : Number of hidden layers per residual block. Default is 3.

            dilations (list[list[int]]) : List of dilation values per layer. If value is none, atrous convolutions will not be used.

            pool_sizes (list[int]) : Pooling ratios for PSP pooling.

            encoder_pool (bool) : Whether to include PSP pooling layer at end of encoder. Should not be used if last layer has a size of less than 16 pixels. Default is False.
        """
        super().__init__()
        channels = _force_list(channels)
        channels = channels*2 if len(channels) == 1 else channels

        if dilations is None:
            warnings.warn("dilations is None, atrous convolutions will not be used.", stacklevel=2)
            dilations = [1]*len(hidden)
        if len(dilations) != len(hidden): raise ValueError(f"len(dilations) must equal len(hidden). Lengths are {len(dilations)} and {len(hidden)}.")
        if hidden[0] % len(pool_sizes) != 0: raise ValueError(f"hidden[0] must be divisible by len(pool_sizes). Sizes are {hidden[0]} and {len(pool_sizes)}.")
        if hidden[-1] % len(pool_sizes) != 0 and encoder_pool: raise ValueError(f"hidden[-1] must be divisible by len(pool_sizes) if encoder_pool is True. Sizes are {hidden[-1]} and {len(pool_sizes)}.")
        
        self.encoder, self.decoder, self.upscale = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        layers = [channels[0], *hidden]
        n_layers = len(layers) - 1
        for layer_idx in range(n_layers):
            self.encoder.append(_ResBlockA(in_channels=layers[layer_idx], out_channels=layers[layer_idx+1], dilations=dilations[layer_idx], depth=depth))
            if layer_idx + 1 < n_layers:
                self.decoder.append(_ResBlockA(in_channels=layers[-layer_idx-1] - int(layers[-layer_idx-2]/2), out_channels=layers[-layer_idx-2], dilations=dilations[-layer_idx-1], depth=depth))
                self.upscale.append(nn.PixelShuffle(2))

        self.encoder_pool = _PSP_Pooling(hidden[-1], pool_sizes) if encoder_pool else None
        self.reconstuction_pool = _PSP_Pooling(hidden[0], pool_sizes)

        self.reconstuction = Reconstruction(channels[0], channels[1], hidden[0], scale)

    def forward(self, x):
        x = x / 128 - 1 # Scale input approx from [0, 255] to [-1, 1]

        skips = [x]
        for idx, layer in enumerate(self.encoder):
            x = layer(x) # ResBlock

            if idx + 1 < len(self.encoder): # Downscale
                skips.append(x)
                x = F.max_pool2d(x, kernel_size=2)
        
        if self.encoder_pool is not None:
            x = self.encoder_pool(x)

        for idx, layer in enumerate(self.decoder):
            x = self.upscale[idx](x) # Upscale

            x = torch.cat([x, skips.pop()], dim=1) # ResBlock
            x = layer(x)

        x = self.reconstuction_pool(x)

        x = torch.cat([x, skips.pop()], dim=1) # Final skip connection before reconstruction
        if len(skips) != 0: raise IndexError(f"Skip connection mismatch between encoder and decoder. {len(skips)} skip connections are unused.")

        x = self.reconstuction(x)

        x = x * 128 + 128 # Scale output approx from [-1, 1] to [0, 255]
        return x

class _ResBlockA(nn.Module):
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

    def forward(self, x):
        x = F.relu(sum([conv(x) for conv in self.dilations]) + self.respass(x))
        return x

class _PSP_Pooling(nn.Module):
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
        x = [F.interpolate(input=F.max_pool2d(x_chunk, kernel_size=self.sizes[idx]), size=size, mode='bilinear') for idx, x_chunk in enumerate(x)]
        x = [F.relu(self.convs[idx](x_chunk)) for idx, x_chunk in enumerate(x)]
        x = torch.concat(x, dim=1)

        x = F.relu(self.norm_out(self.conv_out(x)))
        return x
