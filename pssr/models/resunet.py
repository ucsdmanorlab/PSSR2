import torch
import torch.nn as nn
import torch.nn.functional as F
from ._blocks import PSP_Pooling, Reconstruction, get_resblock
from ..util import _force_list

class ResUNet(nn.Module):
    def __init__(
            self,
            channels : list[int] = 1,
            hidden : list[int] = [64, 128, 256, 512, 1024],
            scale : int = 4,
            depth : int = 3,
            dilations : list[list[int]] = None,
            pool_sizes : list[int] = None,
            encoder_pool : bool = False,
        ):
        r"""A Residual UNet as detailed in Zhang et al., 2017 with an additional image upscaling block.
        If ``dilations`` is provided, instead use a Atrous Residual UNet as detailed in Diakogiannis et al., 2019.

        Channel sizes hidden[0] (and hidden[-1] if encoder_pool is True) must be divisible by pool_sizes if provided.

        Args:
            channels (list[int]) : Number of channels in image data. Can also be a list of in channels (low-resolution) and out channels (high-resolution) respectively.

            hidden (list[int]) : Elementwise list of channels per residual block controlling width and length of model.

            scale (int) : Upscaling factor for predictions. Choose a power of 2 for best results. Default is 4.

            depth (int) : Number of hidden layers per residual block. Default is 3.

            dilations (list[list[int]]) : List of dilation values per layer. If value is None, atrous convolutions will not be used. Default is None.

            pool_sizes (list[int]) : Pooling ratios for PSP pooling. If value is None, PSP pooling will not be used. Default is None.

            encoder_pool (bool) : Whether to include additional PSP pooling layer at end of encoder. Should not be used if last layer has a size of less than 16 pixels. Default is False.
        """
        super().__init__()
        channels = _force_list(channels)
        channels = channels*2 if len(channels) == 1 else channels

        if dilations and len(dilations) != len(hidden): raise ValueError(f"Amount of dilations must equal amount of hidden residual blocks. Given values are {len(dilations)} and {len(hidden)} respectively.")

        if pool_sizes:
            if hidden[0] % len(pool_sizes) != 0: raise ValueError(f"hidden[0] must be divisible by len(pool_sizes). Given values are {hidden[0]} and {len(pool_sizes)} respectively.")
            if encoder_pool and hidden[-1] % len(pool_sizes) != 0: raise ValueError(f"hidden[-1] must be divisible by len(pool_sizes) if encoder_pool is True. Given values are {hidden[-1]} and {len(pool_sizes)} respectively.")
        else:
            if encoder_pool: raise ValueError(f"encoder_pool cannot be True if pool_sizes are not provided.")

        self.norm = nn.BatchNorm2d(channels[0]) if not dilations else None

        self.encoder, self.decoder = nn.ModuleList(), nn.ModuleList()
        layers = [channels[0], *hidden]
        n_layers = len(layers) - 1
        for layer_idx in range(n_layers):
            self.encoder.append(get_resblock(in_channels=layers[layer_idx], out_channels=layers[layer_idx+1], dilations=dilations[layer_idx] if dilations else None, depth=depth))
            if layer_idx + 1 < n_layers:
                self.decoder.append(get_resblock(in_channels=layers[-layer_idx-1]-int(layers[-layer_idx-2]/2), out_channels=layers[-layer_idx-2], dilations=dilations[-layer_idx-1] if dilations else None, depth=depth))

        self.encoder_pool = PSP_Pooling(hidden[-1], pool_sizes) if pool_sizes and encoder_pool else None
        self.reconstruction_pool = PSP_Pooling(hidden[0], pool_sizes) if pool_sizes else None

        self.reconstruction = Reconstruction(channels[0], channels[1], hidden[0], scale)

    def forward(self, x):
        x = x / 128 - 1 # Scale input approx from [0, 255] to [-1, 1]
        if self.norm is not None:
            x = self.norm(x)

        skips = [x]
        for idx, layer in enumerate(self.encoder):
            x = layer(x) # ResBlock

            if idx + 1 < len(self.encoder): # Downscale
                skips.append(x)
                x = F.max_pool2d(x, kernel_size=2)

        if self.encoder_pool is not None:
            x = self.encoder_pool(x)

        for idx, layer in enumerate(self.decoder):
            x = x = F.pixel_shuffle(x, 2) # Upscale

            x = torch.cat([x, skips.pop()], dim=1) # ResBlock
            x = layer(x)

        if self.reconstruction_pool is not None:
            x = self.reconstruction_pool(x)

        x = torch.cat([x, skips.pop()], dim=1) # Final skip connection before reconstruction
        if len(skips) != 0: raise IndexError(f"Skip connection mismatch between encoder and decoder. {len(skips)} skip connections are unused.")

        x = self.reconstruction(x)

        x = x * 128 + 128 # Scale output approx from [-1, 1] to [0, 255]
        return x
    
    def extra_repr(self):
        return f"{'Atrous ' if self.norm is None else ''}ResUNet with {self.reconstruction.scale}x upscaling\n{len(self.encoder)} residual decoder blocks with {self.encoder[0].depth} hidden layers each\nPSP pooling {'enabled' if self.reconstruction_pool else 'disabled'}"

class ResUNetA():
    def __new__(cls,
            channels : int = 1,
            hidden : list[int] = [64, 128, 256, 512, 1024],
            scale : int = 4,
            depth : int = 3,
            dilations : list[list[int]] = [[1,3,15,31],[1,3,15],[1,3],[1],[1]],
            pool_sizes : list[int] = [1, 2, 4, 8],
            encoder_pool : bool = False,
        ):
        r""":class:`ResUNet` wrapper of Atrous Residual UNet as detailed in Diakogiannis et al., 2019.
        Provides alternative default arguments for an atrous network.

        Channel sizes hidden[0] (and hidden[-1] if encoder_pool is True) must be divisible by pool_sizes.

        Args:
            channels (int) : Number of channels in image data. Can also be a list of in channels and out channels respectively.

            hidden (list[int]) : Elementwise list of channels per residual block controlling width and length of model.

            scale (int) : Upscaling factor for predictions. Choose a power of 2 for best results. Default is 4.

            depth (int) : Number of hidden layers per residual block. Default is 3.

            dilations (list[list[int]]) : List of dilation values per layer. If value is None, atrous convolutions will not be used. Default is [[1,3,15,31],[1,3,15],[1,3],[1],[1]].

            pool_sizes (list[int]) : Pooling ratios for PSP pooling. If value is None, PSP pooling will not be used. Default is [1, 2, 4, 8].

            encoder_pool (bool) : Whether to include additional PSP pooling layer at end of encoder. Should not be used if last layer has a size of less than 16 pixels. Default is False.
        """
        return ResUNet(
            channels,
            hidden,
            scale,
            depth,
            dilations,
            pool_sizes,
            encoder_pool,
        )
