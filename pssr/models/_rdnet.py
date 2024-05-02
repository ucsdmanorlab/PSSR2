"""
RDNet
Copyright (c) 2024-present NAVER Cloud Corp.
Apache-2.0

Source code of RDNet has been modified for use in PSSR.
"""

import torch
import torch.nn as nn
from timm.layers import DropPath, LayerNorm2d, EffectiveSEModule
from timm.models import named_apply
from functools import partial

class RDNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_init_features=128,
        patch_size=2,
        growth_rates=(64, 104, 128, 128, 128, 128, 224),
        ds_blocks=(None, True, True, False, False, False, True),
        block_type=["Block", "Block", "BlockESE", "BlockESE", "BlockESE", "BlockESE", "BlockESE"],
        n_blocks=(3, 3, 3, 3, 3, 3, 3),
        bottleneck_width_ratio=4,
        drop_rate=0.0,
        drop_path_rate=0.0,
        transition_compression_ratio=0.5,
        ls_init_value=1e-6,
    ):
        super().__init__()

        block_type = [block_type] * len(growth_rates) if type(block_type) is str else block_type
        n_blocks = [n_blocks] * len(growth_rates) if type(n_blocks) is int else n_blocks

        if not len(growth_rates) == len(ds_blocks): raise ValueError(f"growth_rates and ds_blocks must have the same length. Given values are {len(growth_rates)} and {len(ds_blocks)} respectively.")
        if not len(growth_rates) == len(block_type): raise ValueError(f"growth_rates and block_type must have the same length. Given values are {len(growth_rates)} and {len(block_type)} respectively.")
        if not len(growth_rates) == len(n_blocks): raise ValueError(f"growth_rates and n_blocks must have the same length. Given values are {len(growth_rates)} and {len(n_blocks)} respectively.")

        # Stem
        self.stem = PatchifyStem(in_channels, n_init_features, patch_size=patch_size)

        # Feature encoder
        self.feature_info = []
        self.num_stages = len(growth_rates)
        curr_stride = 4  # stem_stride
        num_features = n_init_features
        dp_rates = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(n_blocks)).split(n_blocks)]

        dense_stages = []
        for i in range(self.num_stages):
            dense_stage_layers = []
            if i != 0:
                compressed_num_features = int(num_features * transition_compression_ratio / 8) * 8
                k_size = stride = 1
                if ds_blocks[i]:
                    curr_stride *= 2
                    k_size = stride = 2
                dense_stage_layers.append(LayerNorm2d(num_features))
                dense_stage_layers.append(
                    nn.Conv2d(num_features, compressed_num_features, kernel_size=k_size, stride=stride, padding=0)
                )
                num_features = compressed_num_features

            stage = DenseStage(
                num_block=n_blocks[i],
                num_input_features=num_features,
                growth_rate=growth_rates[i],
                bottleneck_width_ratio=bottleneck_width_ratio,
                drop_rate=drop_rate,
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                block_type=block_type[i],
            )
            dense_stage_layers.append(stage)
            num_features += n_blocks[i] * growth_rates[i]

            if i + 1 == self.num_stages or (i + 1 != self.num_stages and ds_blocks[i + 1]):
                self.feature_info += [
                    dict(
                        num_chs=num_features,
                        reduction=curr_stride,
                        module=f'dense_stages.{i}',
                        growth_rate=growth_rates[i],
                    )
                ]
            dense_stages.append(nn.Sequential(*dense_stage_layers))
        self.dense_stages = nn.ModuleList(dense_stages)

        # Initialize weights
        named_apply(partial(_init_weights), self)

        self.ds_blocks = ds_blocks

    def forward(self, x):
        x = self.stem(x)

        skips = []
        for idx, layer in enumerate(self.dense_stages):
            if self.ds_blocks[idx]: # Skip connection before each downsample
                skips.append(x)
            x = layer(x)
        
        return *skips, x

class PatchifyStem(nn.Module):
    def __init__(self, num_input_channels, num_init_features, patch_size):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(num_input_channels, num_init_features, kernel_size=patch_size, stride=patch_size),
            LayerNorm2d(num_init_features),
        )

    def forward(self, x):
        return self.stem(x)
    
class DenseStage(nn.Sequential):
    def __init__(self, num_block, num_input_features, drop_path_rates, growth_rate, **kwargs):
        super().__init__()
        for i in range(num_block):
            layer = DenseBlock(
                num_input_features=num_input_features,
                growth_rate=growth_rate,
                drop_path_rate=drop_path_rates[i],
                block_idx=i,
                **kwargs,
            )
            num_input_features += growth_rate
            self.add_module(f"dense_block{i}", layer)
        self.num_out_features = num_input_features

    def forward(self, init_feature):
        features = [init_feature]
        for module in self:
            new_feature = module(features)
            features.append(new_feature)
        return torch.cat(features, 1)

class DenseBlock(nn.Module):
    def __init__(
        self,
        num_input_features,
        growth_rate,
        bottleneck_width_ratio,
        drop_path_rate,
        drop_rate=0.0,
        rand_gather_step_prob=0.0,
        block_idx=0,
        block_type="Block",
        ls_init_value=1e-6,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.rand_gather_step_prob = rand_gather_step_prob
        self.block_idx = block_idx
        self.growth_rate = growth_rate

        self.gamma = nn.Parameter(ls_init_value * torch.ones(growth_rate)) if ls_init_value > 0 else None
        growth_rate = int(growth_rate)
        inter_chs = int(num_input_features * bottleneck_width_ratio / 8) * 8
        self.drop_path = DropPath(drop_path_rate)
        self.layers = eval(block_type)(
            in_chs=num_input_features,
            inter_chs=inter_chs,
            out_chs=growth_rate,
        )

    def forward(self, x):
        if type(x) is list:
            x = torch.cat(x, 1)
        x = self.layers(x)

        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))
        return x

class Block(nn.Module):
    """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
    def __init__(self, in_chs, inter_chs, out_chs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),
            LayerNorm2d(in_chs, eps=1e-6),
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        return self.layers(x)

class BlockESE(nn.Module):
    """D == Dw conv, N == Norm, F == Feed Forward, A == Activation"""
    def __init__(self, in_chs, inter_chs, out_chs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_chs, in_chs, groups=in_chs, kernel_size=7, stride=1, padding=3),
            LayerNorm2d(in_chs, eps=1e-6),
            nn.Conv2d(in_chs, inter_chs, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(inter_chs, out_chs, kernel_size=1, stride=1, padding=0),
            EffectiveSEModule(out_chs),
        )

    def forward(self, x):
        return self.layers(x)

def _init_weights(module, name=None):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
