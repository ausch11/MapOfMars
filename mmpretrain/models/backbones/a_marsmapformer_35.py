# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone
import pywt
import pywt.data
import torch
from torch import nn
from functools import partial
import torch.nn.functional as F
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer
#

class MultiGranularShuffle(nn.Module):
    def __init__(self, combined_channels, levels=2):
        super().__init__()
        self.levels = levels
        self.group_sizes = self._compute_group_sizes(combined_channels, levels)

    def _compute_group_sizes(self, channels, levels):
        group_sizes = []
        for i in range(levels):
            group_size = channels // (2 ** (i + 1))
            group_sizes.append(group_size)
            if group_size < 1:
                raise ValueError(f"通道数 {channels} 过小，无法支持 {self.levels} 层混洗")
        return group_sizes

    def hierarchical_shuffle(self, x):
        B, C, H, W = x.shape
        for group_size in reversed(self.group_sizes):
            groups = C // group_size
            x = x.view(B, groups, group_size, H, W)
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(B, -1, H, W)
        return x

    def forward(self, x):
        return self.hierarchical_shuffle(x)



class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.avg_pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return s


class DynamicAttentionShuffle(nn.Module):
    def __init__(self, channels, reduction=16, max_groups=8):
        super().__init__()
        self.attention = ChannelAttention(channels, reduction)
        self.max_groups = max_groups

    def dynamic_group_assignment(self, scores, max_groups=8):
        _, idx = torch.sort(scores, dim=1, descending=True)
        group_num = max(1, min(max_groups, int(torch.var(scores) * 10)))
        groups = torch.chunk(idx, group_num, dim=1)
        return groups

    def dynamic_channel_shuffle(self,x, groups):
        shuffled = []
        for group in groups:
            perm = torch.randperm(group.size(1))
            shuffled_group = group[:, perm]
            shuffled.append(x[:, shuffled_group, :, :])
        return torch.cat(shuffled, dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        scores = self.attention(x).squeeze()
        groups = self.dynamic_group_assignment(scores, self.max_groups)
        x = self.dynamic_channel_shuffle(x, groups)
        return x



def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)  # 48,128,2,56,56
    x = torch.transpose(x, 1, 2).contiguous()  # 48,2,128,56,56
    # flatten
    x = x.view(batchsize, -1, height, width) #48,256,56,56
    return x

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

class MSDC(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, dw_parallel=True):
        super(MSDC, self).__init__()

        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.dw_parallel = dw_parallel

        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU6(inplace=True)
            )
            for kernel_size in self.kernel_sizes
        ])


    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        return outputs


class DynamicConv2d(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=7,
                 reduction_ratio=4,
                 num_groups=2):
        super(DynamicConv2d, self).__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size),
                                   requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim,
                       dim // reduction_ratio,
                       kernel_size=3,
                       padding=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'), ),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=3, padding=1),
        )

        self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        bias = None
        # 使用动态生成的权重和偏置进行卷积操作
        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)

        return x.reshape(B, C, H, W)


class PatchEmbed(nn.Module):

    def __init__(self,
                 patch_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Pooling(nn.Module):

    def __init__(self, dim, pool_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, kernel_size=pool_size, stride=1, padding=pool_size // 2, groups=dim)

    def forward(self, x):

        out = self.depthwise(x) - x
        return out


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = build_activation_layer(act_cfg)
        self.fc2 = nn.Conv2d(hidden_features*2, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.msdc = MSDC(hidden_features, kernel_sizes=[3, 7], stride=1)
        self.combined_channels = hidden_features*2
        self.hidden_features = hidden_features  # 128
        # self.das = DynamicAttentionShuffle(hidden_features*2)
        self.multi_granular_shuffle = MultiGranularShuffle(
            self.combined_channels,
            2  # 可配置分层数，默认2层
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.msdc(x)
        x = torch.cat(x, dim=1)
        x = self.multi_granular_shuffle(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(BaseModule):
    def __init__(self,
                 dim,
                 pool_size=3,
                 mlp_ratio=2.,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop=0.,
                 drop_path=0.,
                 layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.token_mixer = Pooling(dim, pool_size=pool_size)

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        self.local_unit = DynamicConv2d(
            dim=dim, kernel_size=7, num_groups=2)


    def forward(self, x):
        x_n = self.norm1(x)

        x_pool = self.token_mixer(x_n)
        x_pool = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x_pool

        x_conv = self.local_unit(x_n)

        x = x + self.drop_path(x_pool+x_conv)

        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim,
                 index,
                 layers,
                 pool_size=3,
                 mlp_ratio=2.,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop_rate=.0,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-5):

    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (
            sum(layers) - 1)
        blocks.append(
            PoolFormerBlock(
                dim,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop=drop_rate,
                drop_path=block_dpr,
                layer_scale_init_value=layer_scale_init_value,
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


@MODELS.register_module()
class MarsMapFormer_35(BaseBackbone):
    arch_settings = {
        's12': {
            'layers': [2, 2, 6, 2],
            'embed_dims': [64, 128, 320, 512],
            'mlp_ratios': [4, 4, 4, 4],
            'layer_scale_init_value': 1e-5,
        },
        's24': {
            'layers': [4, 4, 12, 4],
            'embed_dims': [64, 128, 320, 512],
            'mlp_ratios': [4, 4, 4, 4],
            'layer_scale_init_value': 1e-5,
        },
        's36': {
            'layers': [6, 6, 18, 6],
            'embed_dims': [64, 128, 320, 512],
            'mlp_ratios': [2, 2, 2, 2],
            'layer_scale_init_value': 1e-6,
        },
        'm36': {
            'layers': [6, 6, 18, 6],
            'embed_dims': [96, 192, 384, 768],
            'mlp_ratios': [4, 4, 4, 4],
            'layer_scale_init_value': 1e-6,
        },
        'm48': {
            'layers': [8, 8, 24, 8],
            'embed_dims': [96, 192, 384, 768],
            'mlp_ratios': [4, 4, 4, 4],
            'layer_scale_init_value': 1e-6,
        },
    }

    def __init__(self,
                 arch='s12',
                 pool_size=3,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 in_patch_size=7,
                 in_stride=4,
                 in_pad=2,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 out_indices=-1,
                 frozen_stages=0,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        if isinstance(arch, str):
            assert arch in self.arch_settings, \
                f'Unavailable arch, please choose from ' \
                f'({set(self.arch_settings)}) or pass a dict.'
            arch = self.arch_settings[arch]
        elif isinstance(arch, dict):
            assert 'layers' in arch and 'embed_dims' in arch, \
                f'The arch dict must have "layers" and "embed_dims", ' \
                f'but got {list(arch.keys())}.'

        layers = arch['layers']
        embed_dims = arch['embed_dims']
        mlp_ratios = arch['mlp_ratios'] \
            if 'mlp_ratios' in arch else [2, 2, 2, 2]
        layer_scale_init_value = arch['layer_scale_init_value'] \
            if 'layer_scale_init_value' in arch else 1e-5

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size,
            stride=in_stride,
            padding=in_pad,
            in_chans=3,
            embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1]))

        self.network = nn.ModuleList(network)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = 7 + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices
        if self.out_indices:
            for i_layer in self.out_indices:
                layer = build_norm_layer(norm_cfg,
                                         embed_dims[(i_layer + 1) // 2])[1]
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)

        self.frozen_stages = frozen_stages
        self._freeze_stages()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return tuple(outs)

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        return x

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(self.frozen_stages):
            # Include both block and downsample layer.
            module = self.network[i]
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(MarsMapFormer_35, self).train(mode)
        self._freeze_stages()
