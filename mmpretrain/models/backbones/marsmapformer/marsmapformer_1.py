# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

import torch
import torch.nn as nn
from mmcv.cnn.bricks import DropPath, build_activation_layer, build_norm_layer
from mmengine.model import BaseModule

from mmpretrain.registry import MODELS
from .base_backbone import BaseBackbone
from torch.nn import functional as F
from mmcv.cnn.bricks import ConvModule, build_activation_layer, build_norm_layer


class PatchEmbed(nn.Module):
    """Patch Embedding module implemented by a layer of convolution.

    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    Args:
        patch_size (int): Patch size of the patch embedding. Defaults to 16.
        stride (int): Stride of the patch embedding. Defaults to 16.
        padding (int): Padding of the patch embedding. Defaults to 0.
        in_chans (int): Input channels. Defaults to 3.
        embed_dim (int): Output dimension of the patch embedding.
            Defaults to 768.
        norm_layer (module): Normalization module. Defaults to None (not use).
    """

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
    """Pooling module.

    Args:
        pool_size (int): Pooling size. Defaults to 3.
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """Mlp implemented by with 1*1 convolutions.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

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
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):  ### OSRA
    def __init__(self, dim,
                 num_heads=1,
                 qk_scale=None,
                 attn_drop=0,
                 sr_ratio=1, ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                ConvModule(dim, dim,
                           kernel_size=sr_ratio + 3,
                           stride=sr_ratio,
                           padding=(sr_ratio + 3) // 2,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=dict(type='GELU')),
                ConvModule(dim, dim,
                           kernel_size=1,
                           groups=dim,
                           bias=False,
                           norm_cfg=dict(type='BN2d'),
                           act_cfg=None, ), )
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = F.interpolate(relative_pos_enc, size=attn.shape[2:],
                                                 mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)


class DynamicConv2d(nn.Module):  ### IDConv
    def __init__(self,
                 dim,
                 kernel_size=3,
                 reduction_ratio=4,
                 num_groups=1,
                 bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            ConvModule(dim,
                       dim // reduction_ratio,
                       kernel_size=1,
                       norm_cfg=dict(type='BN2d'),
                       act_cfg=dict(type='GELU'), ),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1), )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

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

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, H, W),
                     weight=weight,
                     padding=self.K // 2,
                     groups=B * C,
                     bias=bias)

        return x.reshape(B, C, H, W)


class PoolFormerBlock(BaseModule):
    """PoolFormer Block.

    Args:
        dim (int): Embedding dim.
        pool_size (int): Pooling size. Defaults to 3.
        mlp_ratio (float): Mlp expansion ratio. Defaults to 4.
        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='GN', num_groups=1)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        drop (float): Dropout rate. Defaults to 0.
        drop_path (float): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float): Init value for Layer Scale.
            Defaults to 1e-5.
    """

    def __init__(self,
                 dim,
                 pool_size=3,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop=0.,
                 drop_path=0.,
                 layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        # self.token_mixer = Pooling(pool_size=pool_size)
        self.token_mixer = Pooling(pool_size=pool_size)

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
             layer_scale_init_value * torch.ones((dim//2)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
             layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        #####
        self.local_unit = DynamicConv2d(
            dim=dim // 2, kernel_size=7, num_groups=2)
        reduction_ratio = 8
        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim), )

    def forward(self, x):
        # x = x + self.drop_path(
        #     self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
        #     self.token_mixer(self.norm1(x)))
        # x = x + self.drop_path(
        #     self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
        #     self.mlp(self.norm2(x)))
        x_ori = x

        x = self.norm1(x)
        '''HybridTokenMixer'''
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)

        x2 = self.token_mixer(x2)
        x2 = self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * x2

        x = torch.cat([x1, x2], dim=1)

        x = self.proj(x)
        x = self.proj(x) + x  ## STE

        x = x_ori + self.drop_path(x)
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
            self.mlp(self.norm2(x)))

        return x


def basic_blocks(dim,
                 index,
                 layers,
                 pool_size=3,
                 mlp_ratio=4.,
                 norm_cfg=dict(type='GN', num_groups=1),
                 act_cfg=dict(type='GELU'),
                 drop_rate=.0,
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-5):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
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
class MarsMapFormer_1(BaseBackbone):
    """PoolFormer.

    A PyTorch implementation of PoolFormer introduced by:
    `MetaFormer is Actually What You Need for Vision <https://arxiv.org/abs/2111.11418>`_

    Modified from the `official repo
    <https://github.com/sail-sg/poolformer/blob/main/models/poolformer.py>`.

    Args:
        arch (str | dict): The model's architecture. If string, it should be
            one of architecture in ``PoolFormer.arch_settings``. And if dict, it
            should include the following two keys:

            - layers (list[int]): Number of blocks at each stage.
            - embed_dims (list[int]): The number of channels at each stage.
            - mlp_ratios (list[int]): Expansion ratio of MLPs.
            - layer_scale_init_value (float): Init value for Layer Scale.

            Defaults to 'S12'.

        norm_cfg (dict): The config dict for norm layers.
            Defaults to ``dict(type='LN2d', eps=1e-6)``.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='GELU')``.
        in_patch_size (int): The patch size of input image patch embedding.
            Defaults to 7.
        in_stride (int): The stride of input image patch embedding.
            Defaults to 4.
        in_pad (int): The padding of input image patch embedding.
            Defaults to 2.
        down_patch_size (int): The patch size of downsampling patch embedding.
            Defaults to 3.
        down_stride (int): The stride of downsampling patch embedding.
            Defaults to 2.
        down_pad (int): The padding of downsampling patch embedding.
            Defaults to 1.
        drop_rate (float): Dropout rate. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        out_indices (Sequence | int): Output from which network position.
            Index 0-6 respectively corresponds to
            [stage1, downsampling, stage2, downsampling, stage3, downsampling, stage4]
            Defaults to -1, means the last stage.
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        init_cfg (dict, optional): Initialization config dict
    """  # noqa: E501

    # --layers: [x,x,x,x], numbers of layers for the four stages
    # --embed_dims, --mlp_ratios:
    #     embedding dims and mlp ratios for the four stages
    # --downsamples: flags to apply downsampling or not in four blocks
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
            'mlp_ratios': [4, 4, 4, 4],
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
            if 'mlp_ratios' in arch else [4, 4, 4, 4]
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
        super(MarsMapFormer_1, self).train(mode)
        self._freeze_stages()
