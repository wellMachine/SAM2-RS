# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import pywt.data

from sam2.modeling.backbones.utils import (
    PatchEmbed,
    window_partition,
    window_unpartition,
)

from sam2.modeling.sam2_utils import DropPath, MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt  # 小波变换库


class AdapterOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        # 第一层卷积操作：使用3x3的卷积核，padding=1，depthwise卷积（每个输入通道有独立的卷积核）
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
        # 第二层卷积操作：使用5x5的卷积核，padding=2，depthwise卷积
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
        # 第三层卷积操作：使用7x7的卷积核，padding=3，depthwise卷积
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
        # 1x1卷积，用于通道映射，减少计算量，输出通道数不变
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        # 存储原始输入，后面用于残差连接
        identity = x
        # 对输入x分别进行3x3, 5x5, 7x7卷积操作
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        # 将3个卷积的输出加权平均后与原始输入相加形成残差连接
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        # 更新identity为新计算出的x
        identity = x
        # 通过1x1卷积进行通道调整
        x = self.projector(x)
        # 返回通过残差连接的最终输出
        return identity + x


class BottleneckAdapter:
    def __init__(self,
                 in_dim,
                 factor=4,
                 fft_rate=0.25):  # 添加一个 fft_rate 参数，用于控制频域处理的程度
        super().__init__()
        # 投影层，将输入维度(in_dim)映射到64维
        self.project1 = nn.Linear(in_dim, 64)
        # 使用GELU作为激活函数
        self.nonlinear = F.gelu
        # 第二个投影层，将64维重新映射回原始的in_dim维度
        self.project2 = nn.Linear(64, in_dim)
        # Dropout层，用于防止过拟合，dropout概率为0.1
        self.dropout = nn.Dropout(p=0.1)
        # 用AdapterOp对64维特征进行处理
        self.adapter_conv = AdapterOp(64)
        # 对输入进行层归一化
        self.norm = nn.LayerNorm(in_dim)
        # gamma参数用于缩放归一化后的输入
        self.gamma = nn.Parameter(torch.ones(in_dim) * 1e-6)
        # gammax参数用于与输入的加权融合
        self.gammax = nn.Parameter(torch.ones(in_dim))
        # FFT处理
        self.fft_rate = fft_rate

    def fft(self, x, rate):
        """
        使用快速傅里叶变换 (FFT) 提取频率特征
        参数 rate 控制滤波器的尺寸（rate 越小，图像越平滑；rate 越大，图像越暗）
        """
        mask = torch.zeros(x.shape).to(x.device)
        w, h = x.shape[-2:]  # 获取图像的宽和高
        line = int((w * h * rate) ** .5 // 2)
        mask[:, :, w // 2 - line:w // 2 + line, h // 2 - line:h // 2 + line] = 1

        # 对输入 x 进行二维 FFT，并将零频率成分移到中心
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
        fft = fft * (1 - mask)  # 高通滤波

        # 对 FFT 结果进行逆变换并取绝对值
        fr = fft.real
        fi = fft.imag
        fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
        inv = torch.fft.ifft2(fft_hires, norm="forward").real
        inv = torch.abs(inv)  # 取模

        return inv  # 返回频域特征

    def wavelet_transform(self, x):
        """应用小波变换到输入张量x"""
        # 使用PyWavelets进行二维离散小波变换
        # 假设x的形状为 [batch_size, channels, height, width]
        b, c, h, w = x.shape
        # 将张量展平为二维 (batch_size, c * h * w)
        x = x.view(b, c, -1)

        # 对每个通道分别执行小波变换
        coeffs = []
        for i in range(c):
            coeffs.append(pywt.dwt2(x[:, i, :].cpu().numpy(), 'haar'))  # 使用Haar小波进行DWT
        # 合并小波变换的系数（这里只是示范）
        transformed = torch.tensor(coeffs)
        return transformed

    def forward(self, x, hw_shapes=None):
        identity = x

        # 先进行 FFT 频域处理
        fft_feature = self.fft(x, self.fft_rate)  # 获取 FFT 特征

        # 将 FFT 特征与原始输入拼接（可以选择拼接或直接替换）
        x = torch.cat([x, fft_feature], dim=1)  # 这里选择拼接在通道维度

        # 先应用小波变换
        x = self.wavelet_transform(x)

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2


def do_pool(x: torch.Tensor, pool: nn.Module, norm: nn.Module = None) -> torch.Tensor:
    if pool is None:
        return x
    # (B, H, W, C) -> (B, C, H, W)
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    # (B, C, H', W') -> (B, H', W', C)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)

    return x


class MultiScaleAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            num_heads: int,
            q_pool: nn.Module = None,
    ):
        super().__init__()

        self.dim = dim
        self.dim_out = dim_out

        self.num_heads = num_heads
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5

        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (B, H * W, 3, nHead, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        # q, k, v with shape (B, H * W, nheads, C)
        q, k, v = torch.unbind(qkv, 2)

        # Q pooling (for downsample at stage changes)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]  # downsampled shape
            q = q.reshape(B, H * W, self.num_heads, -1)

        # Torch's SDPA expects [B, nheads, H*W, C] so we transpose
        x = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        # Transpose back
        x = x.transpose(1, 2)
        x = x.reshape(B, H, W, -1)

        x = self.proj(x)

        return x


class MultiScaleBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            num_heads: int,
            mlp_ratio: float = 4.0,
            drop_path: float = 0.0,
            norm_layer: Union[nn.Module, str] = "LayerNorm",
            q_stride: Tuple[int, int] = None,
            act_layer: nn.Module = nn.GELU,
            window_size: int = 0,
            config=None
    ):
        super().__init__()

        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)

        self.window_size = window_size

        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(
                kernel_size=q_stride, stride=q_stride, ceil_mode=False
            )

        self.attn = MultiScaleAttention(
            dim,
            dim_out,
            num_heads=num_heads,
            q_pool=self.pool,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(
            dim_out,
            int(dim_out * mlp_ratio),
            dim_out,
            num_layers=2,
            activation=act_layer,
        )

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)
        if config and getattr(config, "ffn_adapt", False):
            adapter_factor = getattr(config, "adapter_factor", 8)
            # BottleneckAdapter: d_model -> 输入通道维度，bottleneck -> 瓶颈维度
            self.adaptmlp = BottleneckAdapter(
                d_model=dim_out,
                bottleneck=adapter_factor,
                dropout=0.1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  # B, H, W, C
        x = self.norm1(x)

        # Skip connection
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)

        # Window partition
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)

        # Window Attention + Q Pooling (if stage change)
        x = self.attn(x)
        if self.q_stride:
            # Shapes have changed due to Q pooling
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]

            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)

        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)

        # # Apply Adapter before MLP if required
        # if hasattr(self, 'adaptmlp'):
        #     x = self.adaptmlp(x)
        # **在 MLP 之前执行 BottleneckAdapter 适配器**
        if hasattr(self, 'adaptmlp'):
            # x: (B, H, W, C) -> (B, H*W, C)
            B, H, W, C = x.shape
            x_flat = x.view(B, H * W, C)
            # BottleneckAdapter 不需要 hw_shapes
            x_flat = self.adaptmlp(x_flat)
            # 恢复到 (B, H, W, C)
            x = x_flat.view(B, H, W, C)

        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Hiera(nn.Module):
    """
    Reference: https://arxiv.org/abs/2306.00989
    """

    def __init__(
            self,
            embed_dim: int = 96,  # initial embed dim
            num_heads: int = 1,  # initial number of heads
            drop_path_rate: float = 0.0,  # stochastic depth
            q_pool: int = 3,  # number of q_pool stages
            q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
            stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
            dim_mul: float = 2.0,  # dim_mul factor at stage shift
            head_mul: float = 2.0,  # head_mul factor at stage shift
            window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
            # window size per stage, when not using global att.
            window_spec: Tuple[int, ...] = (
                    8,
                    4,
                    14,
                    7,
            ),
            # global attn in these blocks
            global_att_blocks: Tuple[int, ...] = (
                    12,
                    16,
                    20,
            ),
            return_interm_layers=True,  # return feats from every stage
            config=None
    ):
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size)
        )
        self.pos_embed_window = nn.Parameter(
            torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0])
        )

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
                config=config  # Pass config to the block here
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile(
            [x // y for x, y in zip(pos_embed.shape, window_embed.shape)]
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                    i in self.stage_ends and self.return_interm_layers
            ):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs
