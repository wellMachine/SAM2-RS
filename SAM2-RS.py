import torch
import torch.nn as nn
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from torch import Tensor
from typing import List, Optional
from sam2.var import MyVarFeatureEnhance
from typing import Tuple


###############################
# CoordAtt 及其依赖模块定义
###############################

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()
        # kernel_size 只能为 3 或 7
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 使用全局最大池化获得空间显著性图
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        """
        :param inp: 输入通道数
        :param oup: 输出通道数
        :param reduction: 通道压缩比例，默认32
        """
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_end = nn.Conv2d(oup, oup, kernel_size=1, stride=1, padding=0)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, x):
        n, c, h, w = x.size()
        x_h = self.pool_h(x)  # [n, c, h, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [n, c, w, 1]
        y = torch.cat([x_h, x_w], dim=2)  # 拼接后 [n, c, h+w, 1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        # 将 y 按照 h 和 w 分割
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out_ca = x * a_h * a_w
        out_sa = self.self_SA_Enhance(out_ca)
        out = x * out_sa
        out = self.conv_end(out)
        return out


class UP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UP, self).__init__()
        self.conv_adjust = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.pcm = MLPBlock(dim=out_channels, upsample=False)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_adjust(x)
        x = self.pcm(x)
        return x


class PromptAdapter(nn.Module):
    def __init__(self, blk) -> None:
        super(PromptAdapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x


class Enhancement_texture_LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        # conv.weight.size() = [out_channels, in_channels, kernel_size, kernel_size]
        super(Enhancement_texture_LDC, self).__init__()  # 初始化父类 nn.Module

        # 定义一个二维卷积层，参数为输入输出通道数、卷积核大小、步长、padding、扩张、分组、是否带偏置
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)  # 例如尺寸为[12,3,3,3]

        # 定义中心位置掩码矩阵，3x3矩阵中间为1，其他为0，用于突出卷积核中心权重，放到cuda设备
        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]]).cuda()

        # 定义基准掩码参数，尺寸同卷积核权重，元素全为1，不可训练（requires_grad=False）
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)  # [12,3,3,3]

        # 定义可学习掩码参数，尺寸为[输出通道数, 输入通道数]，初始化为全1，允许训练
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)  # [12,3]

        # 定义可学习的标量theta，初始化为0.5，允许训练，用于调节mask的强度
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # [1]

        # print(self.learnable_mask[:, :, None, None].shape)  # 注释掉的调试打印

    def forward(self, x):
        # 计算动态掩码：
        # 基准掩码减去 learnable_theta * learnable_mask 扩展到4维 * center_mask * 卷积核权重在空间维度的和（中心权重放大）
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * \
               self.center_mask * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        # 使用带掩码的卷积核权重进行卷积操作，卷积输入为x，卷积参数继承原卷积层的偏置、步长、padding和分组
        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)

        return out_diff  # 返回经过增强后的特征图


class Differential_enhance(nn.Module):
    def __init__(self, nf=48):
        super(Differential_enhance, self).__init__()  # 初始化父类 nn.Module

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应全局平均池化，输出大小为1x1
        self.act = nn.Sigmoid()  # Sigmoid 激活函数，用于生成权重系数
        self.lastconv = nn.Conv2d(nf, nf // 2, 1, 1)  # 1x1卷积，通道数从nf降为nf//2（未使用）

    def forward(self, fuse, x1, x2):
        b, c, h, w = x1.shape  # 获取输入x1的批次、通道、高、宽尺寸

        sub_1_2 = x1 - x2  # 计算x1和x2的差异特征（x1减x2）
        sub_w_1_2 = self.global_avgpool(sub_1_2)  # 对差异特征做全局平均池化，得到1x1特征图
        w_1_2 = self.act(sub_w_1_2)  # 通过Sigmoid激活，生成权重系数（0~1范围）

        sub_2_1 = x2 - x1  # 计算x2和x1的差异特征（x2减x1）
        sub_w_2_1 = self.global_avgpool(sub_2_1)  # 对差异特征做全局平均池化，得到1x1特征图
        w_2_1 = self.act(sub_w_2_1)  # 通过Sigmoid激活，生成权重系数（0~1范围）

        D_F1 = torch.multiply(w_1_2, fuse)  # 将权重w_1_2与融合特征fuse逐元素相乘，得到加权差异特征D_F1
        D_F2 = torch.multiply(w_2_1, fuse)  # 将权重w_2_1与融合特征fuse逐元素相乘，得到加权差异特征D_F2

        F_1 = torch.add(D_F1, other=x1, alpha=1)  # 将加权差异特征D_F1与x1相加，融合增强后特征F_1
        F_2 = torch.add(D_F2, other=x2, alpha=1)  # 将加权差异特征D_F2与x2相加，融合增强后特征F_2

        return F_1, F_2  # 返回增强后的两个特征张量


class Cross_layer(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0  # 隐藏维度参数，默认为0
    ):
        super().__init__()  # 初始化父类nn.Module
        self.d_model = hidden_dim  # 保存隐藏维度大小

        # 定义第一个纹理增强模块，输入输出通道均为d_model
        self.texture_enhance1 = Enhancement_texture_LDC(self.d_model, self.d_model)
        # 定义第二个纹理增强模块，输入输出通道均为d_model
        self.texture_enhance2 = Enhancement_texture_LDC(self.d_model, self.d_model)
        # 定义差异增强模块，输入通道为d_model
        self.Diff_enhance = Differential_enhance(self.d_model)

    def forward(self, Fuse, x1, x2):
        TX_x1 = self.texture_enhance1(x1)  # 对输入x1做纹理增强处理
        TX_x2 = self.texture_enhance2(x2)  # 对输入x2做纹理增强处理

        DF_x1, DF_x2 = self.Diff_enhance(Fuse, x1, x2)  # 对Fuse、x1、x2计算差异增强，得到两个结果

        F_1 = TX_x1 + DF_x1  # 将x1的纹理增强和差异增强结果相加
        F_2 = TX_x2 + DF_x2  # 将x2的纹理增强和差异增强结果相加

        return F_1, F_2  # 返回两个增强后的特征


###############################
# Memory 模块
###############################

class MemoryModule(nn.Module):
    """
    可学习的记忆检索模块：根据查询动态选择最相关的记忆项。
    """

    def __init__(self, feature_dim: int, num_prototypes: int):
        super(MemoryModule, self).__init__()
        # 确保 P 的维度与 fq_norm 匹配
        self.P = nn.Parameter(torch.randn(num_prototypes, feature_dim))  # [num_prototypes, feature_dim]
        self.attention_weights = nn.Parameter(torch.ones(num_prototypes))  # [num_prototypes]

    def forward(self, f_q: Tensor) -> Tensor:
        """
        f_q: 查询向量，形状 [B, C]，其中 C 是特征维度
        """
        P_norm = F.normalize(self.P, dim=1)  # [K, C]，K 是原型数量，C 是特征维度
        fq_norm = F.normalize(f_q, dim=1)  # [B, C]，查询向量

        sims = torch.matmul(fq_norm, P_norm.t())  # [B, K]
        sims = sims * self.attention_weights  # [B, K]

        idx = sims.argmax(dim=1)  # 获取最相关的记忆原型索引 [B]
        p_q = self.P[idx]  # [B, C]，选取对应的记忆原型
        return p_q

class CrossAttentionMemory(nn.Module):
    """
    Cross-Attention between spatial feature map x4 and memory prototype p_q.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.scale = feature_dim ** -0.5

    def forward(self, x: Tensor, p_q: Tensor) -> Tensor:
        # x: [B, C, H, W], p_q: [B, C]
        B, C, H, W = x.shape
        # Query: flatten spatial
        q = x.view(B, C, -1).permute(0, 2, 1)  # [B, L, C], L=H*W
        # Key/Value: treat p_q as single token
        k = p_q.unsqueeze(1)  # [B, 1, C]
        v = k  # [B, 1, C]
        # Attention scores
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)  # [B, L, 1]
        # Weighted sum
        out = torch.matmul(attn, v)  # [B, L, C]
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return out


class ECA(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super(ECA, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=self.kernel_size, padding=self.kernel_size//2, groups=in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y)
        return x * self.sigmoid(y)


class FEMRM(nn.Module):
    """FEMRM: Feature Enhancement + Memory Retrieval."""

    def __init__(self, memory_dim: int = 1152, num_prototypes: int = 64) -> None:
        super().__init__()

        # Feature_Enhancement (DFEM via Cross_layer)
        self.cross12 = Cross_layer(144)
        self.cross23 = Cross_layer(288)
        self.cross34 = Cross_layer(576)
        self.conv12_down = nn.Conv2d(288, 144, 1)
        self.conv23_down = nn.Conv2d(576, 288, 1)
        self.conv34_down = nn.Conv2d(1152, 576, 1)

        # Memory_Retrieval (Memory Bank + Cross-Attn)
        self.memory_bank = MemoryModule(memory_dim, num_prototypes)
        self.pool_memory = nn.AdaptiveAvgPool2d(1)
        self.mem_attn = CrossAttentionMemory(memory_dim)

    def forward(
        self, x1: Tensor, x2: Tensor, x3: Tensor, x4: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Feature Enhancement
        x2u = F.interpolate(x2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        p12 = self.conv12_down(x2u)
        x1, _ = self.cross12(x1 + p12, x1, p12)

        x3u = F.interpolate(x3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        p23 = self.conv23_down(x3u)
        x2, _ = self.cross23(x2 + p23, x2, p23)

        x4u = F.interpolate(x4, size=x3.shape[2:], mode="bilinear", align_corners=False)
        p34 = self.conv34_down(x4u)
        x3, _ = self.cross34(x3 + p34, x3, p34)

        # Memory Retrieval & Fusion on x4
        f_q = self.pool_memory(x4).view(x4.size(0), -1)  # [B, C]
        p_q = self.memory_bank(f_q)  # [B, C]
        mem_feat = self.mem_attn(x4, p_q)  # [B, C, H4, W4]
        x4 = x4 + mem_feat

        return x1, x2, x3, x4


class MAD(nn.Module):
    """MAD: Multi-scale RFB + refine + decoder."""

    def __init__(self) -> None:
        super().__init__()

        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)

        self.mam1 = CoordAtt(64, 64)
        self.mam2 = CoordAtt(64, 64)
        self.mam3 = CoordAtt(64, 64)
        self.eca4 = ECA(64)

        self.up1 = UP(128, 64)
        self.up2 = UP(128, 64)
        self.up3 = UP(128, 64)
        self.up4 = UP(128, 64)

        self.side1 = nn.Conv2d(64, 1, 1)
        self.side2 = nn.Conv2d(64, 1, 1)
        self.head = nn.Conv2d(64, 1, 1)

    def forward(self, x1: Tensor, x2: Tensor, x3: Tensor, x4: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # multi-scale RFB
        x1 = self.rfb1(x1)
        x2 = self.rfb2(x2)
        x3 = self.rfb3(x3)
        x4 = self.rfb4(x4)

        # refine
        x1 = self.mam1(x1)
        x2 = self.mam2(x2)
        x3 = self.mam3(x3)
        x4 = self.eca4(x4)

        # decode
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode="bilinear", align_corners=False)
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode="bilinear", align_corners=False)
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode="bilinear", align_corners=False)
        return out, out1, out2


class SAM2_RS(nn.Module):
    """SAM2-RS segmentation model.

    Args:
        sam2_checkpoint_path: Path to SAM2 checkpoint (optional).
        sam2_cfg: SAM2 config name (YAML) used by `build_sam2`.
        memory_dim: Memory embedding dimension.
        num_prototypes: Number of memory prototypes.
    """

    def __init__(
        self,
        sam2_checkpoint_path: Optional[str] = None,
        memory_dim: int = 1152,
        num_prototypes: int = 64,
        sam2_cfg: str = "sam2_hiera_l.yaml",
    ) -> None:
        super().__init__()
        # 1) SAM2 backbone
        model = (
            build_sam2(sam2_cfg, sam2_checkpoint_path)
            if sam2_checkpoint_path
            else build_sam2(sam2_cfg)
        )
        for attr in ["sam_mask_decoder", "sam_prompt_encoder",
                     "memory_encoder", "memory_attention",
                     "mask_downsample", "obj_ptr_tpos_proj", "obj_ptr_proj",
                     "image_encoder.neck"]:
            if hasattr(model, attr):
                delattr(model, attr)
        self.encoder = model.image_encoder.trunk
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.blocks = nn.Sequential(*[PromptAdapter(b) for b in self.encoder.blocks])

        self.femrm = FEMRM(memory_dim=memory_dim, num_prototypes=num_prototypes)
        self.mad = MAD()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # backbone
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.femrm(x1, x2, x3, x4)
        return self.mad(x1, x2, x3, x4)

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class MLPBlock(nn.Module):
    def __init__(self,
                 dim,
                 n_div=2,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=nn.BatchNorm2d,
                 pconv_fw_type='split_cat',
                 upsample=True):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]
        self.mlp = nn.Sequential(*mlp_layer)
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)
        self.upsample_flag = upsample
        if upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = None

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.mlp(x)
        if self.up is not None:
            x = self.up(x)
        return x


if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2_RS().cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        print(out.shape, out1.shape, out2.shape)
