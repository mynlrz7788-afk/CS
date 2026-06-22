# -*- coding: utf-8 -*-
"""
RD_v1_C: PEFT-DINO + ResNet34 CNN + plain add fusion + RD decoder.

基础消融实验：
1) 保留 RD_v4 的 PEFT-DINOv3：DINO backbone frozen，只训练 adapter 和 gamma；
2) 保留 RD_v4 的 ResNet34 CNN encoder 和 HL_base-style residual decoder；
3) 删除 Road Prior 与 DLAEM4/8/16；
4) DINO A2/A5/A8/A11 只做必要的通道投影和尺度对齐，分别得到 s1/s2/s3/s4；
5) 直接相加：x1+s1, x2+s2, x3+s3, x4+s4；不使用 gate、attention、concat fusion、可学习放缩系数。
"""

import os
import sys
import math
import importlib
from typing import Dict, Tuple, Sequence, List, Optional, Iterable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["RD_v1_C"]


# =========================================================
# 0. Basic utils
# =========================================================
def _auto_padding(kernel_size, dilation=1):
    if isinstance(kernel_size, tuple):
        if isinstance(dilation, tuple):
            return tuple(((k - 1) // 2) * d for k, d in zip(kernel_size, dilation))
        return tuple(((k - 1) // 2) * dilation for k in kernel_size)
    return ((kernel_size - 1) // 2) * dilation


def _make_gn(channels: int, max_groups: int = 8):
    channels = int(channels)
    groups = min(int(max_groups), channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _logit(x: float) -> float:
    x = float(min(max(x, 1e-6), 1.0 - 1e-6))
    return math.log(x / (1.0 - x))


class BoundedScalar(nn.Module):
    """Learnable scalar in [0, max_value]."""
    def __init__(self, max_value: float, init_value: float):
        super().__init__()
        self.max_value = float(max_value)
        ratio = float(init_value) / max(self.max_value, 1e-8)
        self.raw = nn.Parameter(torch.tensor(_logit(ratio), dtype=torch.float32))

    def forward(self):
        return self.max_value * torch.sigmoid(self.raw)


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        act_layer=nn.ReLU,
        inplace=True,
    ):
        super().__init__()
        if padding is None:
            padding = _auto_padding(kernel_size, dilation)
        if act_layer is nn.ReLU:
            act = act_layer(inplace=inplace)
        else:
            act = act_layer()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            act,
        )

    def forward(self, x):
        return self.block(x)


class ConvGNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        act_layer=nn.GELU,
        gn_groups=8,
    ):
        super().__init__()
        if padding is None:
            padding = _auto_padding(kernel_size, dilation)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            _make_gn(out_channels, gn_groups),
            act_layer(),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1, gn_groups=8, act=True):
        super().__init__()
        padding = _auto_padding(kernel_size, dilation)
        layers = [
            nn.Conv2d(in_ch, in_ch, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_ch, bias=False),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            _make_gn(out_ch, gn_groups),
        ]
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ECALayer(nn.Module):
    """Efficient Channel Attention."""
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


# =========================================================
# 1. PEFT-DINOv3 Global Branch
# =========================================================
class BottleneckAdapter(nn.Module):
    """Transformer-block 内部的瓶颈 Adapter: C -> r -> C。"""
    def __init__(self, dim: int = 384, bottleneck: int = 32, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(bottleneck, dim)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        # up 零初始化仍保持初始输出为 0；gamma 非零后，up.weight 能获得梯度。
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.drop(self.act(self.down(x))))


class AdapterSelfAttentionBlock(nn.Module):
    """
    DINOv3 SelfAttentionBlock wrapper。

    目标公式：
        x1 = x + LS1(MHA(LN(x))) + gamma_a * Adapter_a(MHA(LN(x)))
        x2 = x1 + LS2(FFN(LN(x1))) + gamma_f * Adapter_f(LN(x1))

    这里保留 old_block 的 norm/attn/mlp/layerscale；只额外训练 adapter 和 gamma。
    """
    def __init__(self, old_block: nn.Module, dim: int = 384, bottleneck: int = 32, dropout: float = 0.0):
        super().__init__()
        self.old_block = old_block
        self.adapter_attn = BottleneckAdapter(dim=dim, bottleneck=bottleneck, dropout=dropout)
        self.adapter_ffn = BottleneckAdapter(dim=dim, bottleneck=bottleneck, dropout=dropout)
        self.gamma_attn = nn.Parameter(torch.ones(1, 1, dim) * 1e-3)
        self.gamma_ffn = nn.Parameter(torch.ones(1, 1, dim) * 1e-3)

    def _call_attn(self, x, rope=None):
        # 不同 DINOv3 源码中 Attention 的参数名可能略有差异。
        try:
            return self.old_block.attn(x, rope=rope)
        except TypeError:
            try:
                return self.old_block.attn(x, rope)
            except TypeError:
                return self.old_block.attn(x)

    def _ls1(self, x):
        return self.old_block.ls1(x) if hasattr(self.old_block, "ls1") else x

    def _ls2(self, x):
        return self.old_block.ls2(x) if hasattr(self.old_block, "ls2") else x

    def _forward_tensor(self, x, rope=None):
        norm1 = self.old_block.norm1(x)
        attn_out = self._call_attn(norm1, rope=rope)
        x1 = x + self._ls1(attn_out) + self.gamma_attn * self.adapter_attn(attn_out)

        norm2 = self.old_block.norm2(x1)
        ffn_out = self.old_block.mlp(norm2)
        x2 = x1 + self._ls2(ffn_out) + self.gamma_ffn * self.adapter_ffn(norm2)
        return x2

    def forward(self, x_or_x_list, *args, **kwargs):
        rope = kwargs.pop("rope", None)
        if rope is None and len(args) > 0:
            rope = args[0]

        if torch.is_tensor(x_or_x_list):
            return self._forward_tensor(x_or_x_list, rope=rope)

        # 兼容 DINOv3 的 nested/list forward。
        if rope is None:
            rope_list = [None for _ in x_or_x_list]
        elif isinstance(rope, (list, tuple)):
            rope_list = list(rope)
        else:
            rope_list = [rope for _ in x_or_x_list]

        outs = []
        for x, r in zip(x_or_x_list, rope_list):
            outs.append(self._forward_tensor(x, rope=r))
        return outs


def _import_dino_backbones(dino_repo_path: str):
    """兼容 SEG/dinounet/dinov3 和 SEG/dinov3 两种目录放置方式。"""
    dino_repo_path = os.path.abspath(dino_repo_path)
    candidates = []

    # dino_repo_path = .../SEG/dinounet/dinov3 时：
    candidates.append((os.path.abspath(os.path.join(dino_repo_path, "..", "..")), "dinounet.dinov3.hub.backbones"))
    candidates.append((os.path.abspath(os.path.join(dino_repo_path, "..")), "dinov3.hub.backbones"))

    # dino_repo_path = .../SEG/dinov3 时：
    candidates.append((os.path.abspath(os.path.join(dino_repo_path, "..")), "dinov3.hub.backbones"))

    last_err = None
    for path, module_name in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
        try:
            return importlib.import_module(module_name)
        except Exception as e:
            last_err = e
    raise ImportError(f"无法导入 DINOv3 backbones，请检查 dino_repo_path={dino_repo_path}；最后错误: {repr(last_err)}")


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    clean_state = {}
    for k, v in state_dict.items():
        nk = k
        # 一次只去一个前缀，循环直到干净。
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "backbone.", "student.", "teacher.", "model."):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]
                    changed = True
        clean_state[nk] = v
    return clean_state


def build_dinov3_backbone(model_name: str, ckpt_path: str, dino_repo_path: str):
    backbones = _import_dino_backbones(dino_repo_path)
    if not hasattr(backbones, model_name):
        available = [n for n in dir(backbones) if n.startswith("dinov3_")]
        raise AttributeError(f"dinov3.hub.backbones 中找不到 {model_name}，可用示例: {available[:30]}")

    model = getattr(backbones, model_name)(pretrained=False)

    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"找不到 DINOv3 权重文件: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state = ckpt["model"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            elif "teacher" in ckpt:
                state = ckpt["teacher"]
            else:
                state = ckpt
        else:
            state = ckpt
        state = _clean_state_dict(state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[RD_v4] 加载 DINOv3 权重完成: {ckpt_path}\n"
            f"        missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
        )
        if len(missing) > 0:
            print(f"        missing 示例: {missing[:10]}")
        if len(unexpected) > 0:
            print(f"        unexpected 示例: {unexpected[:10]}")
    return model


def insert_adapters_into_dino(backbone: nn.Module, embed_dim: int = 384, bottleneck: int = 32, dropout: float = 0.0):
    if not hasattr(backbone, "blocks"):
        raise RuntimeError("当前 DINOv3 backbone 没有 blocks，无法插入 Adapter。")
    new_blocks = nn.ModuleList()
    for blk in backbone.blocks:
        new_blocks.append(AdapterSelfAttentionBlock(blk, dim=embed_dim, bottleneck=bottleneck, dropout=dropout))
    backbone.blocks = new_blocks
    return backbone


def freeze_dino_except_adapters(backbone: nn.Module):
    for _, p in backbone.named_parameters():
        p.requires_grad = False
    trainable_keywords = ("adapter_attn", "adapter_ffn", "gamma_attn", "gamma_ffn")
    for name, p in backbone.named_parameters():
        if any(k in name for k in trainable_keywords):
            p.requires_grad = True


class PEFTDINOv3GlobalBranch(nn.Module):
    """输入原图，输出 A2/A5/A8/A11 token bank。"""
    def __init__(
        self,
        dino_model_name="dinov3_vits16",
        dino_repo_path="/home/u2508183004/zyn/SEG/dinounet/dinov3",
        dino_ckpt_path="/home/u2508183004/zyn/SEG/weight/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        out_layers=(2, 5, 8, 11),
        embed_dim=384,
        patch_size=16,
        adapter_bottleneck=64,
        adapter_dropout=0.0,
        dino_normalize=False,
        dino_intermediate_norm=True,
    ):
        super().__init__()
        self.out_layers = tuple(int(x) for x in out_layers)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.dino_normalize = bool(dino_normalize)
        self.dino_intermediate_norm = bool(dino_intermediate_norm)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.backbone = build_dinov3_backbone(dino_model_name, dino_ckpt_path, dino_repo_path)
        self.backbone = insert_adapters_into_dino(
            self.backbone,
            embed_dim=self.embed_dim,
            bottleneck=int(adapter_bottleneck),
            dropout=float(adapter_dropout),
        )
        freeze_dino_except_adapters(self.backbone)

        dino_total = sum(p.numel() for p in self.backbone.parameters())
        dino_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(
            f"[RD_v4] PEFT-DINOv3 ready. layers={self.out_layers}, patch={self.patch_size}, "
            f"DINO total={dino_total / 1e6:.2f}M, trainable={dino_trainable / 1e6:.2f}M"
        )

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for p in self.backbone.parameters():
            if p.requires_grad:
                yield p

    def train(self, mode: bool = True):
        # eval/train 不影响 requires_grad。DINO 主干没有 BN；这里不强制 no_grad。
        super().train(mode)
        return self

    def _feat_to_tokens(self, feat, h: int, w: int):
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        patch_h = h // self.patch_size
        patch_w = w // self.patch_size
        patch_n = patch_h * patch_w

        if feat.dim() == 4:
            b, c, fh, fw = feat.shape
            if (fh, fw) != (patch_h, patch_w):
                feat = F.interpolate(feat, size=(patch_h, patch_w), mode="bilinear", align_corners=False)
            return feat.flatten(2).transpose(1, 2).contiguous()

        if feat.dim() != 3:
            raise RuntimeError(f"DINO 特征维度异常: {feat.shape}")

        b, n, c = feat.shape
        if n > patch_n:
            feat = feat[:, n - patch_n:, :]
        if feat.shape[1] != patch_n:
            raise RuntimeError(
                f"DINO token 无法对齐 patch grid: got={feat.shape[1]}, expected={patch_n}, input={h}x{w}"
            )
        return feat.contiguous()

    def forward(self, x):
        if self.dino_normalize:
            x = (x - self.mean) / self.std
        _, _, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise RuntimeError(f"DINO 输入尺寸必须能被 patch_size 整除，当前 {h}x{w}, patch={self.patch_size}")
        if not hasattr(self.backbone, "get_intermediate_layers"):
            raise RuntimeError("当前 DINOv3 backbone 没有 get_intermediate_layers。")

        # 不能 no_grad：adapter 需要反向传播。
        try:
            feats = self.backbone.get_intermediate_layers(
                x,
                n=self.out_layers,
                reshape=False,
                return_class_token=False,
                return_extra_tokens=False,
                norm=self.dino_intermediate_norm,
            )
        except TypeError:
            try:
                feats = self.backbone.get_intermediate_layers(
                    x,
                    n=self.out_layers,
                    reshape=False,
                    return_class_token=False,
                    norm=self.dino_intermediate_norm,
                )
            except TypeError:
                feats = self.backbone.get_intermediate_layers(
                    x,
                    n=self.out_layers,
                    reshape=False,
                    return_class_token=False,
                )

        if isinstance(feats, torch.Tensor):
            feats = [feats]
        feats = list(feats)
        if len(feats) != len(self.out_layers):
            raise RuntimeError(f"DINO 输出层数不匹配: expected={len(self.out_layers)}, got={len(feats)}")

        tokens = {}
        for lid, feat in zip(self.out_layers, feats):
            tokens[int(lid)] = self._feat_to_tokens(feat, h, w)

        return {
            "tokens": tokens,
            "spatial_shape": (h // self.patch_size, w // self.patch_size),
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
        }


# =========================================================
# 2. Compact CNN Road Detail Encoder
# =========================================================
class BasicRoadBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, gn_groups: int = 8, dilation: int = 1):
        super().__init__()
        self.conv1 = ConvGNAct(in_ch, out_ch, 3, stride=stride, dilation=dilation, gn_groups=gn_groups)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation, bias=False),
            _make_gn(out_ch, gn_groups),
        )
        self.eca = ECALayer(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                _make_gn(out_ch, gn_groups),
            )
        else:
            self.short = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.eca(y)
        return self.act(y + self.short(x))


class CompactRoadEncoder(nn.Module):
    """只到 1/16；C16 从 DLAEM8 增强后的 C8 继续生成。"""
    def __init__(self, in_channels=3, widths=(32, 64, 128, 192), gn_groups=8):
        super().__init__()
        c2, c4, c8, c16 = [int(x) for x in widths]
        self.widths = (c2, c4, c8, c16)

        self.stem = nn.Sequential(
            ConvGNAct(in_channels, c2, 3, stride=2, gn_groups=gn_groups),
            DepthwiseSeparableConv(c2, c2, 3, gn_groups=gn_groups),
        )
        self.stage1 = nn.Sequential(
            BasicRoadBlock(c2, c4, stride=2, gn_groups=gn_groups),
            BasicRoadBlock(c4, c4, stride=1, gn_groups=gn_groups),
        )
        self.stage2 = nn.Sequential(
            BasicRoadBlock(c4, c8, stride=2, gn_groups=gn_groups),
            BasicRoadBlock(c8, c8, stride=1, gn_groups=gn_groups),
        )
        self.stage3 = nn.Sequential(
            BasicRoadBlock(c8, c16, stride=2, gn_groups=gn_groups),
            BasicRoadBlock(c16, c16, stride=1, gn_groups=gn_groups),
            BasicRoadBlock(c16, c16, stride=1, gn_groups=gn_groups, dilation=2),
        )

    def forward_until_c8(self, x):
        c2 = self.stem(x)       # 1/2
        c4 = self.stage1(c2)    # 1/4
        c8 = self.stage2(c4)    # 1/8
        return c2, c4, c8

    def forward_stage3(self, c8_enh):
        return self.stage3(c8_enh)  # 1/16


# =========================================================
# 3. Road prior heads
# =========================================================
class RoadPriorHead(nn.Module):
    """Bottleneck road prior head.

    RD_v4 keeps the prior branch, but avoids an expensive full 3x3 conv at C16=512.
    The head is: 1x1 reduction -> depthwise 3x3 -> 1x1 logits.
    """
    def __init__(self, in_ch: int, mid_ch: Optional[int] = None, mid_ratio: float = 0.25, gn_groups: int = 8):
        super().__init__()
        if mid_ch is None:
            mid_ch = max(32, min(128, int(in_ch * mid_ratio)))
        mid_ch = int(mid_ch)
        self.head = nn.Sequential(
            ConvGNAct(in_ch, mid_ch, 1, gn_groups=gn_groups),
            DepthwiseSeparableConv(mid_ch, mid_ch, 3, gn_groups=gn_groups),
            nn.Conv2d(mid_ch, 1, 1),
        )

    def forward(self, x):
        logits = self.head(x)
        return logits, torch.sigmoid(logits)


# =========================================================
# 4. DLAEM: road-aware deformable LAEM
# =========================================================
class DINOValueProjector(nn.Module):
    def __init__(self, dino_dim: int, out_ch: int, num_levels: int = 2):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(dino_dim, out_ch) for _ in range(num_levels)])
        for m in self.proj:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, token_list: Sequence[torch.Tensor], dino_hw: Tuple[int, int]):
        h, w = int(dino_hw[0]), int(dino_hw[1])
        outs = []
        for token, proj in zip(token_list, self.proj):
            b, n, _ = token.shape
            if n != h * w:
                raise RuntimeError(f"DINO token 数与 dino_hw 不匹配: token={n}, hw={h}x{w}")
            x = proj(token).transpose(1, 2).contiguous().view(b, -1, h, w)
            outs.append(x)
        return outs


class DeformableTokenSampler(nn.Module):
    """纯 PyTorch grid_sample 版 multi-level deformable attention。"""
    def __init__(self, channels: int, num_heads: int = 4, num_levels: int = 2, num_points: int = 4, offset_scale: float = 4.0):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} 必须能被 num_heads={num_heads} 整除")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)
        self.head_dim = self.channels // self.num_heads
        self.offset_scale = float(offset_scale)

        self.offset_conv = nn.Conv2d(channels, num_heads * num_levels * num_points * 2, 3, padding=1)
        self.attn_conv = nn.Conv2d(channels, num_heads * num_levels * num_points, 3, padding=1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            _make_gn(channels, 8),
            nn.GELU(),
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.attn_conv.weight)
        nn.init.zeros_(self.attn_conv.bias)

    @staticmethod
    def _base_grid(h: int, w: int, device, dtype):
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([xx, yy], dim=-1)  # H,W,2; x,y

    def forward(self, q: torch.Tensor, value_maps: Sequence[torch.Tensor]):
        b, c, hq, wq = q.shape
        if len(value_maps) != self.num_levels:
            raise RuntimeError(f"value levels mismatch: expected={self.num_levels}, got={len(value_maps)}")

        offsets = self.offset_conv(q).view(
            b, self.num_heads, self.num_levels, self.num_points, 2, hq, wq
        ).permute(0, 1, 2, 3, 5, 6, 4).contiguous()  # B,H,L,K,Hq,Wq,2
        attn = self.attn_conv(q).view(b, self.num_heads, self.num_levels * self.num_points, hq, wq)
        attn = F.softmax(attn, dim=2).view(b, self.num_heads, self.num_levels, self.num_points, hq, wq)

        base = self._base_grid(hq, wq, q.device, q.dtype).view(1, 1, 1, hq, wq, 2)
        out = q.new_zeros(b, self.num_heads, self.head_dim, hq, wq)

        for li, vmap in enumerate(value_maps):
            bv, cv, hv, wv = vmap.shape
            if bv != b or cv != c:
                raise RuntimeError(f"value map shape mismatch: got={vmap.shape}, expected B={b}, C={c}")

            v = vmap.view(b, self.num_heads, self.head_dim, hv, wv).reshape(b * self.num_heads, self.head_dim, hv, wv)

            off = torch.tanh(offsets[:, :, li])  # B,H,K,Hq,Wq,2
            scale = q.new_tensor([
                2.0 * self.offset_scale / max(float(wv - 1), 1.0),
                2.0 * self.offset_scale / max(float(hv - 1), 1.0),
            ]).view(1, 1, 1, 1, 1, 2)
            grid = base + off * scale  # B,H,K,Hq,Wq,2
            grid = grid.permute(0, 1, 3, 4, 2, 5).contiguous().view(b * self.num_heads, hq, wq * self.num_points, 2)
            sampled = F.grid_sample(v, grid, mode="bilinear", padding_mode="border", align_corners=True)
            sampled = sampled.view(b, self.num_heads, self.head_dim, hq, wq, self.num_points)
            sampled = sampled.permute(0, 1, 2, 5, 3, 4).contiguous()  # B,H,D,K,Hq,Wq
            weight = attn[:, :, li].unsqueeze(2)  # B,H,1,K,Hq,Wq
            out = out + (sampled * weight).sum(dim=3)

        out = out.reshape(b, c, hq, wq)
        return self.out_proj(out)


class RoadAwareDLAEM(nn.Module):
    """C_i + P_i 作为 Query，A_l 作为 Value，输出 C_i'。

    RD_v4 decouples external CNN feature width and internal DLAEM width.
    Example: C16 keeps standard 512 channels, while DLAEM16 uses inner_dim=256.
    This preserves the RD_v1 dataflow but reduces trainable parameters cleanly.
    """
    def __init__(
        self,
        c_dim: int,
        dino_dim: int = 384,
        inner_dim: Optional[int] = None,
        num_heads: int = 4,
        num_levels: int = 2,
        num_points: int = 4,
        offset_scale: float = 4.0,
        gn_groups: int = 8,
    ):
        super().__init__()
        inner_dim = int(inner_dim or c_dim)
        if inner_dim % int(num_heads) != 0:
            raise ValueError(f"inner_dim={inner_dim} 必须能被 num_heads={num_heads} 整除")
        self.c_dim = int(c_dim)
        self.inner_dim = int(inner_dim)
        self.lambda_prior = BoundedScalar(max_value=0.50, init_value=0.02)

        self.q_proj = nn.Sequential(
            ConvGNAct(c_dim, inner_dim, 1, gn_groups=gn_groups),
            DepthwiseSeparableConv(inner_dim, inner_dim, 3, gn_groups=gn_groups),
        )
        self.value_projector = DINOValueProjector(dino_dim, inner_dim, num_levels=num_levels)
        self.sampler = DeformableTokenSampler(inner_dim, num_heads, num_levels, num_points, offset_scale=offset_scale)
        self.refine_inner = nn.Sequential(
            DepthwiseSeparableConv(inner_dim, inner_dim, 3, gn_groups=gn_groups),
            ECALayer(inner_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(inner_dim, c_dim, 1, bias=False),
            _make_gn(c_dim, gn_groups),
            nn.GELU(),
        )

        # Lightweight gate: avoid 3x3 conv on (2*C+1) channels at C16=512.
        self.bg_gate = nn.Sequential(
            ConvGNAct(c_dim * 2 + 1, inner_dim, 1, gn_groups=gn_groups),
            DepthwiseSeparableConv(inner_dim, inner_dim, 3, gn_groups=gn_groups),
            nn.Conv2d(inner_dim, c_dim, 1),
            nn.Sigmoid(),
        )
        self.alpha = BoundedScalar(max_value=0.60, init_value=0.03)

    def forward(self, c: torch.Tensor, prior_prob: torch.Tensor, dino_tokens: Sequence[torch.Tensor], dino_hw: Tuple[int, int]):
        if prior_prob.shape[-2:] != c.shape[-2:]:
            prior_prob = F.interpolate(prior_prob, size=c.shape[-2:], mode="bilinear", align_corners=False)
        c_tilde = c * (1.0 + self.lambda_prior() * prior_prob)
        q = self.q_proj(c_tilde)
        value_maps = self.value_projector(dino_tokens, dino_hw)
        z_inner = self.sampler(q, value_maps)
        z_inner = self.refine_inner(z_inner)
        z = self.out_proj(z_inner)
        gate = self.bg_gate(torch.cat([c, z, prior_prob], dim=1))
        c_enh = c + self.alpha() * gate * z
        aux = {"z": z, "gate_mean": gate.mean(), "alpha": self.alpha(), "lambda_prior": self.lambda_prior()}
        return c_enh, aux


# RD_v4 removes RD_v3's LightLAEMGate4 from the main path.
# The 1/4 skip is enhanced by RoadAwareDLAEM as self.dlaem4.




# =========================================================
# 5. HL_base-style residual decoder block (standalone)
# =========================================================
class ResidualDecoderBlock(nn.Module):
    """
    HL_base 同款残差解码块：
    upsample decoder feature -> concat skip -> 2×ConvBNAct -> ECA -> shortcut -> ReLU。
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNAct(in_channels + skip_channels, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        )
        self.eca = ECALayer(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([x, skip], dim=1)
        out = self.fuse(f)
        out = self.eca(out)
        out = out + self.shortcut(f)
        return self.act(out)

class ResNet34BaselineEncoder(nn.Module):
    """HL_base 同款 ResNet34 encoder，但把 forward 拆开，便于 DLAEM8 串联插入。

    尺度：
        x0: H/2,  64
        x1: H/4,  64
        x2: H/8,  128
        x3: H/16, 256
        x4: H/32, 512
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        encoder = self._get_resnet34(pretrained=pretrained)

        if int(in_channels) != 3:
            self.input_adapter = nn.Conv2d(int(in_channels), 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

    @staticmethod
    def _get_resnet34(pretrained: bool = True):
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet34(weights=weights)
        except Exception:
            from torchvision import models
            try:
                return models.resnet34(pretrained=pretrained)
            except TypeError:
                return models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)

    def forward_until_x2(self, x: torch.Tensor):
        x = self.input_adapter(x)
        x0 = self.stem(x)                    # H/2,  64
        x1 = self.layer1(self.maxpool(x0))   # H/4,  64
        x2 = self.layer2(x1)                 # H/8,  128
        return x0, x1, x2

    def forward_layer3(self, x2: torch.Tensor):
        return self.layer3(x2)               # H/16, 256

    def forward_layer4(self, x3: torch.Tensor):
        return self.layer4(x3)               # H/32, 512



# =========================================================
# 6. DINO plain-add projector
# =========================================================
class DINOPlainAddProjector(nn.Module):
    """Project single DINO layers to CNN feature shapes for plain addition.

    Mapping:
        A2  -> s1, aligned to x1 (1/4, 64ch)
        A5  -> s2, aligned to x2 (1/8, 128ch)
        A8  -> s3, aligned to x3 (1/16, 256ch)
        A11 -> s4, aligned to x4 (1/32, 512ch)

    This module only performs necessary channel projection and spatial resizing.
    The fusion itself is strictly: x_i + s_i.
    """
    def __init__(self, dino_dim: int = 384, gn_groups: int = 16):
        super().__init__()
        self.proj1 = ConvGNAct(int(dino_dim), 64, kernel_size=1, gn_groups=int(gn_groups))
        self.proj2 = ConvGNAct(int(dino_dim), 128, kernel_size=1, gn_groups=int(gn_groups))
        self.proj3 = ConvGNAct(int(dino_dim), 256, kernel_size=1, gn_groups=int(gn_groups))
        self.proj4 = ConvGNAct(int(dino_dim), 512, kernel_size=1, gn_groups=int(gn_groups))

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor, dino_hw: Tuple[int, int]) -> torch.Tensor:
        h, w = int(dino_hw[0]), int(dino_hw[1])
        b, n, c = tokens.shape
        if n != h * w:
            raise RuntimeError(f"DINO token 数与 spatial_shape 不匹配: token={n}, hw={h}x{w}")
        return tokens.transpose(1, 2).contiguous().view(b, c, h, w)

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(
        self,
        a2: torch.Tensor,
        a5: torch.Tensor,
        a8: torch.Tensor,
        a11: torch.Tensor,
        dino_hw: Tuple[int, int],
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ):
        a2m = self._tokens_to_map(a2, dino_hw)
        a5m = self._tokens_to_map(a5, dino_hw)
        a8m = self._tokens_to_map(a8, dino_hw)
        a11m = self._tokens_to_map(a11, dino_hw)

        s1 = self._resize_like(self.proj1(a2m), x1)
        s2 = self._resize_like(self.proj2(a5m), x2)
        s3 = self._resize_like(self.proj3(a8m), x3)
        s4 = self._resize_like(self.proj4(a11m), x4)
        return s1, s2, s3, s4


class RD_v1_C(nn.Module):
    """RD_v1_C: DINO + CNN without DLAEM, using plain addition.

    Data flow:
        X -> PEFT-DINO token bank A2/A5/A8/A11
        X -> ResNet34 CNN encoder -> x0/x1/x2/x3/x4
        A2/A5/A8/A11 -> s1/s2/s3/s4
        x1_hat = x1 + s1; x2_hat = x2 + s2; x3_hat = x3 + s3; x4_hat = x4 + s4
        dec3(x4_hat, x3_hat) -> dec2(..., x2_hat) -> dec1(..., x1_hat) -> dec0(..., x0) -> logits
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        num_classes: Optional[int] = None,
        in_channels: Optional[int] = None,
        pretrained: bool = True,
        return_aux: bool = True,
        # DINO
        dino_model_name: str = "dinov3_vits16",
        dino_repo_path: str = "/home/u2508183004/zyn/SEG/dinounet/dinov3",
        dino_ckpt_path: str = "/home/u2508183004/zyn/SEG/weight/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        dino_layers: Sequence[int] = (2, 5, 8, 11),
        dino_embed_dim: int = 384,
        dino_patch_size: int = 16,
        dino_normalize: bool = False,
        dino_intermediate_norm: bool = True,
        adapter_bottleneck: int = 64,
        adapter_dropout: float = 0.0,
        # CNN / decoder
        resnet_pretrained: Optional[bool] = None,
        gn_groups: int = 16,
        aux_mid_ch: int = 64,
        # compatibility with RD_v4 configs; ignored here
        light_gate_dim: Optional[int] = None,
        prior_mid_max: int = 128,
        dlaem4_heads: int = 4,
        dlaem4_points: int = 4,
        dlaem4_offset_scale: float = 3.0,
        dlaem4_dim: int = 64,
        dlaem8_heads: int = 4,
        dlaem16_heads: int = 8,
        dlaem_points: int = 6,
        dlaem8_offset_scale: float = 4.0,
        dlaem16_offset_scale: float = 5.0,
        dlaem8_dim: int = 256,
        dlaem16_dim: int = 256,
        # optimizer group config
        param_group_lrs: Optional[Dict[str, float]] = None,
        param_group_weight_decays: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__()
        if in_channels is not None:
            n_channels = int(in_channels)
        if num_classes is not None:
            n_classes = int(num_classes)
        if resnet_pretrained is None:
            resnet_pretrained = bool(pretrained)

        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.return_aux = bool(return_aux)
        self.dino_layers = tuple(int(x) for x in dino_layers)
        if len(self.dino_layers) != 4:
            raise ValueError(f"RD_v1_C 需要 4 个 DINO 层，例如 [2,5,8,11]，当前: {self.dino_layers}")

        self.param_group_lrs = param_group_lrs or {}
        self.param_group_weight_decays = param_group_weight_decays or {}

        # 1) PEFT-DINOv3 token bank: A2/A5/A8/A11.
        self.dino = PEFTDINOv3GlobalBranch(
            dino_model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            out_layers=self.dino_layers,
            embed_dim=int(dino_embed_dim),
            patch_size=int(dino_patch_size),
            adapter_bottleneck=int(adapter_bottleneck),
            adapter_dropout=float(adapter_dropout),
            dino_normalize=bool(dino_normalize),
            dino_intermediate_norm=bool(dino_intermediate_norm),
        )

        # 2) HL_base/RD_v4 same ResNet34 encoder.
        self.cnn = ResNet34BaselineEncoder(in_channels=self.n_channels, pretrained=bool(resnet_pretrained))

        # 3) DINO feature alignment for plain addition.
        self.dino_projector = DINOPlainAddProjector(dino_dim=int(dino_embed_dim), gn_groups=int(gn_groups))

        # Fixed ResNet34 baseline channels.
        c2, c4, c8, c16, c32 = 64, 64, 128, 256, 512

        # 4) RD_v4 same decoder.
        self.dec3 = ResidualDecoderBlock(in_channels=c32, skip_channels=c16, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=c8, out_channels=128)
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=c4, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=c2, out_channels=64)

        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, self.n_classes, kernel_size=1),
        )

        # 5) Aux heads: keep RD_v4/RD_v1-series style.
        aux_mid_ch = int(aux_mid_ch)
        self.aux16_head = nn.Sequential(
            ConvBNAct(256, max(aux_mid_ch, 32), kernel_size=3),
            nn.Conv2d(max(aux_mid_ch, 32), self.n_classes, kernel_size=1),
        )
        self.aux8_head = nn.Sequential(
            ConvBNAct(128, max(aux_mid_ch, 32), kernel_size=3),
            nn.Conv2d(max(aux_mid_ch, 32), self.n_classes, kernel_size=1),
        )
        self.aux4_head = nn.Sequential(
            ConvBNAct(96, max(aux_mid_ch, 32), kernel_size=3),
            nn.Conv2d(max(aux_mid_ch, 32), self.n_classes, kernel_size=1),
        )

        self._print_model_info()

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> Dict[str, Any]:
        input_size = x.shape[-2:]

        # 1) PEFT-DINO token bank.
        dino_out = self.dino(x)
        token_dict = dino_out["tokens"]
        dino_hw = dino_out["spatial_shape"]
        a2 = token_dict[self.dino_layers[0]]
        a5 = token_dict[self.dino_layers[1]]
        a8 = token_dict[self.dino_layers[2]]
        a11 = token_dict[self.dino_layers[3]]

        # 2) CNN encoder runs normally; DINO does not alter the CNN main path.
        x0, x1, x2 = self.cnn.forward_until_x2(x)  # H/2, H/4, H/8
        x3 = self.cnn.forward_layer3(x2)           # H/16
        x4 = self.cnn.forward_layer4(x3)           # H/32

        # 3) DINO alignment maps and direct addition.
        s1, s2, s3, s4 = self.dino_projector(a2, a5, a8, a11, dino_hw, x1, x2, x3, x4)
        x1_add = x1 + s1
        x2_add = x2 + s2
        x3_add = x3 + s3
        x4_add = x4 + s4

        # 4) Decoder uses plain-added features.
        d3 = self.dec3(x4_add, x3_add)        # H/16, 256
        d2 = self.dec2(d3, x2_add)            # H/8,  128
        d1 = self.dec1(d2, x1_add)            # H/4,   96
        d0 = self.dec0(d1, x0)                # H/2,   64

        logits_half = self.out_head(d0)       # H/2, n_classes
        logits = F.interpolate(logits_half, size=input_size, mode="bilinear", align_corners=False)

        logit16 = self.aux16_head(x3_add)     # H/16, supervise DINO+CNN plain-added skip
        logit8 = self.aux8_head(d2)           # H/8
        logit4 = self.aux4_head(d1)           # H/4
        logit2 = logits_half                  # H/2

        outputs: Dict[str, Any] = {
            "logits": logits,
            "final_logits": logits,
            "logits_half": logits_half,
            "logit16": logit16,
            "logit8": logit8,
            "logit4": logit4,
            "logit2": logit2,
            # diagnostics for this ablation
            "plain_add_s1_mean_abs": s1.detach().abs().mean(),
            "plain_add_s2_mean_abs": s2.detach().abs().mean(),
            "plain_add_s3_mean_abs": s3.detach().abs().mean(),
            "plain_add_s4_mean_abs": s4.detach().abs().mean(),
            "plain_add_x1_mean_abs": x1.detach().abs().mean(),
            "plain_add_x2_mean_abs": x2.detach().abs().mean(),
            "plain_add_x3_mean_abs": x3.detach().abs().mean(),
            "plain_add_x4_mean_abs": x4.detach().abs().mean(),
        }
        return outputs

    def forward(self, x: torch.Tensor):
        outputs = self.forward_features(x)
        if self.return_aux:
            return outputs
        return outputs["logits"]

    def forward_train(self, x: torch.Tensor):
        return self.forward_features(x)

    # -----------------------------------------------------
    # Optimizer groups for train_RD.py
    # -----------------------------------------------------
    def get_param_groups(self, base_lr: float = 1e-4, weight_decay: float = 1e-2):
        lr_cfg = self.param_group_lrs or {}
        wd_cfg = self.param_group_weight_decays or {}
        used = set()
        groups: List[Dict[str, Any]] = []

        def clean_params(params: Iterable[nn.Parameter]) -> List[nn.Parameter]:
            out = []
            for p in params:
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in used:
                    continue
                used.add(pid)
                out.append(p)
            return out

        def add_group(name: str, params: Iterable[nn.Parameter], default_lr_mult: float, default_wd: Optional[float] = None):
            ps = clean_params(params)
            if not ps:
                return
            lr = float(lr_cfg.get(name, base_lr * float(default_lr_mult)))
            wd = float(wd_cfg.get(name, weight_decay if default_wd is None else default_wd))
            groups.append({"name": name, "params": ps, "lr": lr, "weight_decay": wd})

        add_group("dino_adapters", self.dino.parameters(), default_lr_mult=0.60, default_wd=0.02)
        add_group("cnn_encoder", self.cnn.parameters(), default_lr_mult=0.55, default_wd=weight_decay)
        add_group("dino_projector", self.dino_projector.parameters(), default_lr_mult=1.20, default_wd=weight_decay)
        decoder_params = (
            list(self.dec3.parameters()) + list(self.dec2.parameters()) + list(self.dec1.parameters()) + list(self.dec0.parameters())
            + list(self.out_head.parameters()) + list(self.aux16_head.parameters()) + list(self.aux8_head.parameters()) + list(self.aux4_head.parameters())
        )
        add_group("decoder", decoder_params, default_lr_mult=1.20)

        remaining = clean_params(self.parameters())
        if remaining:
            groups.append({"name": "others", "params": remaining, "lr": float(base_lr), "weight_decay": float(weight_decay)})
        return groups

    def _print_model_info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        dino_total = sum(p.numel() for p in self.dino.parameters())
        dino_train = sum(p.numel() for p in self.dino.parameters() if p.requires_grad)
        print("-" * 60)
        print("[RD_v1_C] PEFT-DINO + ResNet34 CNN + plain add fusion + RD decoder")
        print(f"    - Total params:     {total / 1e6:.2f} M")
        print(f"    - Trainable params: {trainable / 1e6:.2f} M")
        print(f"    - Frozen params:    {frozen / 1e6:.2f} M")
        print(f"    - DINO total/train: {dino_total / 1e6:.2f} M / {dino_train / 1e6:.2f} M")
        print("    - Removed: RoadPrior heads and DLAEM4/8/16")
        print("    - Fusion: s1=A2->x1, s2=A5->x2, s3=A8->x3, s4=A11->x4; x_i_hat = x_i + s_i")
        print("    - Decoder: dec3(x4+s4,x3+s3) -> dec2(d3,x2+s2) -> dec1(d2,x1+s1) -> dec0(d1,x0)")
        print("-" * 60)


if __name__ == "__main__":
    print("RD_v1_C module loaded. Instantiate through models.get_model(config['model']).")
