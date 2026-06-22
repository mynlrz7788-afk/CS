# -*- coding: utf-8 -*-
"""
RD_v4: Standalone Baseline-stable CNN + PEFT-DINO Road-aware DLAEM.

本文件已经内置 RD_v4 需要的全部 DINO / DLAEM4/DLAEM8/DLAEM16 模块，
以及 HL_base 中的 residual decoder 结构；部署时不再依赖 RD_v3.py、RD_v2.py 或 HL_base.py。

重要修复：
1) DLAEM8/DLAEM16 的 alpha 不再从 0 初始化，而是默认 0.03，避免 DLAEM 梯度被关死；
2) DLAEM4 替代 LightGate4，使用轻量 inner_dim/points；
3) DINO Adapter 的 gamma_attn/gamma_ffn 默认 1e-3，避免 adapter 与 gamma 双零初始化导致无梯度。
"""

import os
import sys
import math
import importlib
from typing import Dict, Tuple, Sequence, List, Optional, Iterable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["RD_v4"]


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


class RD_v4(nn.Module):
    """RD_v4 主模型。

    默认训练时返回 dict，兼容 train_RD.py；测试时可通过 return_aux=False 返回 logits tensor。
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
        # Baseline CNN / DLAEM
        resnet_pretrained: Optional[bool] = None,
        gn_groups: int = 16,
        # DLAEM4 is intentionally lightweight because its query grid is 1/4 resolution.
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
        light_gate_dim: Optional[int] = None,  # kept for old config compatibility; RD_v4 uses DLAEM4.
        prior_mid_max: int = 128,
        aux_mid_ch: int = 64,
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
            raise ValueError(f"RD_v4 需要 4 个 DINO 层，例如 [2,5,8,11]，当前: {self.dino_layers}")

        self.param_group_lrs = param_group_lrs or {}
        self.param_group_weight_decays = param_group_weight_decays or {}

        # 1) PEFT-DINOv3 token bank: A2/A5/A8/A11。
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

        # 2) HL_base 同款 ResNet34 encoder。
        self.cnn = ResNet34BaselineEncoder(in_channels=self.n_channels, pretrained=bool(resnet_pretrained))

        # 固定 ResNet34 baseline 通道。
        c2, c4, c8, c16, c32 = 64, 64, 128, 256, 512

        # 3) Soft road prior heads。
        self.prior4 = RoadPriorHead(c4, mid_ch=max(32, min(int(prior_mid_max), c4 // 2)), gn_groups=gn_groups)
        self.prior8 = RoadPriorHead(c8, mid_ch=max(32, min(int(prior_mid_max), c8 // 4)), gn_groups=gn_groups)
        self.prior16 = RoadPriorHead(c16, mid_ch=max(32, min(int(prior_mid_max), c16 // 4)), gn_groups=gn_groups)

        # 4) DINO-CNN interaction：保持初步思路中的 token bank 查询方式。
        self.dlaem8 = RoadAwareDLAEM(
            c_dim=c8,
            dino_dim=int(dino_embed_dim),
            inner_dim=int(dlaem8_dim),
            num_heads=int(dlaem8_heads),
            num_levels=2,
            num_points=int(dlaem_points),
            offset_scale=float(dlaem8_offset_scale),
            gn_groups=int(gn_groups),
        )
        self.dlaem16 = RoadAwareDLAEM(
            c_dim=c16,
            dino_dim=int(dino_embed_dim),
            inner_dim=int(dlaem16_dim),
            num_heads=int(dlaem16_heads),
            num_levels=2,
            num_points=int(dlaem_points),
            offset_scale=float(dlaem16_offset_scale),
            gn_groups=int(gn_groups),
        )
        self.dlaem4 = RoadAwareDLAEM(
            c_dim=c4,
            dino_dim=int(dino_embed_dim),
            inner_dim=int(dlaem4_dim),
            num_heads=int(dlaem4_heads),
            num_levels=2,
            num_points=int(dlaem4_points),
            offset_scale=float(dlaem4_offset_scale),
            gn_groups=int(gn_groups),
        )

        # 5) HL_base 同款 decoder：只把 skip 换成增强后的 x3'/x2'/x1'。
        self.dec3 = ResidualDecoderBlock(in_channels=c32, skip_channels=c16, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=c8, out_channels=128)
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=c4, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=c2, out_channels=64)

        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, self.n_classes, kernel_size=1),
        )

        # 6) 辅助监督头：train_RD.py 可自动识别 logit16/logit8/logit4/logit2。
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

        # 1) PEFT-DINO token bank。只保留 A2/A5/A8/A11，不生成伪 DINO pyramid。
        dino_out = self.dino(x)
        token_dict = dino_out["tokens"]
        dino_hw = dino_out["spatial_shape"]
        a2 = token_dict[self.dino_layers[0]]
        a5 = token_dict[self.dino_layers[1]]
        a8 = token_dict[self.dino_layers[2]]
        a11 = token_dict[self.dino_layers[3]]

        # 2) Baseline encoder 到 x2。
        x0, x1, x2 = self.cnn.forward_until_x2(x)

        # 3) Soft priors at 1/4 and 1/8。
        p4_logits, p4 = self.prior4(x1)
        p8_logits, p8 = self.prior8(x2)

        # 4) DLAEM4: x1 + P4 查询 A2/A5，得到 x1'，替代 RD_v3 的 LightGate4。
        #    注意：1/4 query 很密集，因此 RD_v4 默认使用更轻的 inner_dim/points。
        x1_enh, aux4 = self.dlaem4(x1, p4, [a2, a5], dino_hw)

        # 5) DLAEM8: x2 + P8 查询 A5/A8；x2' 串联进入 layer3。
        x2_enh, aux8 = self.dlaem8(x2, p8, [a5, a8], dino_hw)

        # 6) layer3 从 x2' 生成 x3。
        x3 = self.cnn.forward_layer3(x2_enh)
        p16_logits, p16 = self.prior16(x3)

        # 7) DLAEM16: x3 + P16 查询 A8/A11，得到 x3'。
        x3_enh, aux16 = self.dlaem16(x3, p16, [a8, a11], dino_hw)

        # 8) layer4 从 x3' 继续生成 x4，保留 baseline bottleneck。
        x4 = self.cnn.forward_layer4(x3_enh)

        # 9) Baseline decoder，skip 使用增强后的 x3'/x2'/x1'。
        d3 = self.dec3(x4, x3_enh)        # H/16, 256
        d2 = self.dec2(d3, x2_enh)        # H/8,  128
        d1 = self.dec1(d2, x1_enh)        # H/4,   96
        d0 = self.dec0(d1, x0)            # H/2,   64

        logits_half = self.out_head(d0)   # H/2, n_classes
        logits = F.interpolate(logits_half, size=input_size, mode="bilinear", align_corners=False)

        logit16 = self.aux16_head(x3_enh)        # H/16
        logit8 = self.aux8_head(d2)              # H/8
        logit4 = self.aux4_head(d1)              # H/4
        logit2 = logits_half                     # H/2

        outputs: Dict[str, Any] = {
            "logits": logits,
            "final_logits": logits,
            "logits_half": logits_half,
            "logit16": logit16,
            "logit8": logit8,
            "logit4": logit4,
            "logit2": logit2,
            "prior4_logits": p4_logits,
            "prior8_logits": p8_logits,
            "prior16_logits": p16_logits,
            "dlaem4_alpha": aux4.get("alpha"),
            "dlaem8_alpha": aux8.get("alpha"),
            "dlaem16_alpha": aux16.get("alpha"),
            "dlaem4_gate_mean": aux4.get("gate_mean"),
            "dlaem8_gate_mean": aux8.get("gate_mean"),
            "dlaem16_gate_mean": aux16.get("gate_mean"),
            "dlaem4_lambda_prior": aux4.get("lambda_prior"),
            "dlaem8_lambda_prior": aux8.get("lambda_prior"),
            "dlaem16_lambda_prior": aux16.get("lambda_prior"),
        }
        return outputs

    def forward(self, x: torch.Tensor):
        outputs = self.forward_features(x)
        if self.return_aux:
            return outputs
        return outputs["logits"]

    def forward_train(self, x: torch.Tensor):
        # train_RD.py 会优先调用 forward_train，这里强制返回完整 dict。
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
        add_group("road_priors", list(self.prior4.parameters()) + list(self.prior8.parameters()) + list(self.prior16.parameters()), default_lr_mult=1.00)
        add_group("dlaem", list(self.dlaem4.parameters()) + list(self.dlaem8.parameters()) + list(self.dlaem16.parameters()), default_lr_mult=1.00)
        decoder_params = (
            list(self.dec3.parameters()) + list(self.dec2.parameters()) + list(self.dec1.parameters()) + list(self.dec0.parameters())
            + list(self.out_head.parameters()) + list(self.aux16_head.parameters()) + list(self.aux8_head.parameters()) + list(self.aux4_head.parameters())
        )
        add_group("decoder", decoder_params, default_lr_mult=1.20)

        # 兜底：任何未分组 trainable 参数。
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
        print("[RD_v4] Standalone Baseline-stable CNN + PEFT-DINO Road-aware DLAEM4/8/16")
        print(f"    - Total params:     {total / 1e6:.2f} M")
        print(f"    - Trainable params: {trainable / 1e6:.2f} M")
        print(f"    - Frozen params:    {frozen / 1e6:.2f} M")
        print(f"    - DINO total/train: {dino_total / 1e6:.2f} M / {dino_train / 1e6:.2f} M")
        print("    - Data flow: X -> PEFT-DINO token bank + HL_base encoder -> DLAEM4 + DLAEM8 -> layer3 -> DLAEM16 -> layer4 -> HL_base decoder")
        print("    - Gradient fix: DLAEM4/8/16 alpha init=0.03, DINO adapter gamma init=1e-3")
        print("    - Decoder: dec3(x4,x3') -> dec2(d3,x2') -> dec1(d2,x1') -> dec0(d1,x0)")
        print("-" * 60)


if __name__ == "__main__":
    # 仅做轻量级构建测试时可以传空 DINO 路径会报错；正式运行使用 config 中的 DINO 路径。
    # 这里不自动实例化，避免本地没有 DINO 权重时报错。
    print("RD_v4 module loaded. Instantiate through models.get_model(config['model']).")
