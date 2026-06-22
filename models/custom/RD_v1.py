# -*- coding: utf-8 -*-
"""
RD_v1: PEFT-DINO-only pseudo pyramid + RD decoder.

本文件是 RD_v4 的 DINO-only 消融版：
1) 保留 PEFT-DINOv3：DINO backbone frozen，只训练 adapter 和 gamma；
2) 删除 CNN encoder、Road Prior、DLAEM4/8/16；
3) 使用 A2/A5/A8/A11 构建 DINO pseudo pyramid：d4/d3/d2/d1/d0；
4) 保留 RD_v4 的 HL_base-style residual decoder、out head 和 aux heads。
"""

import os
import sys
import math
import importlib
from typing import Dict, Tuple, Sequence, List, Optional, Iterable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["RD_v1"]


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
# 2. RD_v4 decoder block retained
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




# =========================================================
# 3. DINO-only pseudo pyramid
# =========================================================
class DINOMapFuse(nn.Module):
    """Project selected DINO token maps and fuse them into one spatial feature map.

    输入是若干个 DINO 2D token map，尺度都是 DINO patch grid，例如 512 输入下为 32x32。
    本模块只做尺度适配和通道适配，不引入 RGB/CNN stem，保证 RD_v1 是干净的 DINO-only 消融。
    """
    def __init__(self, in_dim: int, out_ch: int, num_inputs: int, gn_groups: int = 16):
        super().__init__()
        self.out_ch = int(out_ch)
        self.projs = nn.ModuleList([
            ConvGNAct(int(in_dim), int(out_ch), kernel_size=1, gn_groups=int(gn_groups))
            for _ in range(int(num_inputs))
        ])
        self.fuse = nn.Sequential(
            ConvGNAct(int(out_ch) * int(num_inputs), int(out_ch), kernel_size=3, gn_groups=int(gn_groups)),
            DepthwiseSeparableConv(int(out_ch), int(out_ch), kernel_size=3, gn_groups=int(gn_groups)),
            ECALayer(int(out_ch)),
        )

    def forward(self, maps: Sequence[torch.Tensor], out_size: Tuple[int, int]) -> torch.Tensor:
        if len(maps) != len(self.projs):
            raise RuntimeError(f"DINOMapFuse 输入数量不匹配: expected={len(self.projs)}, got={len(maps)}")
        outs = []
        for x, proj in zip(maps, self.projs):
            y = proj(x)
            if tuple(y.shape[-2:]) != tuple(out_size):
                y = F.interpolate(y, size=out_size, mode="bilinear", align_corners=False)
            outs.append(y)
        return self.fuse(torch.cat(outs, dim=1))


class DINOPseudoPyramid(nn.Module):
    """Build d0/d1/d2/d3/d4 from A2/A5/A8/A11.

    对齐 RD_v4 的 ResNet34 feature shapes：
        d0: H/2,  64   <- replaces x0
        d1: H/4,  64   <- replaces x1
        d2: H/8,  128  <- replaces x2
        d3: H/16, 256  <- replaces x3
        d4: H/32, 512  <- replaces x4

    方案 B：不同 DINO 层负责不同尺度：
        A2/A5  -> d1，d1 上采样 -> d0
        A5/A8  -> d2
        A8/A11 -> d3
        A11    -> d4
    """
    def __init__(self, dino_dim: int = 384, gn_groups: int = 16):
        super().__init__()
        self.d1_fuse = DINOMapFuse(dino_dim, 64, 2, gn_groups=gn_groups)    # A2 + A5 -> 1/4
        self.d2_fuse = DINOMapFuse(dino_dim, 128, 2, gn_groups=gn_groups)   # A5 + A8 -> 1/8
        self.d3_fuse = DINOMapFuse(dino_dim, 256, 2, gn_groups=gn_groups)   # A8 + A11 -> 1/16
        self.d4_fuse = DINOMapFuse(dino_dim, 512, 1, gn_groups=gn_groups)   # A11 -> 1/32
        self.d0_refine = nn.Sequential(
            ConvGNAct(64, 64, kernel_size=3, gn_groups=gn_groups),
            DepthwiseSeparableConv(64, 64, kernel_size=3, gn_groups=gn_groups),
            ECALayer(64),
        )

    @staticmethod
    def tokens_to_map(tokens: torch.Tensor, dino_hw: Tuple[int, int]) -> torch.Tensor:
        h, w = int(dino_hw[0]), int(dino_hw[1])
        b, n, c = tokens.shape
        if n != h * w:
            raise RuntimeError(f"token 数与 dino_hw 不匹配: token={n}, hw={h}x{w}")
        return tokens.transpose(1, 2).contiguous().view(b, c, h, w)

    @staticmethod
    def _safe_size(h: int, w: int, div: int) -> Tuple[int, int]:
        return max(1, h // div), max(1, w // div)

    def forward(
        self,
        token_dict: Dict[int, torch.Tensor],
        dino_layers: Sequence[int],
        dino_hw: Tuple[int, int],
        input_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        h, w = int(input_size[0]), int(input_size[1])
        l2, l5, l8, l11 = [int(x) for x in dino_layers]
        a2 = self.tokens_to_map(token_dict[l2], dino_hw)
        a5 = self.tokens_to_map(token_dict[l5], dino_hw)
        a8 = self.tokens_to_map(token_dict[l8], dino_hw)
        a11 = self.tokens_to_map(token_dict[l11], dino_hw)

        size_d0 = self._safe_size(h, w, 2)
        size_d1 = self._safe_size(h, w, 4)
        size_d2 = self._safe_size(h, w, 8)
        size_d3 = self._safe_size(h, w, 16)
        size_d4 = self._safe_size(h, w, 32)

        d1 = self.d1_fuse([a2, a5], size_d1)       # H/4, 64
        d0 = F.interpolate(d1, size=size_d0, mode="bilinear", align_corners=False)
        d0 = self.d0_refine(d0)                    # H/2, 64
        d2 = self.d2_fuse([a5, a8], size_d2)       # H/8, 128
        d3 = self.d3_fuse([a8, a11], size_d3)      # H/16, 256
        d4 = self.d4_fuse([a11], size_d4)          # H/32, 512

        return {"d0": d0, "d1": d1, "d2": d2, "d3": d3, "d4": d4}


# =========================================================
# 4. RD_v1 main model
# =========================================================
class RD_v1(nn.Module):
    """RD_v1: PEFT-DINO-only pseudo pyramid + RD_v4 decoder.

    删除：CNN encoder、Road Prior、DLAEM4/8/16。
    保留：PEFT-DINOv3、A2/A5/A8/A11、DINO pseudo pyramid、RD decoder、out head、aux heads。
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
        gn_groups: int = 16,
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
        if int(n_channels) != 3:
            raise ValueError("RD_v1 是 DINO-only 消融版，当前只支持 3 通道输入。")

        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.return_aux = bool(return_aux)
        self.dino_layers = tuple(int(x) for x in dino_layers)
        if len(self.dino_layers) != 4:
            raise ValueError(f"RD_v1 需要 4 个 DINO 层，例如 [2,5,8,11]，当前: {self.dino_layers}")

        self.param_group_lrs = param_group_lrs or {}
        self.param_group_weight_decays = param_group_weight_decays or {}

        # 1) PEFT-DINOv3 token bank: A2/A5/A8/A11。与 RD_v4 保持一致。
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

        # 2) DINO-only pseudo pyramid: replaces CNN x0/x1/x2/x3/x4.
        self.pseudo = DINOPseudoPyramid(dino_dim=int(dino_embed_dim), gn_groups=int(gn_groups))

        # 3) RD_v4 decoder retained. Feature shapes match ResNet34 baseline encoder.
        self.dec3 = ResidualDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=64, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=64, out_channels=64)

        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, self.n_classes, kernel_size=1),
        )

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

    def forward_features(self, x: torch.Tensor) -> Dict[str, Any]:
        input_size = x.shape[-2:]

        # 1) PEFT-DINO token bank.
        dino_out = self.dino(x)
        token_dict = dino_out["tokens"]
        dino_hw = dino_out["spatial_shape"]

        # 2) Build DINO pseudo pyramid.
        feats = self.pseudo(token_dict, self.dino_layers, dino_hw, input_size)
        d0, d1, d2, d3, d4 = feats["d0"], feats["d1"], feats["d2"], feats["d3"], feats["d4"]

        # 3) RD decoder. No CNN, no DLAEM, no road prior.
        y3 = self.dec3(d4, d3)       # H/16, 256
        y2 = self.dec2(y3, d2)       # H/8,  128
        y1 = self.dec1(y2, d1)       # H/4,   96
        y0 = self.dec0(y1, d0)       # H/2,   64

        logits_half = self.out_head(y0)
        logits = F.interpolate(logits_half, size=input_size, mode="bilinear", align_corners=False)

        logit16 = self.aux16_head(d3)        # H/16
        logit8 = self.aux8_head(y2)          # H/8
        logit4 = self.aux4_head(y1)          # H/4
        logit2 = logits_half                 # H/2

        # Diagnostics are intentionally lightweight and generic.
        prob = torch.sigmoid(logits)
        outputs: Dict[str, Any] = {
            "logits": logits,
            "final_logits": logits,
            "logits_half": logits_half,
            "logit16": logit16,
            "logit8": logit8,
            "logit4": logit4,
            "logit2": logit2,
            "dino_d0_mean": d0.mean(),
            "dino_d1_mean": d1.mean(),
            "dino_d2_mean": d2.mean(),
            "dino_d3_mean": d3.mean(),
            "dino_d4_mean": d4.mean(),
            "prob_mean": prob.mean(),
            "prob_gt_05": (prob > 0.5).float().mean(),
        }
        return outputs

    def forward(self, x: torch.Tensor):
        outputs = self.forward_features(x)
        if self.return_aux:
            return outputs
        return outputs["logits"]

    def forward_train(self, x: torch.Tensor):
        return self.forward_features(x)

    def get_param_groups(self, base_lr: float = 1e-4, weight_decay: float = 1e-2):
        """Param groups compatible with train_RD.py.

        RD_v1 没有 cnn_encoder / road_priors / dlaem 参数组。
        """
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
        add_group("pseudo_encoder", self.pseudo.parameters(), default_lr_mult=1.20, default_wd=weight_decay)
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
        pseudo_total = sum(p.numel() for p in self.pseudo.parameters())
        decoder_total = sum(
            p.numel() for m in [self.dec3, self.dec2, self.dec1, self.dec0, self.out_head, self.aux16_head, self.aux8_head, self.aux4_head]
            for p in m.parameters()
        )
        print("-" * 60)
        print("[RD_v1] PEFT-DINO-only Pseudo Pyramid + RD Decoder")
        print(f"    - Total params:     {total / 1e6:.2f} M")
        print(f"    - Trainable params: {trainable / 1e6:.2f} M")
        print(f"    - Frozen params:    {frozen / 1e6:.2f} M")
        print(f"    - DINO total/train: {dino_total / 1e6:.2f} M / {dino_train / 1e6:.2f} M")
        print(f"    - Pseudo encoder:   {pseudo_total / 1e6:.2f} M")
        print(f"    - Decoder+heads:    {decoder_total / 1e6:.2f} M")
        print("    - Removed: CNN encoder, RoadPrior heads, DLAEM4/8/16")
        print("    - Data flow: X -> PEFT-DINO A2/A5/A8/A11 -> d4/d3/d2/d1/d0 -> RD decoder")
        print("-" * 60)


if __name__ == "__main__":
    print("RD_v1 module loaded. Instantiate through models.get_model(config['model']).")
