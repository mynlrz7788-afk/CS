# -*- coding: utf-8 -*-
"""
RD_v5: Reliable Foundation-Guided Curvilinear Structure Segmentation Network.

本文件基于 RD_v4 的稳定数据流升级为 RD_v5 / RFG-CSNet。
内置 PEFT-DINO、RDFQ4/RDFQ8/RDFQ16、结构先验、可靠性估计、DINO prior、拓扑解码器；
部署时不再依赖 RD_v3.py、RD_v2.py 或 HL_base.py。

主要改进：
1) road-aware 命名升级为 structure-aware / curvilinear structure formulation；
2) DLAEM 升级为 RDFQ：DINO prior + CNN-DINO disagreement + reliability map；
3) 增加 ExternalStructureTokenBank，对 A2/A5/A8/A11 做外部结构适配；
4) 增加 centerline head 和可选 RDV5StructureLoss，便于后续接入拓扑监督；
5) 保留 alpha=0.03、DINO adapter gamma=1e-3 等 RD_v4 中已经验证过的稳定化设计。
"""

import os
import sys
import math
import importlib
from typing import Dict, Tuple, Sequence, List, Optional, Iterable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["RD_v5", "RFG_CSNet", "RDV5StructureLoss"]


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
            f"[RD_v5] 加载 DINOv3 权重完成: {ckpt_path}\n"
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
            f"[RD_v5] PEFT-DINOv3 ready. layers={self.out_layers}, patch={self.patch_size}, "
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
# 2. Structure-aware CNN encoder and prior heads
# =========================================================
class StructurePriorHead(nn.Module):
    """Lightweight structure prior head for curvilinear targets.

    It predicts scale-specific structure evidence s_i from CNN local features.
    The output is not the final mask; it is used as local evidence/query prior
    for reliable DINO token querying.
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

    def forward(self, x: torch.Tensor):
        logits = self.head(x)
        return logits, torch.sigmoid(logits)


# Backward-compatible alias. Old configs/loggers may still look for RoadPriorHead.
RoadPriorHead = StructurePriorHead


class ResNet34LocalDetailEncoder(nn.Module):
    """ResNet34 local-detail encoder, split for inserting RDFQ modules.

    Scales:
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
# 3. PEFT structure token alignment
# =========================================================
class StructureTokenAdapter(nn.Module):
    """External token-level adapter: A_l -> T_l.

    Internal DINO adapters adapt transformer blocks; this external adapter aligns
    multi-level DINO tokens into a curvilinear-structure token space before CNN
    queries sample from them.
    """
    def __init__(self, dim: int = 384, bottleneck: int = 96, dropout: float = 0.0, init_gamma: float = 1e-3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(bottleneck, dim)
        self.gamma = nn.Parameter(torch.ones(1, 1, dim) * float(init_gamma))
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.up(self.drop(self.act(self.down(self.norm(x)))))


class ExternalStructureTokenBank(nn.Module):
    """Layer-wise external adapters for DINO tokens."""
    def __init__(
        self,
        layers: Sequence[int] = (2, 5, 8, 11),
        dim: int = 384,
        bottleneck: int = 96,
        dropout: float = 0.0,
        init_gamma: float = 1e-3,
        enabled: bool = True,
    ):
        super().__init__()
        self.layers = tuple(int(x) for x in layers)
        self.enabled = bool(enabled)
        self.adapters = nn.ModuleDict({
            str(lid): StructureTokenAdapter(dim=dim, bottleneck=bottleneck, dropout=dropout, init_gamma=init_gamma)
            for lid in self.layers
        })

    def forward(self, tokens: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        if not self.enabled:
            return tokens
        out: Dict[int, torch.Tensor] = {}
        for lid in self.layers:
            if lid not in tokens:
                raise KeyError(f"DINO token bank missing layer {lid}; available={list(tokens.keys())}")
            out[lid] = self.adapters[str(lid)](tokens[lid])
        return out


# =========================================================
# 4. Reliable Deformable Foundation Query (RDFQ)
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
    """Pure PyTorch grid_sample multi-level deformable attention."""
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
            grid = base + off * scale
            grid = grid.permute(0, 1, 3, 4, 2, 5).contiguous().view(b * self.num_heads, hq, wq * self.num_points, 2)
            sampled = F.grid_sample(v, grid, mode="bilinear", padding_mode="border", align_corners=True)
            sampled = sampled.view(b, self.num_heads, self.head_dim, hq, wq, self.num_points)
            sampled = sampled.permute(0, 1, 2, 5, 3, 4).contiguous()
            weight = attn[:, :, li].unsqueeze(2)
            out = out + (sampled * weight).sum(dim=3)

        out = out.reshape(b, c, hq, wq)
        return self.out_proj(out)


class ReliableDeformableFoundationQuery(nn.Module):
    """RDFQ: reliable deformable DINO token query.

    c_i' = c_i + alpha_i * R_i * G_i * Z_i

    where Z_i is sampled DINO enhancement, G_i is a feature gate, and R_i is a
    scalar reliability map estimated from CNN-DINO agreement.
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
        reliability_init: float = 0.80,
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

        self.dino_prior_head = nn.Sequential(
            ConvGNAct(c_dim, max(16, min(64, c_dim // 4)), 1, gn_groups=gn_groups),
            nn.Conv2d(max(16, min(64, c_dim // 4)), 1, 1),
        )

        self.feature_gate = nn.Sequential(
            ConvGNAct(c_dim * 2 + 1, inner_dim, 1, gn_groups=gn_groups),
            DepthwiseSeparableConv(inner_dim, inner_dim, 3, gn_groups=gn_groups),
            nn.Conv2d(inner_dim, c_dim, 1),
            nn.Sigmoid(),
        )
        self.reliability_head = nn.Sequential(
            ConvGNAct(c_dim * 2 + 3, inner_dim, 1, gn_groups=gn_groups),
            DepthwiseSeparableConv(inner_dim, inner_dim, 3, gn_groups=gn_groups),
            nn.Conv2d(inner_dim, 1, 1),
            nn.Sigmoid(),
        )
        # Make the initial model close to RD_v4: DINO enhancement is allowed but still controlled by alpha/gate.
        final_conv = self.reliability_head[-2]
        if isinstance(final_conv, nn.Conv2d):
            nn.init.zeros_(final_conv.weight)
            nn.init.constant_(final_conv.bias, _logit(float(reliability_init)))
        self.alpha = BoundedScalar(max_value=0.60, init_value=0.03)

    def forward(self, c: torch.Tensor, structure_prior: torch.Tensor, dino_tokens: Sequence[torch.Tensor], dino_hw: Tuple[int, int]):
        if structure_prior.shape[-2:] != c.shape[-2:]:
            structure_prior = F.interpolate(structure_prior, size=c.shape[-2:], mode="bilinear", align_corners=False)
        c_tilde = c * (1.0 + self.lambda_prior() * structure_prior)
        q = self.q_proj(c_tilde)
        value_maps = self.value_projector(dino_tokens, dino_hw)
        z_inner = self.sampler(q, value_maps)
        z_inner = self.refine_inner(z_inner)
        z = self.out_proj(z_inner)

        dino_prior_logits = self.dino_prior_head(z)
        dino_prior = torch.sigmoid(dino_prior_logits)
        if dino_prior.shape[-2:] != structure_prior.shape[-2:]:
            dino_prior = F.interpolate(dino_prior, size=structure_prior.shape[-2:], mode="bilinear", align_corners=False)
            dino_prior_logits = F.interpolate(dino_prior_logits, size=structure_prior.shape[-2:], mode="bilinear", align_corners=False)

        disagreement = torch.abs(structure_prior - dino_prior)
        gate = self.feature_gate(torch.cat([c, z, structure_prior], dim=1))
        reliability = self.reliability_head(torch.cat([c, z, structure_prior, dino_prior, disagreement], dim=1))
        c_enh = c + self.alpha() * reliability * gate * z

        aux = {
            "z": z,
            "gate": gate,
            "gate_mean": gate.mean(),
            "reliability": reliability,
            "reliability_mean": reliability.mean(),
            "dino_prior_logits": dino_prior_logits,
            "dino_prior": dino_prior,
            "disagreement": disagreement,
            "alpha": self.alpha(),
            "lambda_prior": self.lambda_prior(),
        }
        return c_enh, aux


# Backward-compatible alias: old configs can still import/use RoadAwareDLAEM if needed.
RoadAwareDLAEM = ReliableDeformableFoundationQuery


# =========================================================
# 5. Topology-preserving decoder
# =========================================================
class ResidualDecoderBlock(nn.Module):
    """HL_base-style residual decoder block."""
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


class TopologyHead(nn.Module):
    """Lightweight topology head for centerline/skeleton supervision."""
    def __init__(self, in_ch: int = 64, out_ch: int = 1, mid_ch: int = 48):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(in_ch, mid_ch, kernel_size=3),
            DepthwiseSeparableConv(mid_ch, mid_ch, kernel_size=3, gn_groups=8),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# =========================================================
# 6. RD_v5 / RFG-CSNet
# =========================================================
class RD_v5(nn.Module):
    """RD_v5 / RFG-CSNet.

    Reliable Foundation-Guided Curvilinear Structure Segmentation Network.

    Main upgrades over RD_v4:
      1) Road-aware names -> structure-aware formulation.
      2) External DINO structure token adapters: A_l -> T_l.
      3) RDFQ4/8/16 with DINO prior, CNN-DINO disagreement and reliability map.
      4) Topology-preserving decoder with centerline output.

    Default forward returns a dict compatible with train_RD.py. Set return_aux=False
    to return only final logits for simple testing scripts.
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
        # External token adapter
        use_external_token_adapter: bool = True,
        external_adapter_bottleneck: int = 96,
        external_adapter_dropout: float = 0.0,
        # CNN / RDFQ
        resnet_pretrained: Optional[bool] = None,
        gn_groups: int = 16,
        rdfq4_heads: int = 4,
        rdfq4_points: int = 4,
        rdfq4_offset_scale: float = 3.0,
        rdfq4_dim: int = 64,
        rdfq8_heads: int = 4,
        rdfq16_heads: int = 8,
        rdfq_points: int = 6,
        rdfq8_offset_scale: float = 4.0,
        rdfq16_offset_scale: float = 5.0,
        rdfq8_dim: int = 256,
        rdfq16_dim: int = 256,
        reliability_init: float = 0.80,
        # Old config compatibility: maps DLAEM names to RDFQ names when provided.
        dlaem4_heads: Optional[int] = None,
        dlaem4_points: Optional[int] = None,
        dlaem4_offset_scale: Optional[float] = None,
        dlaem4_dim: Optional[int] = None,
        dlaem8_heads: Optional[int] = None,
        dlaem16_heads: Optional[int] = None,
        dlaem_points: Optional[int] = None,
        dlaem8_offset_scale: Optional[float] = None,
        dlaem16_offset_scale: Optional[float] = None,
        dlaem8_dim: Optional[int] = None,
        dlaem16_dim: Optional[int] = None,
        light_gate_dim: Optional[int] = None,
        prior_mid_max: int = 128,
        aux_mid_ch: int = 64,
        centerline_mid_ch: int = 48,
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

        # Backward-compatible DLAEM aliases.
        if dlaem4_heads is not None:
            rdfq4_heads = int(dlaem4_heads)
        if dlaem4_points is not None:
            rdfq4_points = int(dlaem4_points)
        if dlaem4_offset_scale is not None:
            rdfq4_offset_scale = float(dlaem4_offset_scale)
        if dlaem4_dim is not None:
            rdfq4_dim = int(dlaem4_dim)
        if dlaem8_heads is not None:
            rdfq8_heads = int(dlaem8_heads)
        if dlaem16_heads is not None:
            rdfq16_heads = int(dlaem16_heads)
        if dlaem_points is not None:
            rdfq_points = int(dlaem_points)
        if dlaem8_offset_scale is not None:
            rdfq8_offset_scale = float(dlaem8_offset_scale)
        if dlaem16_offset_scale is not None:
            rdfq16_offset_scale = float(dlaem16_offset_scale)
        if dlaem8_dim is not None:
            rdfq8_dim = int(dlaem8_dim)
        if dlaem16_dim is not None:
            rdfq16_dim = int(dlaem16_dim)

        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.return_aux = bool(return_aux)
        self.dino_layers = tuple(int(x) for x in dino_layers)
        if len(self.dino_layers) != 4:
            raise ValueError(f"RD_v5 需要 4 个 DINO 层，例如 [2,5,8,11]，当前: {self.dino_layers}")

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

        # 2) External structure token alignment: A_l -> T_l.
        self.structure_token_bank = ExternalStructureTokenBank(
            layers=self.dino_layers,
            dim=int(dino_embed_dim),
            bottleneck=int(external_adapter_bottleneck),
            dropout=float(external_adapter_dropout),
            enabled=bool(use_external_token_adapter),
        )

        # 3) Local detail encoder.
        self.cnn = ResNet34LocalDetailEncoder(in_channels=self.n_channels, pretrained=bool(resnet_pretrained))
        c2, c4, c8, c16, c32 = 64, 64, 128, 256, 512

        # 4) Structure prior heads.
        self.prior4 = StructurePriorHead(c4, mid_ch=max(32, min(int(prior_mid_max), c4 // 2)), gn_groups=gn_groups)
        self.prior8 = StructurePriorHead(c8, mid_ch=max(32, min(int(prior_mid_max), c8 // 4)), gn_groups=gn_groups)
        self.prior16 = StructurePriorHead(c16, mid_ch=max(32, min(int(prior_mid_max), c16 // 4)), gn_groups=gn_groups)

        # 5) Reliable deformable foundation query modules.
        self.rdfq4 = ReliableDeformableFoundationQuery(
            c_dim=c4,
            dino_dim=int(dino_embed_dim),
            inner_dim=int(rdfq4_dim),
            num_heads=int(rdfq4_heads),
            num_levels=2,
            num_points=int(rdfq4_points),
            offset_scale=float(rdfq4_offset_scale),
            gn_groups=int(gn_groups),
            reliability_init=float(reliability_init),
        )
        self.rdfq8 = ReliableDeformableFoundationQuery(
            c_dim=c8,
            dino_dim=int(dino_embed_dim),
            inner_dim=int(rdfq8_dim),
            num_heads=int(rdfq8_heads),
            num_levels=2,
            num_points=int(rdfq_points),
            offset_scale=float(rdfq8_offset_scale),
            gn_groups=int(gn_groups),
            reliability_init=float(reliability_init),
        )
        self.rdfq16 = ReliableDeformableFoundationQuery(
            c_dim=c16,
            dino_dim=int(dino_embed_dim),
            inner_dim=int(rdfq16_dim),
            num_heads=int(rdfq16_heads),
            num_levels=2,
            num_points=int(rdfq_points),
            offset_scale=float(rdfq16_offset_scale),
            gn_groups=int(gn_groups),
            reliability_init=float(reliability_init),
        )
        # Old attribute aliases for checkpoint/config compatibility.
        self.dlaem4 = self.rdfq4
        self.dlaem8 = self.rdfq8
        self.dlaem16 = self.rdfq16

        # 6) Topology-preserving decoder.
        self.dec3 = ResidualDecoderBlock(in_channels=c32, skip_channels=c16, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=c8, out_channels=128)
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=c4, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=c2, out_channels=64)

        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, self.n_classes, kernel_size=1),
        )
        self.centerline_head = TopologyHead(64, self.n_classes, mid_ch=int(centerline_mid_ch))

        # 7) Auxiliary supervision heads.
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

        # 1) PEFT-DINO token bank, then external structure-token adaptation.
        dino_out = self.dino(x)
        token_dict = self.structure_token_bank(dino_out["tokens"])
        dino_hw = dino_out["spatial_shape"]
        t2 = token_dict[self.dino_layers[0]]
        t5 = token_dict[self.dino_layers[1]]
        t8 = token_dict[self.dino_layers[2]]
        t11 = token_dict[self.dino_layers[3]]

        # 2) Local CNN encoder.
        x0, x1, x2 = self.cnn.forward_until_x2(x)

        # 3) Structure priors at 1/4 and 1/8.
        s4_logits, s4 = self.prior4(x1)
        s8_logits, s8 = self.prior8(x2)

        # 4) RDFQ4: x1 + S4 query T2/T5.
        x1_enh, aux4 = self.rdfq4(x1, s4, [t2, t5], dino_hw)

        # 5) RDFQ8: x2 + S8 query T5/T8, then layer3.
        x2_enh, aux8 = self.rdfq8(x2, s8, [t5, t8], dino_hw)
        x3 = self.cnn.forward_layer3(x2_enh)

        # 6) RDFQ16: x3 + S16 query T8/T11.
        s16_logits, s16 = self.prior16(x3)
        x3_enh, aux16 = self.rdfq16(x3, s16, [t8, t11], dino_hw)

        # 7) Bottleneck and topology-preserving decoder.
        x4 = self.cnn.forward_layer4(x3_enh)
        d3 = self.dec3(x4, x3_enh)        # H/16, 256
        d2 = self.dec2(d3, x2_enh)        # H/8,  128
        d1 = self.dec1(d2, x1_enh)        # H/4,   96
        d0 = self.dec0(d1, x0)            # H/2,   64

        logits_half = self.out_head(d0)   # H/2, n_classes
        logits = F.interpolate(logits_half, size=input_size, mode="bilinear", align_corners=False)
        centerline_half = self.centerline_head(d0)
        centerline_logits = F.interpolate(centerline_half, size=input_size, mode="bilinear", align_corners=False)

        logit16 = self.aux16_head(x3_enh)  # H/16
        logit8 = self.aux8_head(d2)        # H/8
        logit4 = self.aux4_head(d1)        # H/4
        logit2 = logits_half               # H/2

        outputs: Dict[str, Any] = {
            "logits": logits,
            "final_logits": logits,
            "logits_half": logits_half,
            "centerline_logits": centerline_logits,
            "centerline_logits_half": centerline_half,
            "logit16": logit16,
            "logit8": logit8,
            "logit4": logit4,
            "logit2": logit2,
            # Structure priors: new names.
            "structure_prior4_logits": s4_logits,
            "structure_prior8_logits": s8_logits,
            "structure_prior16_logits": s16_logits,
            "structure_prior4": s4,
            "structure_prior8": s8,
            "structure_prior16": s16,
            # Old prior names kept for train_RD.py compatibility.
            "prior4_logits": s4_logits,
            "prior8_logits": s8_logits,
            "prior16_logits": s16_logits,
            # DINO priors.
            "dino_prior4_logits": aux4["dino_prior_logits"],
            "dino_prior8_logits": aux8["dino_prior_logits"],
            "dino_prior16_logits": aux16["dino_prior_logits"],
            "dino_prior4": aux4["dino_prior"],
            "dino_prior8": aux8["dino_prior"],
            "dino_prior16": aux16["dino_prior"],
            # Reliability and disagreement maps.
            "reliability4": aux4["reliability"],
            "reliability8": aux8["reliability"],
            "reliability16": aux16["reliability"],
            "disagreement4": aux4["disagreement"],
            "disagreement8": aux8["disagreement"],
            "disagreement16": aux16["disagreement"],
            # Diagnostics: new RDFQ names.
            "rdfq4_alpha": aux4.get("alpha"),
            "rdfq8_alpha": aux8.get("alpha"),
            "rdfq16_alpha": aux16.get("alpha"),
            "rdfq4_gate_mean": aux4.get("gate_mean"),
            "rdfq8_gate_mean": aux8.get("gate_mean"),
            "rdfq16_gate_mean": aux16.get("gate_mean"),
            "rdfq4_reliability_mean": aux4.get("reliability_mean"),
            "rdfq8_reliability_mean": aux8.get("reliability_mean"),
            "rdfq16_reliability_mean": aux16.get("reliability_mean"),
            "rdfq4_lambda_prior": aux4.get("lambda_prior"),
            "rdfq8_lambda_prior": aux8.get("lambda_prior"),
            "rdfq16_lambda_prior": aux16.get("lambda_prior"),
            # Old diagnostics kept for old logger compatibility.
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
        add_group("structure_token_adapter", self.structure_token_bank.parameters(), default_lr_mult=0.90, default_wd=0.02)
        add_group("cnn_encoder", self.cnn.parameters(), default_lr_mult=0.55, default_wd=weight_decay)
        add_group("structure_priors", list(self.prior4.parameters()) + list(self.prior8.parameters()) + list(self.prior16.parameters()), default_lr_mult=1.00)
        add_group("rdfq", list(self.rdfq4.parameters()) + list(self.rdfq8.parameters()) + list(self.rdfq16.parameters()), default_lr_mult=1.00)
        decoder_params = (
            list(self.dec3.parameters()) + list(self.dec2.parameters()) + list(self.dec1.parameters()) + list(self.dec0.parameters())
            + list(self.out_head.parameters()) + list(self.centerline_head.parameters())
            + list(self.aux16_head.parameters()) + list(self.aux8_head.parameters()) + list(self.aux4_head.parameters())
        )
        add_group("topology_decoder", decoder_params, default_lr_mult=1.20)

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
        print("[RD_v5 / RFG-CSNet] Reliable Foundation-Guided Curvilinear Structure Segmentation")
        print(f"    - Total params:     {total / 1e6:.2f} M")
        print(f"    - Trainable params: {trainable / 1e6:.2f} M")
        print(f"    - Frozen params:    {frozen / 1e6:.2f} M")
        print(f"    - DINO total/train: {dino_total / 1e6:.2f} M / {dino_train / 1e6:.2f} M")
        print("    - Data flow: X -> PEFT-DINO A2/A5/A8/A11 -> T2/T5/T8/T11 + Local Encoder")
        print("    - Interaction: RDFQ4(x1,S4,T2/T5) + RDFQ8(x2,S8,T5/T8) + RDFQ16(x3,S16,T8/T11)")
        print("    - Reliability: DINO prior + CNN-DINO disagreement -> reliability map")
        print("    - Decoder: topology-preserving decoder with mask and centerline outputs")
        print("-" * 60)


# Paper-style alias.
RFG_CSNet = RD_v5


# =========================================================
# 7. Optional RD_v5 structure-aware loss utilities
# =========================================================
def _resize_like(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    logits = _resize_like(logits, target)
    prob = torch.sigmoid(logits)
    target = target.float()
    dims = tuple(range(1, prob.dim()))
    inter = (prob * target).sum(dim=dims)
    den = prob.sum(dim=dims) + target.sum(dim=dims)
    loss = 1.0 - (2.0 * inter + eps) / (den + eps)
    return loss.mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    logits = _resize_like(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    return bce + dice_loss_with_logits(logits, target)


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    if img.shape[1] != 1:
        # For multi-class binary heads, process channel-wise through grouped max pooling equivalent.
        return torch.cat([soft_erode(img[:, i:i+1]) for i in range(img.shape[1])], dim=1)
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    img = img.clamp(0, 1)
    skel = F.relu(img - soft_open(img))
    for _ in range(int(iters)):
        img = soft_erode(img)
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0, 1)


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    logits = _resize_like(logits, target)
    pred = torch.sigmoid(logits)
    target = target.float().clamp(0, 1)
    skel_pred = soft_skeletonize(pred, iters=iters)
    skel_true = soft_skeletonize(target, iters=iters)
    dims = tuple(range(1, pred.dim()))
    tprec = (skel_pred * target).sum(dim=dims) / (skel_pred.sum(dim=dims) + eps)
    tsens = (skel_true * pred).sum(dim=dims) / (skel_true.sum(dim=dims) + eps)
    cl = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return (1.0 - cl).mean()


class RDV5StructureLoss(nn.Module):
    """Optional multi-output loss for RD_v5.

    Existing train_RD.py may keep its original loss. This class is provided so
    RD_v5's extra outputs can be supervised when you are ready to enable the
    full PR-style training protocol.
    """
    def __init__(
        self,
        w_main: float = 1.0,
        w_deep: float = 0.35,
        w_structure_prior: float = 0.15,
        w_dino_prior: float = 0.10,
        w_centerline: float = 0.20,
        w_cldice: float = 0.20,
        cldice_iters: int = 10,
    ):
        super().__init__()
        self.w_main = float(w_main)
        self.w_deep = float(w_deep)
        self.w_structure_prior = float(w_structure_prior)
        self.w_dino_prior = float(w_dino_prior)
        self.w_centerline = float(w_centerline)
        self.w_cldice = float(w_cldice)
        self.cldice_iters = int(cldice_iters)

    def forward(self, outputs: Dict[str, torch.Tensor], mask: torch.Tensor, centerline: Optional[torch.Tensor] = None):
        if not isinstance(outputs, dict):
            outputs = {"logits": outputs}
        mask = mask.float()
        losses: Dict[str, torch.Tensor] = {}

        losses["loss_main"] = bce_dice_loss(outputs["logits"], mask)

        deep_terms = []
        for key, w in (("logit16", 0.20), ("logit8", 0.15), ("logit4", 0.10), ("logit2", 0.05)):
            if key in outputs:
                deep_terms.append(float(w) * bce_dice_loss(outputs[key], mask))
        losses["loss_deep"] = torch.stack(deep_terms).sum() if deep_terms else mask.new_tensor(0.0)

        sp_terms = []
        for key in ("structure_prior4_logits", "structure_prior8_logits", "structure_prior16_logits"):
            if key in outputs:
                sp_terms.append(bce_dice_loss(outputs[key], mask))
        losses["loss_structure_prior"] = torch.stack(sp_terms).mean() if sp_terms else mask.new_tensor(0.0)

        dp_terms = []
        for key in ("dino_prior4_logits", "dino_prior8_logits", "dino_prior16_logits"):
            if key in outputs:
                dp_terms.append(bce_dice_loss(outputs[key], mask))
        losses["loss_dino_prior"] = torch.stack(dp_terms).mean() if dp_terms else mask.new_tensor(0.0)

        if centerline is not None and "centerline_logits" in outputs:
            centerline = centerline.float()
            losses["loss_centerline"] = bce_dice_loss(outputs["centerline_logits"], centerline)
        else:
            losses["loss_centerline"] = mask.new_tensor(0.0)

        losses["loss_cldice"] = soft_cldice_loss(outputs["logits"], mask, iters=self.cldice_iters) if self.w_cldice > 0 else mask.new_tensor(0.0)

        total = (
            self.w_main * losses["loss_main"]
            + self.w_deep * losses["loss_deep"]
            + self.w_structure_prior * losses["loss_structure_prior"]
            + self.w_dino_prior * losses["loss_dino_prior"]
            + self.w_centerline * losses["loss_centerline"]
            + self.w_cldice * losses["loss_cldice"]
        )
        losses["loss_total"] = total
        return total, losses


if __name__ == "__main__":
    print("RD_v5 module loaded. Instantiate through models.get_model(config['model']).")
