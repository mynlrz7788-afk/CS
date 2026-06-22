"""
DC_v2_step2.py

第二步目标：
1. 继承 DC_v2_step1 已经跑通的 HL_base 分支和 DINOv3 + Adapter 手写串联 forward。
2. 不再做 DINO 四层注意力融合。
3. 在 DINO block 2 / 5 / 8 / 11 后插入四个强双向 SB-RTGFI 模块。
4. 每个 SB-RTGFI 内部都做 CNN -> DINO 和 DINO -> CNN 的相互指导。
5. Decoder 只使用 guided CNN features：F1、F2、F3、F4。
6. C3-D8 使用轻量可变形交叉注意力。
7. C4-D11 使用 DINO 控制的多感受野上下文。

建议放置路径：SEG/models/custom/DC_v2_step2.py
主模块名：DC_v2_step2
"""

import os
import sys
import importlib
import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["DC_v2_step2"]


def _auto_padding(kernel_size, dilation=1):
    if isinstance(kernel_size, tuple):
        if isinstance(dilation, tuple):
            return tuple(((k - 1) // 2) * d for k, d in zip(kernel_size, dilation))
        return tuple(((k - 1) // 2) * dilation for k in kernel_size)
    return ((kernel_size - 1) // 2) * dilation


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: int = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        act_layer=nn.ReLU,
        inplace: bool = True,
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
            nn.BatchNorm2d(out_channels),
            act_layer(inplace=inplace),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, dilation=1):
        super().__init__()
        padding = _auto_padding(kernel_size, dilation)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ECALayer(nn.Module):
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


class ResidualScale(nn.Module):
    """受限残差系数，避免 alpha / beta 训练初期过大破坏基线特征。"""
    def __init__(self, init_value: float = 0.1, max_value: float = 0.5):
        super().__init__()
        init_value = float(max(1e-4, min(init_value, max_value - 1e-4)))
        self.max_value = float(max_value)
        logit = math.log(init_value / (max_value - init_value))
        self.logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    def forward(self):
        return self.max_value * torch.sigmoid(self.logit)


class TokenAdapter(nn.Module):
    """DINOv3-PEFT 风格 token adapter。输入输出都是 B × N × dim。"""
    def __init__(self, dim: int = 384, bottleneck: int = 64, init_scale: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.act(self.down(self.norm(x))))
        return x + self.scale * residual


class RoadGeometryExtractor(nn.Module):
    """从 CNN 特征中提取道路几何提示。"""
    def __init__(self, in_channels: int, mid_channels: int = 64):
        super().__init__()
        self.reduce = ConvBNAct(in_channels, mid_channels, kernel_size=1)
        self.local = DepthwiseSeparableConv(mid_channels, mid_channels, kernel_size=3)
        self.h_strip = ConvBNAct(mid_channels, mid_channels, kernel_size=(1, 7))
        self.v_strip = ConvBNAct(mid_channels, mid_channels, kernel_size=(7, 1))
        self.dilate = DepthwiseSeparableConv(mid_channels, mid_channels, kernel_size=3, dilation=2)
        self.fuse = nn.Sequential(
            ConvBNAct(mid_channels * 4, mid_channels, kernel_size=1),
            DepthwiseSeparableConv(mid_channels, mid_channels, kernel_size=3),
        )

    def forward(self, x):
        x0 = self.reduce(x)
        l = self.local(x0)
        h = self.h_strip(x0)
        v = self.v_strip(x0)
        d = self.dilate(x0)
        return self.fuse(torch.cat([l, h, v, d], dim=1))


class DinoSemanticProjector(nn.Module):
    """把 384 通道 DINO feature map 投影到道路语义提示空间。"""
    def __init__(self, in_channels: int = 384, mid_channels: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, kernel_size=1),
            DepthwiseSeparableConv(mid_channels, mid_channels, kernel_size=3),
        )

    def forward(self, x):
        return self.proj(x)


class LightweightDeformCrossAttention2D(nn.Module):
    """轻量 2D 可变形交叉注意力。

    Query 来自 CNN，Value 来自 DINO。只在 C3-D8 的 64×64 层使用，避免显存过大。
    """
    def __init__(self, channels: int = 64, num_heads: int = 4, num_points: int = 4, offset_scale: float = 2.0):
        super().__init__()
        assert channels % num_heads == 0, "channels 必须能被 num_heads 整除"
        self.channels = channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = channels // num_heads
        self.offset_scale = float(offset_scale)

        self.v_proj = ConvBNAct(channels, channels, kernel_size=1)
        self.offset_weight = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=3),
            nn.Conv2d(channels, num_heads * num_points * 3, kernel_size=1),
        )
        self.out_proj = ConvBNAct(channels, channels, kernel_size=1)

    def forward(self, query: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        b, c, h, w = query.shape
        if value.shape[-2:] != (h, w):
            value = F.interpolate(value, size=(h, w), mode="bilinear", align_corners=False)

        v = self.v_proj(value).view(b, self.num_heads, self.head_dim, h, w)
        ow = self.offset_weight(torch.cat([query, value], dim=1))
        ow = ow.view(b, self.num_heads, self.num_points, 3, h, w)
        offsets = torch.tanh(ow[:, :, :, 0:2]) * self.offset_scale
        weights = torch.softmax(ow[:, :, :, 2], dim=2)

        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=query.device, dtype=query.dtype),
            torch.linspace(-1.0, 1.0, w, device=query.device, dtype=query.dtype),
            indexing="ij",
        )
        base_grid = torch.stack([xx, yy], dim=-1).view(1, 1, 1, h, w, 2)
        norm = torch.tensor([max((w - 1) / 2.0, 1.0), max((h - 1) / 2.0, 1.0)], device=query.device, dtype=query.dtype)
        offsets = offsets.permute(0, 1, 2, 4, 5, 3) / norm.view(1, 1, 1, 1, 1, 2)

        out = query.new_zeros(b, self.num_heads, self.head_dim, h, w)
        for head in range(self.num_heads):
            v_h = v[:, head]
            for point in range(self.num_points):
                grid = base_grid[:, :, :, :, :, :] + offsets[:, head:head + 1, point:point + 1]
                grid = grid.squeeze(1).squeeze(1)
                sampled = F.grid_sample(v_h, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                out[:, head] = out[:, head] + sampled * weights[:, head, point].unsqueeze(1)

        out = out.reshape(b, c, h, w)
        return self.out_proj(out)


class SBRTGFIBlock(nn.Module):
    """Strong Bidirectional RTGFI。

    内部强双向：CNN -> DINO，DINO -> CNN。
    外部输出单一路径：guided CNN feature F_i。
    """
    def __init__(
        self,
        cnn_channels: int,
        dino_channels: int = 384,
        mid_channels: int = 64,
        mode: str = "gate",
        alpha_init: float = 0.1,
        beta_init: float = 0.1,
        gamma_init: float = 0.05,
        deform_heads: int = 4,
        deform_points: int = 4,
        context_channels: int = 256,
    ):
        super().__init__()
        self.cnn_channels = cnn_channels
        self.dino_channels = dino_channels
        self.mid_channels = mid_channels
        self.mode = mode

        self.cnn_geo = RoadGeometryExtractor(cnn_channels, mid_channels)
        self.dino_sem = DinoSemanticProjector(dino_channels, mid_channels)

        self.dino_update = nn.Sequential(
            ConvBNAct(mid_channels * 3, mid_channels, kernel_size=3),
            DepthwiseSeparableConv(mid_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, dino_channels, kernel_size=1),
        )

        self.rel_gate = nn.Sequential(
            ConvBNAct(mid_channels * 3, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

        self.alpha = ResidualScale(alpha_init, max_value=0.5)
        self.beta = ResidualScale(beta_init, max_value=0.5)
        self.gamma = ResidualScale(gamma_init, max_value=0.3)

        # gate 模式，用在 C1-D2、C2-D5。
        self.cnn_gate = nn.Sequential(
            ConvBNAct(mid_channels * 3, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, cnn_channels, kernel_size=1),
        )
        self.cnn_pos = nn.Sequential(
            ConvBNAct(mid_channels * 3, mid_channels, kernel_size=3),
            DepthwiseSeparableConv(mid_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, cnn_channels, kernel_size=1),
        )
        self.cnn_neg = nn.Sequential(
            ConvBNAct(mid_channels * 3, mid_channels, kernel_size=3),
            DepthwiseSeparableConv(mid_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, cnn_channels, kernel_size=1),
        )

        # deform 模式，用在 C3-D8。
        self.q_proj = ConvBNAct(cnn_channels, mid_channels, kernel_size=1)
        self.deform_attn = LightweightDeformCrossAttention2D(mid_channels, deform_heads, deform_points)
        self.attn_out = nn.Sequential(
            ConvBNAct(mid_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, cnn_channels, kernel_size=1),
        )

        # context 模式，用在 C4-D11。
        self.context_reduce = ConvBNAct(cnn_channels, context_channels, kernel_size=1)
        self.ctx_l1 = DepthwiseSeparableConv(context_channels, context_channels, kernel_size=3, dilation=1)
        self.ctx_l3 = DepthwiseSeparableConv(context_channels, context_channels, kernel_size=3, dilation=3)
        self.ctx_l5 = DepthwiseSeparableConv(context_channels, context_channels, kernel_size=3, dilation=5)
        self.ctx_l7 = DepthwiseSeparableConv(context_channels, context_channels, kernel_size=3, dilation=7)
        self.ctx_proj = nn.ModuleList([nn.Conv2d(context_channels, mid_channels, kernel_size=1) for _ in range(4)])
        self.ctx_out = nn.Sequential(
            ConvBNAct(context_channels, context_channels, kernel_size=3),
            nn.Conv2d(context_channels, cnn_channels, kernel_size=1),
        )
        self.ctx_conf = nn.Sequential(
            ConvBNAct(mid_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    @staticmethod
    def _resize(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, cnn_feat: torch.Tensor, dino_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        cnn_size = cnn_feat.shape[-2:]
        dino_size = dino_feat.shape[-2:]

        cnn_geo = self.cnn_geo(cnn_feat)                         # B,64,Hc,Wc
        dino_sem = self.dino_sem(dino_feat)                      # B,64,64,64
        cnn_geo_to_dino = self._resize(cnn_geo, dino_size)
        diff_dino = torch.abs(cnn_geo_to_dino - dino_sem)

        # CNN -> DINO：差异感知几何注入。
        dino_update = self.dino_update(torch.cat([cnn_geo_to_dino, dino_sem, diff_dino], dim=1))
        beta = self.beta()
        dino_guided = dino_feat + beta * dino_update
        dino_sem_guided = self.dino_sem(dino_guided)

        # DINO -> CNN：先对齐到 CNN 尺度，再做不同形式的指导。
        dino_to_cnn = self._resize(dino_sem_guided, cnn_size)
        diff_cnn = torch.abs(cnn_geo - dino_to_cnn)
        h_cnn = torch.cat([cnn_geo, dino_to_cnn, diff_cnn], dim=1)
        rel = torch.sigmoid(self.rel_gate(h_cnn))
        alpha = self.alpha()
        gamma = self.gamma()

        if self.mode == "deform":
            # C3-D8：CNN query，DINO key/value，可变形交叉注意力。
            q = self.q_proj(cnn_feat)
            value = dino_sem_guided
            if value.shape[-2:] != q.shape[-2:]:
                value = F.interpolate(value, size=q.shape[-2:], mode="bilinear", align_corners=False)
            z = self.deform_attn(q, value)
            residual = self.attn_out(z)
            guided_cnn = cnn_feat + alpha * rel * residual

        elif self.mode == "context":
            # C4-D11：DINO 控制 CNN 多感受野上下文。
            f0 = self.context_reduce(cnn_feat)
            branches = [self.ctx_l1(f0), self.ctx_l3(f0), self.ctx_l5(f0), self.ctx_l7(f0)]
            scores = []
            d_ctx = dino_to_cnn
            for branch, proj in zip(branches, self.ctx_proj):
                e = proj(branch)
                scores.append((e * d_ctx).sum(dim=1, keepdim=True) / math.sqrt(float(self.mid_channels)))
            weights = torch.softmax(torch.cat(scores, dim=1), dim=1)
            f_ctx = 0.0
            for idx, branch in enumerate(branches):
                f_ctx = f_ctx + weights[:, idx:idx + 1] * branch
            conf = torch.sigmoid(self.ctx_conf(d_ctx))
            residual = self.ctx_out(f_ctx)
            guided_cnn = cnn_feat + alpha * conf * residual

        else:
            # C1-D2、C2-D5：强双向 gate + 正负残差。
            gate = torch.sigmoid(self.cnn_gate(h_cnn))
            pos = self.cnn_pos(h_cnn)
            neg = self.cnn_neg(h_cnn)
            guided_cnn = cnn_feat * (1.0 + alpha * gate) + alpha * rel * pos - gamma * rel * neg

        debug = {
            "alpha": alpha.detach(),
            "beta": beta.detach(),
            "gamma": gamma.detach(),
            "rel_mean": rel.detach().mean(),
            "diff_mean": diff_cnn.detach().mean(),
        }
        return dino_guided, guided_cnn, debug


class DinoPriorHead(nn.Module):
    """从单层 guided DINO feature 生成低分辨率道路先验。"""
    def __init__(self, in_channels: int = 384, mid_channels: int = 64, num_classes: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, kernel_size=1),
            DepthwiseSeparableConv(mid_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class DINOv3AdapterBranch(nn.Module):
    """带 Adapter 的 DINOv3 串联前向分支，支持在指定层插入 RTGFI 并回写 token。"""
    def __init__(
        self,
        dino_model_name: str = "dinov3_vits16",
        dino_repo_path: Optional[str] = None,
        dino_ckpt_path: Optional[str] = None,
        out_layers: Sequence[int] = (2, 5, 8, 11),
        embed_dim: int = 384,
        patch_size: int = 16,
        adapter_bottleneck: int = 64,
        adapter_init_scale: float = 0.1,
        dino_normalize: bool = False,
    ):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.dino_repo_path = dino_repo_path
        self.dino_ckpt_path = dino_ckpt_path
        self.out_layers = list(out_layers)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dino_normalize = dino_normalize

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.backbone = self._build_dino_backbone()
        self._freeze_dino_backbone()
        self.blocks = self._get_blocks()
        if len(self.blocks) == 0:
            raise RuntimeError("没有在 DINOv3 backbone 中找到 blocks，请检查本地 DINOv3 结构。")
        self.adapters = nn.ModuleList([
            TokenAdapter(dim=embed_dim, bottleneck=adapter_bottleneck, init_scale=adapter_init_scale)
            for _ in range(len(self.blocks))
        ])

    def _build_dino_backbone(self):
        if self.dino_repo_path is None:
            raise ValueError("需要提供 dino_repo_path，例如 /home/u2508183004/zyn/SEG/dinounet/dinov3")
        if not os.path.isdir(self.dino_repo_path):
            raise FileNotFoundError(f"找不到 dino_repo_path: {self.dino_repo_path}")

        dinov3_parent = os.path.dirname(self.dino_repo_path)
        if dinov3_parent not in sys.path:
            sys.path.insert(0, dinov3_parent)
        try:
            backbones = importlib.import_module("dinov3.hub.backbones")
        except Exception as e:
            raise ImportError(
                "导入 dinov3.hub.backbones 失败。请确认 dino_repo_path 指向 .../dinounet/dinov3。\n"
                f"原始错误：{repr(e)}"
            )
        if not hasattr(backbones, self.dino_model_name):
            available = [n for n in dir(backbones) if n.startswith("dinov3_")]
            raise AttributeError(f"找不到 {self.dino_model_name}，可用模型示例：{available[:30]}")
        model = getattr(backbones, self.dino_model_name)(pretrained=False)

        if self.dino_ckpt_path:
            if not os.path.isfile(self.dino_ckpt_path):
                raise FileNotFoundError(f"找不到 DINO 权重文件: {self.dino_ckpt_path}")
            ckpt = torch.load(self.dino_ckpt_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    state_dict = ckpt["model"]
                elif "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                elif "teacher" in ckpt:
                    state_dict = ckpt["teacher"]
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt
            clean_state = {}
            for k, v in state_dict.items():
                nk = k
                for prefix in ("module.", "backbone.", "student.", "teacher."):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                clean_state[nk] = v
            missing, unexpected = model.load_state_dict(clean_state, strict=False)
            print(
                f"[DC_v2_step2] 加载 DINOv3 权重完成: {self.dino_ckpt_path}\n"
                f"        missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
            )
        return model

    def _freeze_dino_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _get_blocks(self) -> nn.ModuleList:
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks
        if hasattr(self.backbone, "block_chunks"):
            blocks = []
            for chunk in self.backbone.block_chunks:
                for blk in chunk:
                    if not isinstance(blk, nn.Identity):
                        blocks.append(blk)
            return nn.ModuleList(blocks)
        return nn.ModuleList([])

    def _unwrap_tokens(self, obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            for key in ("x", "tokens", "x_prenorm"):
                if key in obj and torch.is_tensor(obj[key]):
                    return obj[key]
            for v in obj.values():
                if torch.is_tensor(v) and v.ndim == 3:
                    return v
            raise RuntimeError(f"DINO 返回 dict，但找不到 token Tensor，keys={list(obj.keys())}")
        if isinstance(obj, (tuple, list)):
            for v in obj:
                if torch.is_tensor(v) and v.ndim == 3:
                    return v
            if len(obj) > 0:
                return self._unwrap_tokens(obj[0])
            raise RuntimeError("DINO 返回空 tuple/list，无法取得 token。")
        raise RuntimeError(f"无法识别的 DINO token 类型: {type(obj)}")

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "prepare_tokens_with_masks"):
            try:
                tokens = self.backbone.prepare_tokens_with_masks(x, None)
            except TypeError:
                tokens = self.backbone.prepare_tokens_with_masks(x)
            return self._unwrap_tokens(tokens)

        if hasattr(self.backbone, "prepare_tokens"):
            tokens = self.backbone.prepare_tokens(x)
            return self._unwrap_tokens(tokens)

        if not hasattr(self.backbone, "patch_embed"):
            raise RuntimeError("当前 DINOv3 backbone 没有 patch_embed / prepare_tokens_with_masks，无法手写 forward。")

        tokens = self.backbone.patch_embed(x)
        if tokens.ndim == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
        b = tokens.shape[0]

        if hasattr(self.backbone, "cls_token"):
            cls = self.backbone.cls_token.expand(b, -1, -1)
            if hasattr(self.backbone, "register_tokens") and self.backbone.register_tokens is not None:
                reg = self.backbone.register_tokens.expand(b, -1, -1)
                tokens = torch.cat([cls, reg, tokens], dim=1)
            else:
                tokens = torch.cat([cls, tokens], dim=1)

        if hasattr(self.backbone, "interpolate_pos_encoding"):
            try:
                tokens = tokens + self.backbone.interpolate_pos_encoding(tokens, x.shape[-2], x.shape[-1])
            except TypeError:
                tokens = tokens + self.backbone.interpolate_pos_encoding(tokens, x)
        elif hasattr(self.backbone, "pos_embed"):
            pos = self.backbone.pos_embed
            if pos.shape[1] == tokens.shape[1]:
                tokens = tokens + pos

        return self._unwrap_tokens(tokens)

    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = self._unwrap_tokens(x)
        try:
            y = block(x)
        except TypeError:
            y = block(x, None)
        return self._unwrap_tokens(y)

    def _tokens_to_map(self, tokens: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        patch_n = patch_h * patch_w
        special_n = tokens.shape[1] - patch_n
        if special_n < 0:
            raise RuntimeError(f"token 数 {tokens.shape[1]} 小于 patch 数 {patch_n}，无法 reshape。")
        patch_tokens = tokens[:, special_n:, :]
        return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w).contiguous()

    def _map_to_tokens(self, old_tokens: torch.Tensor, fmap: torch.Tensor) -> torch.Tensor:
        b, c, h, w = fmap.shape
        patch_n = h * w
        special_n = old_tokens.shape[1] - patch_n
        if special_n < 0:
            raise RuntimeError(f"token 数 {old_tokens.shape[1]} 小于 patch 数 {patch_n}，无法回写 token。")
        special = old_tokens[:, :special_n, :]
        patch = fmap.flatten(2).transpose(1, 2).contiguous()
        return torch.cat([special, patch], dim=1) if special_n > 0 else patch

    def forward_with_rtgfi(
        self,
        x: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        c3: torch.Tensor,
        c4: torch.Tensor,
        rtgfi1: nn.Module,
        rtgfi2: nn.Module,
        rtgfi3: nn.Module,
        rtgfi4: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        if self.dino_normalize:
            x = (x - self.mean) / self.std
        _, _, h, w = x.shape
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        tokens = self._prepare_tokens(x)
        debug = {}
        guided_dino = {}
        f1 = f2 = f3 = f4 = None

        for i, block in enumerate(self.blocks):
            tokens = self._run_block(block, tokens)
            tokens = self.adapters[i](tokens)

            if i == self.out_layers[0]:
                d2 = self._tokens_to_map(tokens, patch_h, patch_w)
                d2_g, f1, dbg = rtgfi1(c1, d2)
                guided_dino[i] = d2_g
                debug["rtgfi1"] = dbg
                tokens = self._map_to_tokens(tokens, d2_g)

            elif i == self.out_layers[1]:
                d5 = self._tokens_to_map(tokens, patch_h, patch_w)
                d5_g, f2, dbg = rtgfi2(c2, d5)
                guided_dino[i] = d5_g
                debug["rtgfi2"] = dbg
                tokens = self._map_to_tokens(tokens, d5_g)

            elif i == self.out_layers[2]:
                d8 = self._tokens_to_map(tokens, patch_h, patch_w)
                d8_g, f3, dbg = rtgfi3(c3, d8)
                guided_dino[i] = d8_g
                debug["rtgfi3"] = dbg
                tokens = self._map_to_tokens(tokens, d8_g)

            elif i == self.out_layers[3]:
                d11 = self._tokens_to_map(tokens, patch_h, patch_w)
                d11_g, f4, dbg = rtgfi4(c4, d11)
                guided_dino[i] = d11_g
                debug["rtgfi4"] = dbg

        if any(v is None for v in (f1, f2, f3, f4)):
            raise RuntimeError("SB-RTGFI 没有得到完整 F1/F2/F3/F4，请检查 out_layers 和 blocks 数量。")

        return {
            "F1": f1,
            "F2": f2,
            "F3": f3,
            "F4": f4,
            "guided_dino": guided_dino,
            "debug": debug,
        }


class DecoderBlock(nn.Module):
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

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([x, skip], dim=1)
        out = self.fuse(f)
        out = self.eca(out)
        out = out + self.shortcut(f)
        return self.act(out)


class GuidedCNNDecoder(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 96)
        self.final_up = nn.Sequential(
            ConvBNAct(96, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
        )
        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, f1, f2, f3, f4, input_size: Tuple[int, int]):
        d3 = self.dec3(f4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        d0 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        d0 = self.final_up(d0)
        logits_half = self.out_head(d0)
        logits = F.interpolate(logits_half, size=input_size, mode="bilinear", align_corners=False)
        return logits, logits_half, {"d3": d3, "d2": d2, "d1": d1, "d0": d0}


class DC_v2_step2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        n_channels: int = 3,
        pretrained: bool = True,
        return_aux: bool = False,
        dino_model_name: str = "dinov3_vits16",
        dino_repo_path: Optional[str] = None,
        dino_ckpt_path: Optional[str] = None,
        dino_layers: Sequence[int] = (2, 5, 8, 11),
        dino_embed_dim: int = 384,
        dino_patch_size: int = 16,
        adapter_bottleneck: int = 64,
        adapter_init_scale: float = 0.1,
        rtgfi_mid_channels: int = 64,
        deform_heads: int = 4,
        deform_points: int = 4,
        dino_normalize: bool = False,
        dino_prior_source: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.return_aux = return_aux
        self.dino_layers = list(dino_layers)
        self.dino_prior_source = int(dino_prior_source)

        from models.baselines.HL_base import HL_base

        self.hl_base = HL_base(
            n_channels=n_channels,
            n_classes=num_classes,
            pretrained=pretrained,
            return_aux=False,
        )

        self.dino = DINOv3AdapterBranch(
            dino_model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            out_layers=dino_layers,
            embed_dim=dino_embed_dim,
            patch_size=dino_patch_size,
            adapter_bottleneck=adapter_bottleneck,
            adapter_init_scale=adapter_init_scale,
            dino_normalize=dino_normalize,
        )

        self.rtgfi1 = SBRTGFIBlock(64, dino_embed_dim, rtgfi_mid_channels, mode="gate", alpha_init=0.05, beta_init=0.05, gamma_init=0.03)
        self.rtgfi2 = SBRTGFIBlock(128, dino_embed_dim, rtgfi_mid_channels, mode="gate", alpha_init=0.10, beta_init=0.10, gamma_init=0.05)
        self.rtgfi3 = SBRTGFIBlock(
            256,
            dino_embed_dim,
            rtgfi_mid_channels,
            mode="deform",
            alpha_init=0.20,
            beta_init=0.20,
            gamma_init=0.05,
            deform_heads=deform_heads,
            deform_points=deform_points,
        )
        self.rtgfi4 = SBRTGFIBlock(512, dino_embed_dim, rtgfi_mid_channels, mode="context", alpha_init=0.20, beta_init=0.10, gamma_init=0.05)

        self.decoder = GuidedCNNDecoder(num_classes=num_classes)
        self.dino_prior_head = DinoPriorHead(in_channels=dino_embed_dim, mid_channels=rtgfi_mid_channels, num_classes=num_classes)

    def set_return_aux(self, flag: bool):
        self.return_aux = bool(flag)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[-2:]
        base_logits, hl_aux = self.hl_base.forward_features(x)

        # 这里沿用当前讨论里的 C1-C4 定义：H/4, H/8, H/16, H/32。
        c1, c2, c3, c4 = hl_aux["x1"], hl_aux["x2"], hl_aux["x3"], hl_aux["x4"]

        rtgfi_out = self.dino.forward_with_rtgfi(
            x,
            c1,
            c2,
            c3,
            c4,
            self.rtgfi1,
            self.rtgfi2,
            self.rtgfi3,
            self.rtgfi4,
        )
        f1, f2, f3, f4 = rtgfi_out["F1"], rtgfi_out["F2"], rtgfi_out["F3"], rtgfi_out["F4"]
        logits, aux_logits, dec_aux = self.decoder(f1, f2, f3, f4, input_size=input_size)

        guided_dino = rtgfi_out["guided_dino"]
        if self.dino_prior_source in guided_dino:
            dino_prior_feat = guided_dino[self.dino_prior_source]
        else:
            # 兜底用最后一层。
            dino_prior_feat = guided_dino[self.dino_layers[-1]]
        dino_prior_logits = self.dino_prior_head(dino_prior_feat)

        if not self.return_aux:
            return logits

        return {
            "final_logits": logits,
            "coarse_logits": logits,
            "base_logits": base_logits,
            "aux_logits": aux_logits,
            "dino_prior_logits": dino_prior_logits,
            "F1": f1,
            "F2": f2,
            "F3": f3,
            "F4": f4,
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "C4": c4,
            "decoder_aux": dec_aux,
            "rtgfi_debug": rtgfi_out["debug"],
        }


if __name__ == "__main__":
    print("DC_v2_step2 module loaded.")
