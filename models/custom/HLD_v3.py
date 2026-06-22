import os
import sys
import math
import importlib
from typing import Dict, Tuple, Optional, Sequence, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["HLD_v3"]


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
    """learnable scalar in [0, max_value], initialized near init_value."""
    def __init__(self, max_value: float, init_value: float):
        super().__init__()
        self.max_value = float(max_value)
        ratio = float(init_value) / max(self.max_value, 1e-8)
        self.raw = nn.Parameter(torch.tensor(_logit(ratio), dtype=torch.float32))

    def forward(self):
        return self.max_value * torch.sigmoid(self.raw)


class ConvBNAct(nn.Module):
    """Conv-BN-ReLU，保持 HL_base 的 encoder/decoder 风格。"""
    def __init__(
        self,
        in_channels,
        out_channels,
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
    """Conv-GN-GELU，用于 DINO adapter / pyramid / prior 分支。"""
    def __init__(
        self,
        in_channels,
        out_channels,
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


class ECALayer(nn.Module):
    """Efficient Channel Attention（高效通道注意力），与 HL_base 保持一致。"""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class ResidualRefineGN(nn.Module):
    """轻量残差细化块，用于 DINO guidance 分支。"""
    def __init__(self, channels, gn_groups=8):
        super().__init__()
        self.conv1 = ConvGNAct(channels, channels, kernel_size=3, gn_groups=gn_groups)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _make_gn(channels, gn_groups),
        )
        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return self.act(x + y)


class ResidualDecoderBlock(nn.Module):
    """HL_base 同款残差解码块：up decoder feature → concat skip → residual refine。"""
    def __init__(self, in_channels, skip_channels, out_channels):
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
        feat = torch.cat([x, skip], dim=1)
        out = self.fuse(feat)
        out = self.eca(out)
        out = out + self.shortcut(feat)
        return self.act(out)


class FrozenDINOv3TokenExtractor(nn.Module):
    """
    Frozen DINOv3 token extractor.

    输出：
        {layer_id: B × N × C}
    N = H/patch_size × W/patch_size。
    """
    def __init__(
        self,
        dino_model_name="dinov3_vits16",
        dino_repo_path="/home/u2508183004/zyn/SEG/dinounet/dinov3",
        dino_ckpt_path="/home/u2508183004/zyn/SEG/weight/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        out_layers=(2, 5, 8, 11),
        embed_dim=384,
        patch_size=16,
        dino_normalize=False,
        dino_intermediate_norm=False,
    ):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.dino_repo_path = dino_repo_path
        self.dino_ckpt_path = dino_ckpt_path
        self.out_layers = [int(x) for x in out_layers]
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.dino_normalize = bool(dino_normalize)
        self.dino_intermediate_norm = bool(dino_intermediate_norm)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.backbone = self._build_dino_backbone()
        self._freeze_backbone()
        print(
            "[HLD_v3] Frozen DINOv3 token extractor ready. "
            f"out_layers={self.out_layers}, patch_size={self.patch_size}"
        )

    def _build_dino_backbone(self):
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
                f"原始错误: {repr(e)}"
            )

        if not hasattr(backbones, self.dino_model_name):
            available = [n for n in dir(backbones) if n.startswith("dinov3_")]
            raise AttributeError(
                f"在 dinov3.hub.backbones 中找不到模型: {self.dino_model_name}\n"
                f"可用模型示例: {available[:30]}"
            )

        model = getattr(backbones, self.dino_model_name)(pretrained=False)

        if self.dino_ckpt_path:
            if not os.path.isfile(self.dino_ckpt_path):
                raise FileNotFoundError(f"找不到 DINOv3 权重文件: {self.dino_ckpt_path}")
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
                for prefix in ("module.", "backbone.", "student.", "teacher.", "model."):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                clean_state[nk] = v

            missing, unexpected = model.load_state_dict(clean_state, strict=False)
            print(
                f"[HLD_v3] 加载 DINOv3 权重完成: {self.dino_ckpt_path}\n"
                f"          missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
            )
            if len(missing) > 0:
                print(f"          missing 示例: {missing[:10]}")
            if len(unexpected) > 0:
                print(f"          unexpected 示例: {unexpected[:10]}")
        return model

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _feat_to_tokens(feat, h, w, patch_size):
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        patch_h = h // patch_size
        patch_w = w // patch_size
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
            # 一些实现会带 cls / register token，取最后 patch_n 个 patch token。
            feat = feat[:, n - patch_n:, :]
        if feat.shape[1] != patch_n:
            raise RuntimeError(
                f"DINO token 无法对齐 patch grid。当前 token 数={feat.shape[1]}, "
                f"预期 patch 数={patch_n}, 输入尺寸={h}x{w}, patch_size={patch_size}"
            )
        return feat.contiguous()

    @torch.no_grad()
    def forward(self, x):
        if self.dino_normalize:
            x = (x - self.mean) / self.std
        _, _, h, w = x.shape
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise RuntimeError(
                f"DINO 输入尺寸必须能被 patch_size 整除。当前输入: {h}x{w}, patch_size={self.patch_size}"
            )
        if not hasattr(self.backbone, "get_intermediate_layers"):
            raise RuntimeError("当前 DINOv3 backbone 没有 get_intermediate_layers。")

        try:
            feats = self.backbone.get_intermediate_layers(
                x,
                n=self.out_layers,
                reshape=False,
                return_class_token=False,
                norm=self.dino_intermediate_norm,
            )
        except TypeError:
            try:
                feats = self.backbone.get_intermediate_layers(
                    x,
                    n=self.out_layers,
                    reshape=False,
                    return_class_token=False,
                )
            except TypeError:
                feats = self.backbone.get_intermediate_layers(
                    x,
                    n=self.out_layers,
                    reshape=True,
                    return_class_token=False,
                )

        if isinstance(feats, torch.Tensor):
            feats = [feats]
        feats = list(feats)
        if len(feats) != len(self.out_layers):
            raise RuntimeError(f"DINO 输出层数不匹配。期望 {len(self.out_layers)} 层，实际得到 {len(feats)} 层。")

        outputs = {}
        for layer, feat in zip(self.out_layers, feats):
            outputs[int(layer)] = self._feat_to_tokens(feat, h, w, self.patch_size)
        return outputs


class StructurePrompt16(nn.Module):
    """CNN 高层结构提示：由 x3 + up(x4) 生成 H/16 structure prompt。"""
    def __init__(self, in_channels=768, out_channels=128, gn_groups=8, strip_kernel=7):
        super().__init__()
        self.proj = ConvGNAct(in_channels, out_channels, kernel_size=1, gn_groups=gn_groups)
        self.local = ConvGNAct(out_channels, out_channels, kernel_size=3, gn_groups=gn_groups)
        self.strip_h = ConvGNAct(out_channels, out_channels, kernel_size=(1, strip_kernel), gn_groups=gn_groups)
        self.strip_v = ConvGNAct(out_channels, out_channels, kernel_size=(strip_kernel, 1), gn_groups=gn_groups)
        self.fuse = nn.Sequential(
            ConvGNAct(out_channels * 3, out_channels, kernel_size=1, gn_groups=gn_groups),
            ResidualRefineGN(out_channels, gn_groups=gn_groups),
        )

    def forward(self, x3, x4):
        x4_up = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x3, x4_up], dim=1)
        x = self.proj(x)
        local = self.local(x)
        h = self.strip_h(x)
        v = self.strip_v(x)
        return self.fuse(torch.cat([local, h, v], dim=1))


class HighLevelDINOAdapter(nn.Module):
    """CNN -> DINO：用 SP16 轻量校准高层 DINO token map。"""
    def __init__(
        self,
        dino_dim=384,
        prompt_dim=128,
        adapter_dim=128,
        alpha_max=0.10,
        alpha_init=0.01,
        gn_groups=8,
    ):
        super().__init__()
        self.dino_proj = ConvGNAct(dino_dim, adapter_dim, kernel_size=1, gn_groups=gn_groups)
        self.prompt_proj = ConvGNAct(prompt_dim, adapter_dim, kernel_size=1, gn_groups=gn_groups)
        self.delta = nn.Sequential(
            ConvGNAct(adapter_dim * 2, adapter_dim, kernel_size=3, gn_groups=gn_groups),
            ConvGNAct(adapter_dim, adapter_dim, kernel_size=3, gn_groups=gn_groups),
            nn.Conv2d(adapter_dim, dino_dim, kernel_size=1, bias=True),
        )
        self.conf_gate = nn.Sequential(
            nn.Conv2d(dino_dim, max(dino_dim // 4, 32), kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(max(dino_dim // 4, 32), 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = BoundedScalar(max_value=alpha_max, init_value=alpha_init)

        # 初始不扰动 Frozen DINO。
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        nn.init.zeros_(self.conf_gate[-2].weight)
        nn.init.constant_(self.conf_gate[-2].bias, 0.0)

    def forward(self, token_map, sp16):
        if sp16.shape[-2:] != token_map.shape[-2:]:
            sp16 = F.interpolate(sp16, size=token_map.shape[-2:], mode="bilinear", align_corners=False)
        a = self.dino_proj(token_map)
        p = self.prompt_proj(sp16)
        delta = self.delta(torch.cat([a, p], dim=1))
        conf = self.conf_gate(token_map)
        return token_map + self.alpha() * conf * delta


class DINOSemanticPyramid(nn.Module):
    """从 DINO T2/T5/T8'/T11' 生成 D32/D16/D8/D4/D2。"""
    def __init__(self, dino_dim=384, gn_groups=8):
        super().__init__()
        self.fuse16 = nn.Sequential(
            ConvGNAct(dino_dim * 2, 256, kernel_size=1, gn_groups=gn_groups),
            ResidualRefineGN(256, gn_groups=gn_groups),
        )
        self.fuse8_raw = nn.Sequential(
            ConvGNAct(dino_dim * 2, 128, kernel_size=1, gn_groups=gn_groups),
            ResidualRefineGN(128, gn_groups=gn_groups),
        )
        self.top16_to_8 = ConvGNAct(256, 128, kernel_size=1, gn_groups=gn_groups)
        self.refine8 = ResidualRefineGN(128, gn_groups=gn_groups)

        self.fuse4_raw = nn.Sequential(
            ConvGNAct(dino_dim * 2, 64, kernel_size=1, gn_groups=gn_groups),
            ResidualRefineGN(64, gn_groups=gn_groups),
        )
        self.top8_to_4 = ConvGNAct(128, 64, kernel_size=1, gn_groups=gn_groups)
        self.refine4 = ResidualRefineGN(64, gn_groups=gn_groups)

        self.to32 = nn.Sequential(
            ConvGNAct(256, 512, kernel_size=1, gn_groups=gn_groups),
            ResidualRefineGN(512, gn_groups=gn_groups),
        )
        self.to2 = nn.Sequential(
            ConvGNAct(64, 64, kernel_size=3, gn_groups=gn_groups),
            ResidualRefineGN(64, gn_groups=gn_groups),
        )

    def forward(self, t2, t5, t8, t11, size32, size16, size8, size4, size2):
        d16 = self.fuse16(torch.cat([t8, t11], dim=1))
        if d16.shape[-2:] != size16:
            d16 = F.interpolate(d16, size=size16, mode="bilinear", align_corners=False)

        d8 = self.fuse8_raw(torch.cat([t5, t8], dim=1))
        d8 = F.interpolate(d8, size=size8, mode="bilinear", align_corners=False)
        d8 = self.refine8(d8 + self.top16_to_8(F.interpolate(d16, size=size8, mode="bilinear", align_corners=False)))

        d4 = self.fuse4_raw(torch.cat([t2, t5], dim=1))
        d4 = F.interpolate(d4, size=size4, mode="bilinear", align_corners=False)
        d4 = self.refine4(d4 + self.top8_to_4(F.interpolate(d8, size=size4, mode="bilinear", align_corners=False)))

        d32 = F.interpolate(d16, size=size32, mode="bilinear", align_corners=False)
        d32 = self.to32(d32)

        d2 = F.interpolate(d4, size=size2, mode="bilinear", align_corners=False)
        d2 = self.to2(d2)
        return d32, d16, d8, d4, d2


class RoadPriorHead(nn.Module):
    """DINO road prior head：监督 D4 具备道路/非道路判别能力。"""
    def __init__(self, in_channels=64, hidden=64, out_channels=1, gn_groups=8):
        super().__init__()
        self.head = nn.Sequential(
            ConvGNAct(in_channels, hidden, kernel_size=3, gn_groups=gn_groups),
            ConvGNAct(hidden, max(hidden // 2, 16), kernel_size=3, gn_groups=gn_groups),
            nn.Conv2d(max(hidden // 2, 16), out_channels, kernel_size=1),
        )

    def forward(self, d4):
        return self.head(d4)


class SkeletonPriorHead(nn.Module):
    """DINO skeleton prior head：从 D16/D8/D4 生成 H/4 道路中心线/连通主轴先验。"""
    def __init__(self, c16=256, c8=128, c4=64, mid=64, out_channels=1, gn_groups=8):
        super().__init__()
        self.p16 = ConvGNAct(c16, mid, kernel_size=1, gn_groups=gn_groups)
        self.p8 = ConvGNAct(c8, mid, kernel_size=1, gn_groups=gn_groups)
        self.p4 = ConvGNAct(c4, mid, kernel_size=1, gn_groups=gn_groups)
        self.head = nn.Sequential(
            ConvGNAct(mid * 3, mid, kernel_size=1, gn_groups=gn_groups),
            ConvGNAct(mid, mid, kernel_size=3, gn_groups=gn_groups),
            ConvGNAct(mid, max(mid // 2, 16), kernel_size=3, gn_groups=gn_groups),
            nn.Conv2d(max(mid // 2, 16), out_channels, kernel_size=1),
        )

    def forward(self, d16, d8, d4):
        size = d4.shape[-2:]
        p16 = F.interpolate(self.p16(d16), size=size, mode="bilinear", align_corners=False)
        p8 = F.interpolate(self.p8(d8), size=size, mode="bilinear", align_corners=False)
        p4 = self.p4(d4)
        return self.head(torch.cat([p16, p8, p4], dim=1))


class GuidedSkipRefiner(nn.Module):
    """
    DINO -> CNN：semantic-topology residual suppression + DINO residual injection。

    x_safe = x * [1 - rho * (1-M) * (1-K)]
    s      = x_safe + beta * G * Proj(D)
    """
    def __init__(
        self,
        x_channels: int,
        d_channels: int,
        rho_max: float,
        beta_max: float = 0.30,
        rho_init: float = 0.05,
        beta_init: float = 0.05,
        gn_groups: int = 8,
        gate_bias: float = 2.0,
    ):
        super().__init__()
        self.d_proj = ConvGNAct(d_channels, x_channels, kernel_size=1, gn_groups=gn_groups)
        hidden = max(x_channels // 2, 16)
        self.semantic_gate = nn.Sequential(
            ConvGNAct(d_channels, hidden, kernel_size=3, gn_groups=gn_groups),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.semantic_gate[-2].bias, float(gate_bias))

        self.inject_gate = nn.Sequential(
            ConvGNAct(x_channels * 2 + 1, hidden, kernel_size=3, gn_groups=gn_groups),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.inject_gate[-2].bias)

        self.rho = BoundedScalar(max_value=rho_max, init_value=rho_init)
        self.beta = BoundedScalar(max_value=beta_max, init_value=beta_init)
        self.post = nn.Identity()

    def forward(self, x, d, k, return_aux: bool = False):
        if d.shape[-2:] != x.shape[-2:]:
            d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if k.shape[-2:] != x.shape[-2:]:
            k = F.interpolate(k, size=x.shape[-2:], mode="bilinear", align_corners=False)
        k = k.clamp(0.0, 1.0)

        m = self.semantic_gate(d)
        rho = self.rho()
        suppress = 1.0 - rho * (1.0 - m) * (1.0 - k)
        suppress = suppress.clamp(1.0 - float(rho.detach().max()), 1.0)
        x_safe = x * suppress

        d_proj = self.d_proj(d)
        g = self.inject_gate(torch.cat([x_safe, d_proj, k], dim=1))
        beta = self.beta()
        out = self.post(x_safe + beta * g * d_proj)

        if return_aux:
            return out, {
                "semantic_gate": m,
                "inject_gate": g,
                "suppress": suppress,
                "rho": rho.detach(),
                "beta": beta.detach(),
            }
        return out


class HLD_v3(nn.Module):
    """
    HLD_v3: DINO Semantic-Topology Mutual Guidance Network.

    设计原则：
    1) 保留 HL_base 的 ResNet34 encoder + ResidualDecoderBlock decoder 作为稳定底盘；
    2) Frozen DINO 生成多尺度 semantic pyramid、road prior、skeleton prior；
    3) CNN 高层结构提示轻量校准 DINO 高层 token；
    4) DINO semantic/topology prior 对每一级 CNN skip 做 residual suppression 和 residual injection；
    5) decoder 仍走 HL_base 主线：dec3(s4,s3) → dec2 → dec1 → dec0。
    """
    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        num_classes=None,
        in_channels=None,
        pretrained=True,
        return_aux=False,
        dino_model_name="dinov3_vits16",
        dino_repo_path="/home/u2508183004/zyn/SEG/dinounet/dinov3",
        dino_ckpt_path="/home/u2508183004/zyn/SEG/weight/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        dino_layers=(2, 5, 8, 11),
        dino_embed_dim=384,
        dino_patch_size=16,
        dino_normalize=False,
        dino_intermediate_norm=False,
        adapter_dim=128,
        adapter_alpha_max=0.10,
        adapter_alpha_init=0.01,
        structure_prompt_dim=128,
        gn_groups=8,
        rho32_max=0.25,
        rho16_max=0.30,
        rho8_max=0.35,
        rho4_max=0.45,
        rho2_max=0.25,
        beta_max=0.30,
        beta2_max=0.20,
        rho_init=0.05,
        beta_init=0.05,
        param_group_lrs=None,
        param_group_weight_decays=None,
        **kwargs,
    ):
        super().__init__()
        if in_channels is not None:
            n_channels = in_channels
        if num_classes is not None:
            n_classes = num_classes

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.return_aux = bool(return_aux)
        self.dino_layers = [int(x) for x in dino_layers]
        if len(self.dino_layers) != 4:
            raise ValueError(f"HLD_v3 需要 4 个 DINO 层，例如 [2,5,8,11]，当前: {self.dino_layers}")
        self.layer_t2, self.layer_t5, self.layer_t8, self.layer_t11 = self.dino_layers
        self.dino_patch_size = int(dino_patch_size)
        self.param_group_lrs = param_group_lrs or {}
        self.param_group_weight_decays = param_group_weight_decays or {}

        encoder = self._get_resnet34(pretrained=pretrained)
        self.input_adapter = nn.Conv2d(n_channels, 3, kernel_size=1, bias=False) if n_channels != 3 else nn.Identity()

        # HL_base encoder
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        # Frozen DINO branch
        self.dino = FrozenDINOv3TokenExtractor(
            dino_model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            out_layers=self.dino_layers,
            embed_dim=dino_embed_dim,
            patch_size=dino_patch_size,
            dino_normalize=dino_normalize,
            dino_intermediate_norm=dino_intermediate_norm,
        )

        # CNN -> DINO
        self.structure_prompt = StructurePrompt16(
            in_channels=256 + 512,
            out_channels=structure_prompt_dim,
            gn_groups=gn_groups,
        )
        self.adapter8 = HighLevelDINOAdapter(
            dino_dim=dino_embed_dim,
            prompt_dim=structure_prompt_dim,
            adapter_dim=adapter_dim,
            alpha_max=adapter_alpha_max,
            alpha_init=adapter_alpha_init,
            gn_groups=gn_groups,
        )
        self.adapter11 = HighLevelDINOAdapter(
            dino_dim=dino_embed_dim,
            prompt_dim=structure_prompt_dim,
            adapter_dim=adapter_dim,
            alpha_max=adapter_alpha_max,
            alpha_init=adapter_alpha_init,
            gn_groups=gn_groups,
        )

        # DINO semantic/topology prior
        self.dino_pyramid = DINOSemanticPyramid(dino_dim=dino_embed_dim, gn_groups=gn_groups)
        self.road_prior_head = RoadPriorHead(in_channels=64, hidden=64, out_channels=n_classes, gn_groups=gn_groups)
        self.skeleton_prior_head = SkeletonPriorHead(c16=256, c8=128, c4=64, mid=64, out_channels=1, gn_groups=gn_groups)

        # DINO -> CNN guided skip
        self.guide_x4 = GuidedSkipRefiner(512, 512, rho_max=rho32_max, beta_max=beta_max, rho_init=rho_init, beta_init=beta_init, gn_groups=gn_groups)
        self.guide_x3 = GuidedSkipRefiner(256, 256, rho_max=rho16_max, beta_max=beta_max, rho_init=rho_init, beta_init=beta_init, gn_groups=gn_groups)
        self.guide_x2 = GuidedSkipRefiner(128, 128, rho_max=rho8_max, beta_max=beta_max, rho_init=rho_init, beta_init=beta_init, gn_groups=gn_groups)
        self.guide_x1 = GuidedSkipRefiner(64, 64, rho_max=rho4_max, beta_max=beta_max, rho_init=rho_init, beta_init=beta_init, gn_groups=gn_groups)
        self.guide_x0 = GuidedSkipRefiner(64, 64, rho_max=rho2_max, beta_max=beta2_max, rho_init=rho_init, beta_init=beta_init, gn_groups=gn_groups)

        # HL_base decoder
        self.dec3 = ResidualDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=64, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=64, out_channels=64)
        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

        self._print_summary()

    @staticmethod
    def _get_resnet34(pretrained=True):
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

    def _print_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dino_total = sum(p.numel() for p in self.dino.parameters())
        dino_trainable = sum(p.numel() for p in self.dino.parameters() if p.requires_grad)
        print("--------------------------------------------------")
        print("🔧 HLD_v3 DINO Semantic-Topology Mutual Guidance Network")
        print(f"    - Total params:     {total / 1e6:.2f} M")
        print(f"    - Trainable params: {trainable / 1e6:.2f} M")
        print(f"    - DINO total:       {dino_total / 1e6:.2f} M")
        print(f"    - DINO trainable:   {dino_trainable / 1e6:.2f} M")
        print("    - HL_base encoder/decoder retained; DINO guides every skip via semantic-topology prior.")
        print("--------------------------------------------------")

    def train(self, mode: bool = True):
        super().train(mode)
        # DINO backbone 永远 eval，外部 adapter/readout/head 正常 train/eval。
        if hasattr(self, "dino"):
            self.dino.eval()
        return self

    @staticmethod
    def tokens_to_map(tokens, patch_hw: Tuple[int, int]):
        if tokens.dim() == 4:
            return tokens
        if tokens.dim() != 3:
            raise RuntimeError(f"token shape 异常: {tokens.shape}")
        b, n, c = tokens.shape
        ph, pw = patch_hw
        if n != ph * pw:
            raise RuntimeError(f"token 数与 patch_hw 不匹配: n={n}, patch_hw={patch_hw}")
        return tokens.transpose(1, 2).reshape(b, c, ph, pw).contiguous()

    def forward_features(self, x):
        input_size = x.shape[-2:]
        x_in = self.input_adapter(x)

        # 1) HL_base CNN encoder
        x0 = self.stem(x_in)                    # H/2,  64
        x1 = self.layer1(self.maxpool(x0))      # H/4,  64
        x2 = self.layer2(x1)                    # H/8,  128
        x3 = self.layer3(x2)                    # H/16, 256
        x4 = self.layer4(x3)                    # H/32, 512

        # 2) Frozen DINO token extraction
        token_dict = self.dino(x_in)
        patch_hw = (x_in.shape[-2] // self.dino_patch_size, x_in.shape[-1] // self.dino_patch_size)
        t2 = self.tokens_to_map(token_dict[self.layer_t2], patch_hw)
        t5 = self.tokens_to_map(token_dict[self.layer_t5], patch_hw)
        t8 = self.tokens_to_map(token_dict[self.layer_t8], patch_hw)
        t11 = self.tokens_to_map(token_dict[self.layer_t11], patch_hw)

        # 3) CNN -> DINO structure adaptation
        sp16 = self.structure_prompt(x3, x4)
        t8_adapt = self.adapter8(t8, sp16)
        t11_adapt = self.adapter11(t11, sp16)

        # 4) DINO semantic pyramid
        d32, d16, d8, d4, d2 = self.dino_pyramid(
            t2, t5, t8_adapt, t11_adapt,
            size32=x4.shape[-2:], size16=x3.shape[-2:], size8=x2.shape[-2:], size4=x1.shape[-2:], size2=x0.shape[-2:],
        )

        # 5) DINO road prior & skeleton prior
        road_prior_logits = self.road_prior_head(d4)
        skeleton_logits = self.skeleton_prior_head(d16, d8, d4)
        k4_prob = torch.sigmoid(skeleton_logits)

        k32 = F.interpolate(k4_prob, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        k16 = F.interpolate(k4_prob, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        k8 = F.interpolate(k4_prob, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        k2 = F.interpolate(k4_prob, size=x0.shape[-2:], mode="bilinear", align_corners=False)

        # 6) DINO-guided skip refinement
        s4, aux4 = self.guide_x4(x4, d32, k32, return_aux=True)
        s3, aux3 = self.guide_x3(x3, d16, k16, return_aux=True)
        s2, aux2 = self.guide_x2(x2, d8, k8, return_aux=True)
        s1, aux1 = self.guide_x1(x1, d4, k4_prob, return_aux=True)
        s0, aux0 = self.guide_x0(x0, d2, k2, return_aux=True)

        # 7) HL_base-style decoder
        d3 = self.dec3(s4, s3)                  # H/16, 256
        d2_dec = self.dec2(d3, s2)              # H/8,  128
        d1 = self.dec1(d2_dec, s1)              # H/4,   96
        d0 = self.dec0(d1, s0)                  # H/2,   64

        logits_half = self.out_head(d0)         # H/2, n_classes
        logits = F.interpolate(logits_half, size=input_size, mode="bilinear", align_corners=False)

        aux = {
            "final_logits": logits,
            "logits": logits,
            "logits_half": logits_half,
            "road_prior_logits": road_prior_logits,
            "skeleton_logits": skeleton_logits,
            "road_prior_prob": torch.sigmoid(road_prior_logits),
            "skeleton_prob": k4_prob,
            "x0": x0, "x1": x1, "x2": x2, "x3": x3, "x4": x4,
            "s0": s0, "s1": s1, "s2": s2, "s3": s3, "s4": s4,
            "D32": d32, "D16": d16, "D8": d8, "D4": d4, "D2": d2,
            "SP16": sp16,
            "d3": d3, "d2": d2_dec, "d1": d1, "d0": d0,
            "guide_x4": aux4, "guide_x3": aux3, "guide_x2": aux2, "guide_x1": aux1, "guide_x0": aux0,
        }
        return logits, aux

    def forward(self, x):
        logits, aux = self.forward_features(x)
        if self.return_aux:
            return aux
        return logits

    def forward_train(self, x):
        logits, aux = self.forward_features(x)
        return aux

    def _named_params_from_modules(self, modules: Sequence[nn.Module]):
        for module in modules:
            for p in module.parameters():
                yield p

    def get_param_groups(self, base_lr=1e-4, weight_decay=1e-2):
        lr_cfg = self.param_group_lrs or {}
        wd_cfg = self.param_group_weight_decays or {}
        base_lr = float(lr_cfg.get("base_lr", base_lr))
        base_wd = float(wd_cfg.get("base_wd", weight_decay))

        specs = [
            (
                "road_branch",
                [self.input_adapter, self.stem, self.maxpool, self.layer1, self.layer2, self.layer3, self.layer4],
                float(lr_cfg.get("road_branch_lr", 4e-5)),
                float(wd_cfg.get("road_branch_wd", 0.01)),
            ),
            (
                "dino_adapter",
                [self.structure_prompt, self.adapter8, self.adapter11],
                float(lr_cfg.get("dino_adapter_lr", 1e-4)),
                float(wd_cfg.get("dino_adapter_wd", 0.02)),
            ),
            (
                "dino_pyramid",
                [self.dino_pyramid],
                float(lr_cfg.get("dino_pyramid_lr", 1e-4)),
                float(wd_cfg.get("dino_pyramid_wd", 0.02)),
            ),
            (
                "dino_prior_heads",
                [self.road_prior_head, self.skeleton_prior_head],
                float(lr_cfg.get("dino_prior_lr", 1e-4)),
                float(wd_cfg.get("dino_prior_wd", 0.01)),
            ),
            (
                "guided_skip",
                [self.guide_x4, self.guide_x3, self.guide_x2, self.guide_x1, self.guide_x0],
                float(lr_cfg.get("guided_skip_lr", 1e-4)),
                float(wd_cfg.get("guided_skip_wd", 0.01)),
            ),
            (
                "decoder",
                [self.dec3, self.dec2, self.dec1, self.dec0, self.out_head],
                float(lr_cfg.get("decoder_lr", 1e-4)),
                float(wd_cfg.get("decoder_wd", 0.01)),
            ),
        ]

        used = set()
        groups = []
        for name, modules, lr, wd in specs:
            params = []
            for p in self._named_params_from_modules(modules):
                if p.requires_grad and id(p) not in used:
                    params.append(p)
                    used.add(id(p))
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": wd, "name": name})

        missing = []
        for p in self.parameters():
            if p.requires_grad and id(p) not in used:
                missing.append(p)
                used.add(id(p))
        if missing:
            groups.append({"params": missing, "lr": base_lr, "weight_decay": base_wd, "name": "others"})
        return groups


if __name__ == "__main__":
    # 只做无 DINO 实例化测试会缺少本地 dinov3，因此这里保留最小提示。
    print("HLD_v3 module loaded. Instantiate inside your project with valid dino_repo_path / dino_ckpt_path.")
