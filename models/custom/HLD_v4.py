import os
import sys
import math
import importlib
from typing import Dict, Tuple, Optional, Sequence, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["HLD_v4"]


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
            "[HLD_v4] Frozen DINOv3 token extractor ready. "
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
                f"[HLD_v4] 加载 DINOv3 权重完成: {self.dino_ckpt_path}\n"
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
    """从 DINO T2/T5/T8'/T11' 生成 D32/D16/D8/D4。

    HLD_v4 不再生成 D2。原因是 DINO patch=16 的原始 token map 是 H/16，
    继续上采样到 H/2 后再指导 x0，容易把粗语义注入低层边界，造成背景线状误检。
    """
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

    def forward(self, t2, t5, t8, t11, size32, size16, size8, size4):
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
        return d32, d16, d8, d4


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



class RoadConstrainedGuidedSkipRefiner(nn.Module):
    """Road-Constrained Guided Skip Refiner（道路约束引导跳跃细化器）。

    只用于 x4/x3/x2 中高层。
    与 HLD_v3 的主要区别：road prior 显式进入 suppression/injection，
    skeleton 只能在 road prior 允许的区域辅助拓扑恢复，避免全图线状结构被增强。
    """
    def __init__(
        self,
        x_channels: int,
        d_channels: int,
        rho_max: float,
        beta_max: float = 0.30,
        rho_init: float = 0.05,
        beta_init: float = 0.05,
        road_floor: float = 0.10,
        topo_floor: float = 0.50,
        gn_groups: int = 8,
        gate_bias: float = 1.5,
    ):
        super().__init__()
        self.road_floor = float(road_floor)
        self.topo_floor = float(topo_floor)
        self.d_proj = ConvGNAct(d_channels, x_channels, kernel_size=1, gn_groups=gn_groups)
        hidden = max(x_channels // 2, 16)
        self.semantic_gate = nn.Sequential(
            ConvGNAct(d_channels, hidden, kernel_size=3, gn_groups=gn_groups),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.semantic_gate[-2].bias, float(gate_bias))

        self.inject_gate = nn.Sequential(
            ConvGNAct(x_channels * 2 + 2, hidden, kernel_size=3, gn_groups=gn_groups),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.inject_gate[-2].bias)

        self.rho = BoundedScalar(max_value=rho_max, init_value=rho_init)
        self.beta = BoundedScalar(max_value=beta_max, init_value=beta_init)
        self.post = nn.Identity()

    def forward(self, x, d, road, skel, return_aux: bool = False):
        if d.shape[-2:] != x.shape[-2:]:
            d = F.interpolate(d, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if road.shape[-2:] != x.shape[-2:]:
            road = F.interpolate(road, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if skel.shape[-2:] != x.shape[-2:]:
            skel = F.interpolate(skel, size=x.shape[-2:], mode="bilinear", align_corners=False)
        road = road.clamp(0.0, 1.0)
        skel = skel.clamp(0.0, 1.0)

        m = self.semantic_gate(d)
        rho = self.rho()
        suppress = 1.0 - rho * (1.0 - road) * (1.0 - m)
        suppress = suppress.clamp(1.0 - float(rho.detach().max()), 1.0)
        x_safe = x * suppress

        d_proj = self.d_proj(d)
        g = self.inject_gate(torch.cat([x_safe, d_proj, road, skel], dim=1))
        beta = self.beta()
        road_allow = self.road_floor + (1.0 - self.road_floor) * road
        topo_allow = self.topo_floor + (1.0 - self.topo_floor) * skel
        out = self.post(x_safe + beta * g * road_allow * topo_allow * d_proj)

        if return_aux:
            return out, {
                "semantic_gate": m,
                "inject_gate": g,
                "suppress": suppress,
                "road_allow": road_allow,
                "topo_allow": topo_allow,
                "rho": rho.detach(),
                "beta": beta.detach(),
            }
        return out


class LowLevelDetailRefiner(nn.Module):
    """低层细节保护模块。用于 x1/x0，不接收 DINO feature。

    低层只保留 CNN 的边界、纹理、细路定位能力；road prior 只做弱门控，
    防止 DINO 粗语义污染 H/4、H/2 细节层。
    """
    def __init__(self, channels: int, alpha_max: float = 0.30, alpha_init: float = 0.03):
        super().__init__()
        self.local = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3),
            ConvBNAct(channels, channels, kernel_size=3),
        )
        self.road_gate = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = BoundedScalar(max_value=alpha_max, init_value=alpha_init)

    def forward(self, x, road, return_aux: bool = False):
        if road.shape[-2:] != x.shape[-2:]:
            road = F.interpolate(road, size=x.shape[-2:], mode="bilinear", align_corners=False)
        road = road.clamp(0.0, 1.0)
        local = self.local(x)
        gate = self.road_gate(road)
        alpha = self.alpha()
        out = x + alpha * gate * local
        if return_aux:
            return out, {"road_gate": gate, "alpha": alpha.detach()}
        return out


class GapAwareTopologyRecoveryBlock(nn.Module):
    """Gap-aware Topology Recovery Block（断点感知拓扑恢复块）。

    只放在 dec2 后的 H/8 位置。skeleton 不再全尺度指导 skip，
    而是在 road prior 支持、skeleton 支持、模型不确定的位置做温和拓扑恢复。
    """
    def __init__(self, channels: int = 128, lambda_max: float = 0.30, lambda_init: float = 0.03):
        super().__init__()
        branch = max(channels // 2, 32)
        self.coarse_head = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.local = ConvBNAct(channels, channels, kernel_size=3)
        self.strip_h = ConvBNAct(channels, branch, kernel_size=(1, 7))
        self.strip_v = ConvBNAct(channels, branch, kernel_size=(7, 1))
        self.dilated = ConvBNAct(channels, branch, kernel_size=3, dilation=2)
        self.fuse = nn.Sequential(
            ConvBNAct(channels + branch * 3, channels, kernel_size=1),
            ConvBNAct(channels, channels, kernel_size=3),
        )
        self.lam = BoundedScalar(max_value=lambda_max, init_value=lambda_init)

    def forward(self, x, road, skel, return_aux: bool = False):
        if road.shape[-2:] != x.shape[-2:]:
            road = F.interpolate(road, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if skel.shape[-2:] != x.shape[-2:]:
            skel = F.interpolate(skel, size=x.shape[-2:], mode="bilinear", align_corners=False)
        road = road.clamp(0.0, 1.0)
        skel = skel.clamp(0.0, 1.0)

        coarse_logits = self.coarse_head(x)
        p = torch.sigmoid(coarse_logits)
        uncertainty = (4.0 * p * (1.0 - p)).clamp(0.0, 1.0)
        gap_gate = (road * skel * uncertainty).clamp(0.0, 1.0)

        topo_feat = self.fuse(torch.cat([
            self.local(x),
            self.strip_h(x),
            self.strip_v(x),
            self.dilated(x),
        ], dim=1))
        lam = self.lam()
        out = x + lam * gap_gate * topo_feat
        if return_aux:
            return out, {
                "coarse_logits": coarse_logits,
                "uncertainty": uncertainty,
                "gap_gate": gap_gate,
                "lambda": lam.detach(),
            }
        return out


class RoadBackgroundDecoupledHead(nn.Module):
    """道路-背景解耦输出头。

    body: 道路主体；detail: 细路/边界残差；suppress: 背景线状抑制残差。
    detail 只在 road prior 支持区域增强；suppress 主要在非道路区域工作。
    """
    def __init__(
        self,
        d0_channels: int = 64,
        s0_channels: int = 64,
        n_classes: int = 1,
        detail_max: float = 0.40,
        detail_init: float = 0.08,
        suppress_max: float = 0.40,
        suppress_init: float = 0.05,
    ):
        super().__init__()
        self.body_head = nn.Sequential(
            ConvBNAct(d0_channels, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )
        self.detail_head = nn.Sequential(
            ConvBNAct(d0_channels + s0_channels, 64, kernel_size=3),
            ConvBNAct(64, 32, kernel_size=3),
            nn.Conv2d(32, n_classes, kernel_size=1),
        )
        self.suppress_head = nn.Sequential(
            ConvBNAct(d0_channels + s0_channels + 1, 64, kernel_size=3),
            ConvBNAct(64, 32, kernel_size=3),
            nn.Conv2d(32, n_classes, kernel_size=1),
        )
        self.detail_w = BoundedScalar(max_value=detail_max, init_value=detail_init)
        self.suppress_w = BoundedScalar(max_value=suppress_max, init_value=suppress_init)

        # 让新增 residual 分支初始影响很小，避免一开始破坏 HL_base decoder 的稳定输出。
        nn.init.zeros_(self.detail_head[-1].weight)
        nn.init.zeros_(self.detail_head[-1].bias)
        nn.init.zeros_(self.suppress_head[-1].weight)
        nn.init.zeros_(self.suppress_head[-1].bias)

    def forward(self, d0, s0, road2, return_aux: bool = False):
        if s0.shape[-2:] != d0.shape[-2:]:
            s0 = F.interpolate(s0, size=d0.shape[-2:], mode="bilinear", align_corners=False)
        if road2.shape[-2:] != d0.shape[-2:]:
            road2 = F.interpolate(road2, size=d0.shape[-2:], mode="bilinear", align_corners=False)
        road2 = road2.clamp(0.0, 1.0)

        body = self.body_head(d0)
        detail = self.detail_head(torch.cat([d0, s0], dim=1))
        raw_suppress = self.suppress_head(torch.cat([d0, s0, road2], dim=1))
        # zero-centered softplus: final conv 为 0 时，初始 suppress_signal=0 且仍有梯度。
        suppress_signal = F.softplus(raw_suppress) - math.log(2.0)

        dw = self.detail_w()
        sw = self.suppress_w()
        logits_half = body + dw * road2 * detail - sw * (1.0 - road2) * suppress_signal

        if return_aux:
            return logits_half, {
                "body_logits": body,
                "detail_logits": detail,
                "suppress_logits": raw_suppress,
                "detail_weight": dw.detach(),
                "suppress_weight": sw.detach(),
            }
        return logits_half


class FullResolutionRefine(nn.Module):
    """轻量全分辨率残差细化头。

    修正 H/2 logits 上采样后的细路边界，但 residual 必须受 road prior 门控，
    防止把所有高频边缘都增强成道路。
    """
    def __init__(self, s0_channels: int = 64, n_classes: int = 1, eta_max: float = 0.30, eta_init: float = 0.03):
        super().__init__()
        self.reduce_s0 = ConvBNAct(s0_channels, 16, kernel_size=1)
        self.refine = nn.Sequential(
            ConvBNAct(n_classes + 16 + 1, 32, kernel_size=3),
            ConvBNAct(32, 16, kernel_size=3),
            nn.Conv2d(16, n_classes, kernel_size=1),
        )
        self.eta = BoundedScalar(max_value=eta_max, init_value=eta_init)
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, logits_half, s0, road_full, output_size, return_aux: bool = False):
        logits_up = F.interpolate(logits_half, size=output_size, mode="bilinear", align_corners=False)
        s0_red = self.reduce_s0(s0)
        s0_up = F.interpolate(s0_red, size=output_size, mode="bilinear", align_corners=False)
        if road_full.shape[-2:] != output_size:
            road_full = F.interpolate(road_full, size=output_size, mode="bilinear", align_corners=False)
        road_full = road_full.clamp(0.0, 1.0)
        residual = self.refine(torch.cat([logits_up, s0_up, road_full], dim=1))
        eta = self.eta()
        logits = logits_up + eta * road_full * residual
        if return_aux:
            return logits, {"logits_up": logits_up, "full_residual": residual, "eta": eta.detach()}
        return logits



class HLD_v4(nn.Module):
    """
    HLD_v4-RCGT: Road-Constrained Guided Topology Network.

    核心修改：
    1) DINO 只指导 x4/x3/x2 中高层，不再强行指导 x1/x0；
    2) road prior 显式约束 DINO injection 与 suppression；
    3) skeleton prior 不再全尺度增强，只在 H/8 的 GapAwareTopologyRecoveryBlock 中恢复断点；
    4) 输出头拆分 road body/detail/background suppression，并加轻量全分辨率残差细化。
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
        # 下面几个参数保留是为了兼容旧 HLD_v3 json；HLD_v4 不再使用 x1/x0 DINO guidance。
        rho4_max=0.45,
        rho2_max=0.25,
        beta_max=0.30,
        beta2_max=0.20,
        rho_init=0.05,
        beta_init=0.05,
        low_alpha_max=0.30,
        low_alpha_init=0.03,
        topo_lambda_max=0.30,
        topo_lambda_init=0.03,
        head_detail_max=0.40,
        head_detail_init=0.08,
        head_suppress_max=0.40,
        head_suppress_init=0.05,
        full_eta_max=0.30,
        full_eta_init=0.03,
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
            raise ValueError(f"HLD_v4 需要 4 个 DINO 层，例如 [2,5,8,11]，当前: {self.dino_layers}")
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

        # CNN -> DINO high-level structure adaptation
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

        # DINO semantic/topology priors
        self.dino_pyramid = DINOSemanticPyramid(dino_dim=dino_embed_dim, gn_groups=gn_groups)
        self.road_prior_head = RoadPriorHead(in_channels=64, hidden=64, out_channels=n_classes, gn_groups=gn_groups)
        self.skeleton_prior_head = SkeletonPriorHead(c16=256, c8=128, c4=64, mid=64, out_channels=1, gn_groups=gn_groups)

        # DINO -> CNN：只指导 x4/x3/x2；x1/x0 保留低层细节。
        self.guide_x4 = RoadConstrainedGuidedSkipRefiner(
            512, 512, rho_max=rho32_max, beta_max=beta_max,
            rho_init=rho_init, beta_init=beta_init, gn_groups=gn_groups,
        )
        self.guide_x3 = RoadConstrainedGuidedSkipRefiner(
            256, 256, rho_max=rho16_max, beta_max=beta_max,
            rho_init=rho_init, beta_init=beta_init, gn_groups=gn_groups,
        )
        self.guide_x2 = RoadConstrainedGuidedSkipRefiner(
            128, 128, rho_max=rho8_max, beta_max=beta_max,
            rho_init=rho_init, beta_init=beta_init, gn_groups=gn_groups,
        )
        self.low_x1 = LowLevelDetailRefiner(64, alpha_max=low_alpha_max, alpha_init=low_alpha_init)
        self.low_x0 = LowLevelDetailRefiner(64, alpha_max=low_alpha_max, alpha_init=low_alpha_init)

        # HL_base decoder + H/8 topology recovery
        self.dec3 = ResidualDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.topology_recovery = GapAwareTopologyRecoveryBlock(
            channels=128, lambda_max=topo_lambda_max, lambda_init=topo_lambda_init,
        )
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=64, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=64, out_channels=64)

        # Road-background decoupled output + full-resolution refinement
        self.out_head = RoadBackgroundDecoupledHead(
            d0_channels=64,
            s0_channels=64,
            n_classes=n_classes,
            detail_max=head_detail_max,
            detail_init=head_detail_init,
            suppress_max=head_suppress_max,
            suppress_init=head_suppress_init,
        )
        self.full_refine = FullResolutionRefine(
            s0_channels=64,
            n_classes=n_classes,
            eta_max=full_eta_max,
            eta_init=full_eta_init,
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
        print("🔧 HLD_v4-RCGT Road-Constrained Guided Topology Network")
        print(f"    - Total params:     {total / 1e6:.2f} M")
        print(f"    - Trainable params: {trainable / 1e6:.2f} M")
        print(f"    - DINO total:       {dino_total / 1e6:.2f} M")
        print(f"    - DINO trainable:   {dino_trainable / 1e6:.2f} M")
        print("    - DINO guides x4/x3/x2 only; x1/x0 keep CNN detail; road prior constrains injection; skeleton only recovers gaps at H/8.")
        print("--------------------------------------------------")

    def train(self, mode: bool = True):
        super().train(mode)
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

        # 4) DINO semantic pyramid: no D2 in HLD_v4
        d32, d16, d8, d4 = self.dino_pyramid(
            t2, t5, t8_adapt, t11_adapt,
            size32=x4.shape[-2:], size16=x3.shape[-2:], size8=x2.shape[-2:], size4=x1.shape[-2:],
        )

        # 5) DINO road prior & skeleton prior
        road_prior_logits = self.road_prior_head(d4)
        skeleton_logits = self.skeleton_prior_head(d16, d8, d4)
        road4_prob = torch.sigmoid(road_prior_logits)
        skel4_prob = torch.sigmoid(skeleton_logits)

        road32 = F.interpolate(road4_prob, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        road16 = F.interpolate(road4_prob, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        road8 = F.interpolate(road4_prob, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        road2 = F.interpolate(road4_prob, size=x0.shape[-2:], mode="bilinear", align_corners=False)

        skel32 = F.interpolate(skel4_prob, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        skel16 = F.interpolate(skel4_prob, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        skel8 = F.interpolate(skel4_prob, size=x2.shape[-2:], mode="bilinear", align_corners=False)

        # 6) Selective road-constrained skip refinement
        s4, aux4 = self.guide_x4(x4, d32, road32, skel32, return_aux=True)
        s3, aux3 = self.guide_x3(x3, d16, road16, skel16, return_aux=True)
        s2, aux2 = self.guide_x2(x2, d8, road8, skel8, return_aux=True)
        s1, aux1 = self.low_x1(x1, road4_prob, return_aux=True)
        s0, aux0 = self.low_x0(x0, road2, return_aux=True)

        # 7) Decoder with gap-aware topology recovery at H/8
        d3 = self.dec3(s4, s3)                  # H/16, 256
        d2_dec = self.dec2(d3, s2)              # H/8,  128
        d2_topo, aux_topo = self.topology_recovery(d2_dec, road8, skel8, return_aux=True)
        d1 = self.dec1(d2_topo, s1)             # H/4,   96
        d0 = self.dec0(d1, s0)                  # H/2,   64

        # 8) Road-background decoupled head + full-resolution refinement
        logits_half, aux_head = self.out_head(d0, s0, road2, return_aux=True)
        road_full = F.interpolate(road4_prob, size=input_size, mode="bilinear", align_corners=False)
        logits, aux_full = self.full_refine(logits_half, s0, road_full, output_size=input_size, return_aux=True)

        aux = {
            "final_logits": logits,
            "logits": logits,
            "logits_half": logits_half,
            "road_prior_logits": road_prior_logits,
            "skeleton_logits": skeleton_logits,
            "road_prior_prob": road4_prob,
            "skeleton_prob": skel4_prob,
            "x0": x0, "x1": x1, "x2": x2, "x3": x3, "x4": x4,
            "s0": s0, "s1": s1, "s2": s2, "s3": s3, "s4": s4,
            "D32": d32, "D16": d16, "D8": d8, "D4": d4,
            "SP16": sp16,
            "d3": d3, "d2": d2_topo, "d2_before_topo": d2_dec, "d1": d1, "d0": d0,
            "guide_x4": aux4, "guide_x3": aux3, "guide_x2": aux2,
            "low_x1": aux1, "low_x0": aux0,
            "topology_recovery": aux_topo,
            "decoupled_head": aux_head,
            "full_refine": aux_full,
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
                [self.guide_x4, self.guide_x3, self.guide_x2],
                float(lr_cfg.get("guided_skip_lr", 1e-4)),
                float(wd_cfg.get("guided_skip_wd", 0.01)),
            ),
            (
                "low_detail",
                [self.low_x1, self.low_x0],
                float(lr_cfg.get("detail_lr", lr_cfg.get("guided_skip_lr", 1e-4))),
                float(wd_cfg.get("detail_wd", 0.01)),
            ),
            (
                "topology_recovery",
                [self.topology_recovery],
                float(lr_cfg.get("topology_lr", 1e-4)),
                float(wd_cfg.get("topology_wd", 0.01)),
            ),
            (
                "decoder",
                [self.dec3, self.dec2, self.dec1, self.dec0],
                float(lr_cfg.get("decoder_lr", 1e-4)),
                float(wd_cfg.get("decoder_wd", 0.01)),
            ),
            (
                "head",
                [self.out_head, self.full_refine],
                float(lr_cfg.get("head_lr", lr_cfg.get("decoder_lr", 1e-4))),
                float(wd_cfg.get("head_wd", 0.01)),
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
    print("HLD_v4 module loaded. Instantiate inside your project with valid dino_repo_path / dino_ckpt_path.")
