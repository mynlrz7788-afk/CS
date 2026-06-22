import os
import sys
import math
import importlib
from typing import Dict, Tuple, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["HLD_v1_G"]


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


class ConvBNAct(nn.Module):
    """Conv-BN-ReLU，保持你现有 HL_base / DC 系列的主干和 decoder 风格。"""
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
    """Conv-GN-GELU，用在 DINO / Prompt / Adapter 分支，适合小 batch。"""
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
    """Efficient Channel Attention（高效通道注意力），保持 HL_base 风格。"""
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
    """轻量残差细化块，用于 DINO guidance / prompt feature。"""
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
    """HL_base 同款残差解码块。"""
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

    输出 token sequence：
        {layer_id: B × N × C}
    其中 N = H/patch_size × W/patch_size。
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
            "[HLD_v1_G] Frozen DINOv3 token extractor ready. "
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

        if self.dino_ckpt_path is not None and self.dino_ckpt_path != "":
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
                f"[HLD_v1_G] 加载 DINOv3 权重完成: {self.dino_ckpt_path}\n"
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


class HighResolutionRoadBranch(nn.Module):
    """ResNet34-BS4 高分辨率道路分支，只到 layer3，不注册 layer4。"""
    def __init__(self, n_channels=3, pretrained=True):
        super().__init__()
        encoder = self._get_resnet34(pretrained=pretrained)
        self.input_adapter = nn.Conv2d(n_channels, 3, kernel_size=1, bias=False) if n_channels != 3 else nn.Identity()
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3

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

    def forward(self, x):
        x = self.input_adapter(x)
        x0 = self.stem(x)                    # B, 64,  H/2
        x1 = self.layer1(self.maxpool(x0))   # B, 64,  H/4
        x2 = self.layer2(x1)                 # B, 128, H/8
        x3 = self.layer3(x2)                 # B, 256, H/16
        return x0, x1, x2, x3, x


class DynamicSnakeConv2d(nn.Module):
    """
    可微分 Dynamic Snake Convolution 近似实现。

    orientation='x': 沿 x 方向形成蛇形采样线，学习 y 偏移；
    orientation='y': 沿 y 方向形成蛇形采样线，学习 x 偏移。

    这里不依赖外部 CUDA op，使用 grid_sample 实现，适合先在当前工程里跑通。
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        orientation: str = "x",
        max_offset: float = 3.0,
        gn_groups: int = 8,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("DynamicSnakeConv2d 的 kernel_size 必须是奇数。")
        if orientation not in ("x", "y"):
            raise ValueError("orientation 必须是 'x' 或 'y'。")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.orientation = orientation
        self.max_offset = float(max_offset)
        self.center = kernel_size // 2

        self.offset_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            _make_gn(channels, gn_groups),
            nn.GELU(),
            nn.Conv2d(channels, kernel_size, kernel_size=3, padding=1, bias=True),
        )
        self.weight = nn.Parameter(torch.empty(channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.post = nn.Sequential(
            _make_gn(channels, gn_groups),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _make_gn(channels, gn_groups),
            nn.GELU(),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / float(self.kernel_size))
        nn.init.zeros_(self.offset_conv[-1].weight)
        nn.init.zeros_(self.offset_conv[-1].bias)

    @staticmethod
    def _base_grid(batch_size, height, width, device, dtype):
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def forward(self, x):
        b, c, h, w = x.shape
        if c != self.channels:
            raise RuntimeError(f"DynamicSnakeConv2d 通道不匹配: 期望 {self.channels}, 实际 {c}")

        offsets = torch.tanh(self.offset_conv(x)) * self.max_offset  # B, K, H, W, pixel unit
        base = self._base_grid(b, h, w, x.device, x.dtype)
        sampled_list = []

        # 将像素单位位移转成 [-1, 1] 坐标位移。
        sx = 2.0 / max(float(w - 1), 1.0)
        sy = 2.0 / max(float(h - 1), 1.0)

        for i in range(self.kernel_size):
            pos = float(i - self.center)
            off = offsets[:, i]  # B,H,W
            grid = base.clone()
            if self.orientation == "x":
                grid[..., 0] = grid[..., 0] + pos * sx
                grid[..., 1] = grid[..., 1] + off * sy
            else:
                grid[..., 0] = grid[..., 0] + off * sx
                grid[..., 1] = grid[..., 1] + pos * sy

            sampled = F.grid_sample(
                x,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            sampled_list.append(sampled)

        sampled = torch.stack(sampled_list, dim=2)  # B,C,K,H,W
        weight = self.weight.view(1, c, self.kernel_size, 1, 1)
        out = (sampled * weight).sum(dim=2) + self.bias.view(1, c, 1, 1)
        return self.post(out)


class GroupedDilatedConv(nn.Module):
    """DSGD-style 分组多尺度空洞卷积。"""
    def __init__(self, channels, dilations=(1, 3, 5, 7), gn_groups=8):
        super().__init__()
        self.channels = int(channels)
        self.dilations = tuple(int(d) for d in dilations)
        self.num_groups = len(self.dilations)
        if channels % self.num_groups != 0:
            raise ValueError(f"channels={channels} 必须能被 dilation 分组数 {self.num_groups} 整除。")
        group_ch = channels // self.num_groups
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    group_ch,
                    group_ch,
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                    groups=group_ch,
                    bias=False,
                ),
                _make_gn(group_ch, gn_groups),
                nn.GELU(),
            )
            for d in self.dilations
        ])
        self.fuse = ConvGNAct(channels, channels, kernel_size=1, padding=0, gn_groups=gn_groups)

    def forward(self, x):
        chunks = torch.chunk(x, self.num_groups, dim=1)
        ys = [branch(chunk) for branch, chunk in zip(self.branches, chunks)]
        return self.fuse(torch.cat(ys, dim=1))


class RoadDSGDBlock(nn.Module):
    """Road-DSGD: DSConv-X/Y + grouped dilated convolution + residual refinement。"""
    def __init__(
        self,
        channels,
        kernel_size=7,
        dilations=(1, 3, 5, 7),
        max_offset=3.0,
        gn_groups=8,
    ):
        super().__init__()
        self.snake_x = DynamicSnakeConv2d(channels, kernel_size=kernel_size, orientation="x", max_offset=max_offset, gn_groups=gn_groups)
        self.snake_y = DynamicSnakeConv2d(channels, kernel_size=kernel_size, orientation="y", max_offset=max_offset, gn_groups=gn_groups)
        self.snake_fuse = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.gdc = GroupedDilatedConv(channels, dilations=dilations, gn_groups=gn_groups)
        self.out = nn.Sequential(
            ConvGNAct(channels, channels, kernel_size=3, gn_groups=gn_groups),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _make_gn(channels, gn_groups),
        )
        self.act = nn.GELU()

    def forward(self, x):
        s = self.snake_fuse(torch.cat([self.snake_x(x), self.snake_y(x)], dim=1))
        y = self.gdc(s)
        y = self.out(y)
        return self.act(x + y)


class DirectionalStripBlock(nn.Module):
    """多方向条带卷积，用于道路方向连通和高分辨率细节传播。"""
    def __init__(self, channels, kernel_size=15, gn_groups=8):
        super().__init__()
        pad = kernel_size // 2
        self.h = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, pad), groups=channels, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0), groups=channels, bias=False)
        self.d = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
        self.l = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            _make_gn(channels, gn_groups),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _make_gn(channels, gn_groups),
        )
        self.act = nn.GELU()

    def forward(self, x):
        y = torch.cat([self.h(x), self.v(x), self.d(x), self.l(x)], dim=1)
        y = self.fuse(y)
        return self.act(x + y)


class RoadDSGDPromptExtractorG(nn.Module):
    """
    DINO-safe road prompt extractor for HLD_v1_G.

    改动点：
    1) P16 保留 Road-DSGD / DSConv，用于和 DINO token 对齐的语义道路提示；
    2) P8 去掉 DSConv，只保留普通卷积 + Directional Strip，降低中高分辨率纹理过拟合；
    3) P4 只做轻量 boundary/detail prompt，后续会被 G4 语义门控。
    """
    def __init__(
        self,
        c4=96,
        c8=128,
        c16=128,
        gn_groups=8,
        snake_kernel=7,
        snake_offset=3.0,
    ):
        super().__init__()
        # H/16：保留最强 Road-DSGD，作为 DINO token adaptation 的稳定结构提示。
        self.p16_reduce = ConvGNAct(256, c16, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.p16_dsgd = RoadDSGDBlock(
            c16,
            kernel_size=snake_kernel,
            max_offset=snake_offset,
            gn_groups=gn_groups,
        )

        # H/8：去掉 DSConv，避免中尺度纹理/背景细线过拟合。
        self.p8_reduce = ConvGNAct(128, c8, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.p8_context = nn.Sequential(
            ConvGNAct(c8, c8, kernel_size=3, gn_groups=gn_groups),
            DirectionalStripBlock(c8, kernel_size=15, gn_groups=gn_groups),
            ResidualRefineGN(c8, gn_groups=gn_groups),
        )

        # H/4：轻量细节提示，后续必须经过 G4 gate 才能进入 decoder。
        self.p4_reduce = ConvGNAct(64, c4, kernel_size=3, gn_groups=gn_groups)
        self.p4_detail = nn.Sequential(
            DirectionalStripBlock(c4, kernel_size=7, gn_groups=gn_groups),
            ResidualRefineGN(c4, gn_groups=gn_groups),
        )

    def forward(self, x1, x2, x3):
        p16 = self.p16_dsgd(self.p16_reduce(x3))
        p8 = self.p8_context(self.p8_reduce(x2))
        p4 = self.p4_detail(self.p4_reduce(x1))
        return p4, p8, p16


def _window_partition(x: torch.Tensor, window_size: int):
    """x: B,H,W,C -> windows: B*nW, window_size*window_size, C"""
    b, h, w, c = x.shape
    if h % window_size != 0 or w % window_size != 0:
        raise RuntimeError(f"window_size={window_size} 不能整除 feature size={h}x{w}")
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, c)
    return windows


def _window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int, batch_size: int):
    """windows: B*nW, Ws*Ws, C -> B,H,W,C"""
    c = windows.shape[-1]
    x = windows.view(batch_size, h // window_size, w // window_size, window_size, window_size, c)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(batch_size, h, w, c)
    return x


class LocalWindowAssociation(nn.Module):
    """局部窗口高低关联：DINO token window attends to road prompt window。"""
    def __init__(self, token_dim=384, prompt_dim=128, inner_dim=128, heads=4, window_size=8, dropout=0.0):
        super().__init__()
        if inner_dim % heads != 0:
            raise ValueError(f"inner_dim={inner_dim} 必须能被 heads={heads} 整除。")
        self.window_size = int(window_size)
        self.q = nn.Linear(token_dim, inner_dim, bias=False)
        self.k = nn.Linear(prompt_dim, inner_dim, bias=False)
        self.v = nn.Linear(prompt_dim, inner_dim, bias=False)
        self.q_norm = nn.LayerNorm(inner_dim)
        self.k_norm = nn.LayerNorm(inner_dim)
        self.attn = nn.MultiheadAttention(inner_dim, heads, dropout=dropout, batch_first=True)

    def forward(self, tokens, prompt, patch_hw: Tuple[int, int]):
        b, n, _ = tokens.shape
        h, w = patch_hw
        if n != h * w:
            raise RuntimeError(f"token 数和 patch_hw 不匹配: N={n}, patch_hw={patch_hw}")
        p = prompt.flatten(2).transpose(1, 2).contiguous()
        q = self.q(tokens).view(b, h, w, -1)
        k = self.k(p).view(b, h, w, -1)
        v = self.v(p).view(b, h, w, -1)

        q_win = _window_partition(q, self.window_size)
        k_win = _window_partition(k, self.window_size)
        v_win = _window_partition(v, self.window_size)

        out, _ = self.attn(self.q_norm(q_win), self.k_norm(k_win), v_win, need_weights=False)
        out = _window_reverse(out, self.window_size, h, w, b)
        return out.view(b, n, -1).contiguous()


class RoadPrototypeAssociation(nn.Module):
    """全局道路原型关联：用少量 learned prototypes 替代 4096×4096 全注意力。"""
    def __init__(self, token_dim=384, prompt_dim=128, inner_dim=128, heads=4, num_prototypes=32, dropout=0.0):
        super().__init__()
        self.num_prototypes = int(num_prototypes)
        self.prompt_proj = nn.Linear(prompt_dim, inner_dim, bias=False)
        self.token_q = nn.Linear(token_dim, inner_dim, bias=False)
        self.prototype_queries = nn.Parameter(torch.zeros(1, num_prototypes, inner_dim))
        self.pool_attn = nn.MultiheadAttention(inner_dim, heads, dropout=dropout, batch_first=True)
        self.token_attn = nn.MultiheadAttention(inner_dim, heads, dropout=dropout, batch_first=True)
        self.p_norm = nn.LayerNorm(inner_dim)
        self.t_norm = nn.LayerNorm(inner_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.prototype_queries, std=0.02)

    def forward(self, tokens, prompt):
        b = tokens.shape[0]
        p = prompt.flatten(2).transpose(1, 2).contiguous()
        p = self.prompt_proj(p)
        q_proto = self.prototype_queries.expand(b, -1, -1)
        prototypes, _ = self.pool_attn(q_proto, self.p_norm(p), p, need_weights=False)
        q_token = self.token_q(tokens)
        out, _ = self.token_attn(self.t_norm(q_token), self.p_norm(prototypes), prototypes, need_weights=False)
        return out


class DirectionalStripPropagation(nn.Module):
    """在 DINO patch grid 上做方向条带传播，补充跨窗口道路方向连续性。"""
    def __init__(self, dim=128, kernel_size=11, gn_groups=8):
        super().__init__()
        self.strip = DirectionalStripBlock(dim, kernel_size=kernel_size, gn_groups=gn_groups)

    def forward(self, delta_tokens, patch_hw: Tuple[int, int]):
        b, n, c = delta_tokens.shape
        h, w = patch_hw
        if n != h * w:
            raise RuntimeError(f"DirectionalStripPropagation token 数不匹配: N={n}, patch_hw={patch_hw}")
        x = delta_tokens.transpose(1, 2).contiguous().view(b, c, h, w)
        y = self.strip(x)
        return y.flatten(2).transpose(1, 2).contiguous()


class WPDAAdapterG(nn.Module):
    """
    DINO-safe Window-Prototype Directional Association Adapter.

    与 HLD_v1 的区别：
    1) 加入 DINO confidence gate：由 DINO token 自己决定是否接受 road prompt 的更新；
    2) alpha bounded：限制 prompt 对 DINO token 的最大扰动，避免高分辨率提示污染 foundation feature；
    3) 默认 prototype 数量建议从 32 降到 16。
    """
    def __init__(
        self,
        token_dim=384,
        prompt_dim=128,
        inner_dim=128,
        heads=4,
        window_size=8,
        num_prototypes=16,
        alpha_init=0.01,
        alpha_max=0.15,
        dropout=0.05,
        gn_groups=8,
    ):
        super().__init__()
        self.local = LocalWindowAssociation(token_dim, prompt_dim, inner_dim, heads, window_size, dropout)
        self.proto = RoadPrototypeAssociation(token_dim, prompt_dim, inner_dim, heads, num_prototypes, dropout)
        self.dir = DirectionalStripPropagation(inner_dim, kernel_size=11, gn_groups=gn_groups)
        self.fuse = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, inner_dim * 2, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim * 2, token_dim, bias=False),
        )
        self.conf_gate = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, max(token_dim // 4, 64), bias=True),
            nn.GELU(),
            nn.Linear(max(token_dim // 4, 64), 1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha_max = float(alpha_max)
        init = float(alpha_init)
        init = min(max(init, 1e-6), self.alpha_max - 1e-6)
        raw = math.log(init / (self.alpha_max - init))
        self.alpha_raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        self._init_weights()

    def _init_weights(self):
        # 初始不破坏 Frozen DINO 表征，训练后逐渐学习安全增量。
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.conf_gate[-2].weight)
        nn.init.zeros_(self.conf_gate[-2].bias)

    def forward(self, tokens, prompt, patch_hw: Tuple[int, int]):
        d_local = self.local(tokens, prompt, patch_hw)
        d_proto = self.proto(tokens, prompt)
        d_dir = self.dir(d_local + d_proto, patch_hw)
        delta = self.fuse(d_local + d_proto + d_dir)
        conf = self.conf_gate(tokens)  # B,N,1，控制权来自 DINO token 本身。
        alpha = self.alpha_max * torch.sigmoid(self.alpha_raw)
        return tokens + alpha * conf * delta


class PromptFuse(nn.Module):
    """将 DINO guidance 与高分辨率 prompt 逐级融合，避免假高分辨率。"""
    def __init__(self, in_g, in_p, out_ch, gn_groups=8):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvGNAct(in_g + in_p, out_ch, kernel_size=1, padding=0, gn_groups=gn_groups),
            ResidualRefineGN(out_ch, gn_groups=gn_groups),
        )

    def forward(self, g, p):
        if g.shape[-2:] != p.shape[-2:]:
            g = F.interpolate(g, size=p.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([g, p], dim=1))


class HLDReadout(nn.Module):
    """T2/T5'/T8'/T11' -> G16/G8/G4。"""
    def __init__(
        self,
        embed_dim=384,
        c16=256,
        c8=128,
        c4=96,
        p8_channels=128,
        p4_channels=96,
        gn_groups=8,
    ):
        super().__init__()
        self.w16 = nn.Parameter(torch.tensor([0.45, 0.55], dtype=torch.float32))
        self.w8 = nn.Parameter(torch.tensor([0.45, 0.55], dtype=torch.float32))
        self.w4 = nn.Parameter(torch.tensor([0.55, 0.45], dtype=torch.float32))

        self.g16_proj = nn.Sequential(
            ConvGNAct(embed_dim, c16, kernel_size=1, padding=0, gn_groups=gn_groups),
            ResidualRefineGN(c16, gn_groups=gn_groups),
        )
        self.g8_proj = ConvGNAct(embed_dim, c8, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.g8_fuse = PromptFuse(c8, p8_channels, c8, gn_groups=gn_groups)

        self.g4_proj = ConvGNAct(embed_dim, c4, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.p8_to_c4 = ConvGNAct(p8_channels, c4, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.g4_fuse8 = PromptFuse(c4, c4, c4, gn_groups=gn_groups)
        self.g4_fuse4 = PromptFuse(c4, p4_channels, c4, gn_groups=gn_groups)

    @staticmethod
    def _tokens_to_grid(tokens, patch_hw: Tuple[int, int]):
        b, n, c = tokens.shape
        h, w = patch_hw
        if n != h * w:
            raise RuntimeError(f"token 数和 patch_hw 不匹配: N={n}, patch_hw={patch_hw}")
        return tokens.transpose(1, 2).contiguous().view(b, c, h, w)

    def _weighted_sum(self, a, b, weights):
        w = torch.softmax(weights, dim=0)
        return w[0] * a + w[1] * b

    def forward(self, token_dict: Dict[int, torch.Tensor], p8, p4, patch_hw: Tuple[int, int]):
        t2 = self._tokens_to_grid(token_dict[2], patch_hw)
        t5 = self._tokens_to_grid(token_dict[5], patch_hw)
        t8 = self._tokens_to_grid(token_dict[8], patch_hw)
        t11 = self._tokens_to_grid(token_dict[11], patch_hw)

        g16 = self.g16_proj(self._weighted_sum(t8, t11, self.w16))

        g8_raw = self.g8_proj(self._weighted_sum(t5, t8, self.w8))
        g8 = self.g8_fuse(g8_raw, p8)

        g4_raw = self.g4_proj(self._weighted_sum(t2, t5, self.w4))
        p8_c4 = self.p8_to_c4(p8)
        g4_mid = self.g4_fuse8(g4_raw, p8_c4)
        g4 = self.g4_fuse4(g4_mid, p4)

        return g16, g8, g4


class TripleFeatureFuse(nn.Module):
    """CNN feature + adapted DINO guidance + road prompt 三路融合。"""
    def __init__(self, in_channels_list: Sequence[int], out_channels: int, gn_groups=8):
        super().__init__()
        self.out_channels = int(out_channels)
        in_channels = int(sum(in_channels_list))
        self.reduce = ConvGNAct(in_channels, out_channels, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.refine = nn.Sequential(
            ConvGNAct(out_channels, out_channels, kernel_size=3, gn_groups=gn_groups),
            ResidualRefineGN(out_channels, gn_groups=gn_groups),
        )
        self.eca = ECALayer(out_channels)

    def forward(self, feats: List[torch.Tensor], target_size: Tuple[int, int]):
        aligned = []
        for f in feats:
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(f)
        x = self.reduce(torch.cat(aligned, dim=1))
        x = self.refine(x)
        return self.eca(x)


class RoadBodyPath(nn.Module):
    """道路主体恢复路径，负责宽窄变化、弯曲道路和主体区域。"""
    def __init__(self, c16=256, c8=128, c4=96, p16=128, p8=128, p4=96, gn_groups=8):
        super().__init__()
        self.f16 = TripleFeatureFuse([256, c16, p16], c16, gn_groups=gn_groups)
        self.b16 = RoadDSGDBlock(c16, kernel_size=5, max_offset=2.0, gn_groups=gn_groups)

        self.up16_to8 = ConvGNAct(c16, c8, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.f8 = TripleFeatureFuse([c8, 128, c8, p8], c8, gn_groups=gn_groups)
        self.b8 = nn.Sequential(
            RoadDSGDBlock(c8, kernel_size=5, max_offset=2.0, gn_groups=gn_groups),
            DirectionalStripBlock(c8, kernel_size=15, gn_groups=gn_groups),
        )

        self.up8_to4 = ConvGNAct(c8, c4, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.f4 = TripleFeatureFuse([c4, 64, c4, p4], c4, gn_groups=gn_groups)
        self.b4 = nn.Sequential(
            DirectionalStripBlock(c4, kernel_size=7, gn_groups=gn_groups),
            ResidualRefineGN(c4, gn_groups=gn_groups),
        )

    def forward(self, x1, x2, x3, p4, p8, p16, g4, g8, g16):
        f16 = self.b16(self.f16([x3, g16, p16], target_size=x3.shape[-2:]))
        f16_up = self.up16_to8(F.interpolate(f16, size=x2.shape[-2:], mode="bilinear", align_corners=False))
        f8 = self.b8(self.f8([f16_up, x2, g8, p8], target_size=x2.shape[-2:]))
        f8_up = self.up8_to4(F.interpolate(f8, size=x1.shape[-2:], mode="bilinear", align_corners=False))
        f4 = self.b4(self.f4([f8_up, x1, g4, p4], target_size=x1.shape[-2:]))
        return f16, f8, f4


class FixedSobelFeature(nn.Module):
    """固定 Sobel feature response，不作为监督，只作为 detail path 的边界响应。"""
    def __init__(self, channels: int, gn_groups=8):
        super().__init__()
        self.channels = int(channels)
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4.0
        ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 4.0
        weight = torch.stack([kx, ky], dim=0).view(2, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer("sobel_weight", weight, persistent=False)
        self.fuse = ConvGNAct(channels * 2, channels, kernel_size=1, padding=0, gn_groups=gn_groups)

    def forward(self, x):
        edge = F.conv2d(x, self.sobel_weight.to(dtype=x.dtype), padding=1, groups=self.channels)
        return self.fuse(edge)


class ConnectivityBoundaryDetailPathG(nn.Module):
    """
    DINO-gated connectivity-boundary detail path.

    与 HLD_v1 的区别：
    - 不再直接使用原始 x1；
    - 输入的 P4 已经被 G4 语义门控成 P4_safe；
    - detail path 只作为 body feature 的残差细化来源。
    """
    def __init__(self, c4=96, p4=96, gn_groups=8):
        super().__init__()
        self.reduce = ConvGNAct(c4 + p4 + c4, c4, kernel_size=1, padding=0, gn_groups=gn_groups)
        self.strip = DirectionalStripBlock(c4, kernel_size=15, gn_groups=gn_groups)
        self.sobel = FixedSobelFeature(c4, gn_groups=gn_groups)
        self.fuse = nn.Sequential(
            ConvGNAct(c4 * 2, c4, kernel_size=1, padding=0, gn_groups=gn_groups),
            ResidualRefineGN(c4, gn_groups=gn_groups),
        )

    def forward(self, g4, p4_safe, f4_body):
        x = self.reduce(torch.cat([g4, p4_safe, f4_body], dim=1))
        s = self.strip(x)
        e = self.sobel(x)
        return self.fuse(torch.cat([s, e], dim=1))


class DualAttentionFusion(nn.Module):
    """DAFF-like 空间-通道双注意融合。"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.spatial = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.refine = ResidualRefineGN(channels, gn_groups=8)

    def forward(self, body, detail):
        body_mean = body.mean(dim=1, keepdim=True)
        detail_mean = detail.mean(dim=1, keepdim=True)
        body_max = body.amax(dim=1, keepdim=True)
        detail_max = detail.amax(dim=1, keepdim=True)
        a_s = self.spatial(torch.cat([body_mean, detail_mean, body_max, detail_max], dim=1))
        f = a_s * body + (1.0 - a_s) * detail
        a_c = self.channel(torch.cat([body, detail], dim=1))
        return self.refine(f * a_c + f)


class HeterogeneousRoadDecoderG(nn.Module):
    """
    DINO-safe heterogeneous road decoder.

    核心约束：
    1) G4 语义门控 P4，抑制背景细节进入解码器；
    2) detail path 不再与 body path 平起平坐，而是 beta 缩放的残差细化；
    3) 可选 road presence logit bias，用全局 G16/P16 抑制空图/弱道路图上的误检。
    """
    def __init__(
        self,
        c16=256,
        c8=128,
        c4=96,
        c2=64,
        p16=128,
        p8=128,
        p4=96,
        n_classes=1,
        gn_groups=8,
        detail_beta_init=0.05,
        use_presence_gate=True,
        presence_bias_scale=0.5,
    ):
        super().__init__()
        self.body = RoadBodyPath(c16=c16, c8=c8, c4=c4, p16=p16, p8=p8, p4=p4, gn_groups=gn_groups)
        self.p4_gate = nn.Sequential(
            ConvGNAct(c4, c4, kernel_size=3, gn_groups=gn_groups),
            nn.Conv2d(c4, p4, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.detail = ConnectivityBoundaryDetailPathG(c4=c4, p4=p4, gn_groups=gn_groups)
        self.detail_refine = ResidualRefineGN(c4, gn_groups=gn_groups)
        self.detail_beta = nn.Parameter(torch.tensor(float(detail_beta_init), dtype=torch.float32))
        self.final_dec = ResidualDecoderBlock(in_channels=c4, skip_channels=64, out_channels=c2)
        self.out_head = nn.Sequential(
            ConvBNAct(c2, c2, kernel_size=3),
            ConvBNAct(c2, c2, kernel_size=3),
            nn.Conv2d(c2, n_classes, kernel_size=1),
        )

        self.use_presence_gate = bool(use_presence_gate)
        self.presence_bias_scale = float(presence_bias_scale)
        if self.use_presence_gate:
            hidden = max((c16 + p16) // 4, 64)
            self.presence_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c16 + p16, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            # zero-init: 初始不影响 logits，后续学习全局 road presence 偏置。
            nn.init.zeros_(self.presence_head[-1].weight)
            nn.init.zeros_(self.presence_head[-1].bias)
        else:
            self.presence_head = None

    def forward(self, x0, x1, x2, x3, p4, p8, p16, g4, g8, g16, input_size: Tuple[int, int]):
        m4 = self.p4_gate(g4)
        # 不完全压死细节：0.25 + 0.75*M4，避免 recall 过度下降。
        p4_safe = p4 * (0.25 + 0.75 * m4)

        f16_body, f8_body, f4_body = self.body(x1, x2, x3, p4_safe, p8, p16, g4, g8, g16)
        f4_detail = self.detail(g4, p4_safe, f4_body)
        beta = torch.clamp(self.detail_beta, 0.0, 1.0)
        f4 = self.detail_refine(f4_body + beta * m4 * f4_detail)

        d0 = self.final_dec(f4, x0)
        logits_half = self.out_head(d0)
        logits = F.interpolate(logits_half, size=input_size, mode="bilinear", align_corners=False)

        presence_bias = None
        if self.use_presence_gate:
            gp = torch.cat([g16, p16], dim=1)
            presence_bias = torch.tanh(self.presence_head(gp)).view(-1, 1, 1, 1) * self.presence_bias_scale
            logits = logits + presence_bias
            logits_half = logits_half + presence_bias

        aux = {
            "f16_body": f16_body,
            "f8_body": f8_body,
            "f4_body": f4_body,
            "f4_detail": f4_detail,
            "f4_fused": f4,
            "p4_gate": m4,
            "p4_safe": p4_safe,
            "detail_beta": beta.detach(),
            "presence_bias": presence_bias,
            "d0": d0,
            "logits_half": logits_half,
        }
        return logits, aux


class HLD_v1_G(nn.Module):
    """
    HLD_v1_G: DINO-safe High-Low DINO Adaptation Network for Road Extraction.

    核心：
        High-resolution Road Experts -> WPDA DINO token adaptation -> DINO-guided heterogeneous decoder.

    第一版只使用 final logits 监督，不输出 boundary/connect/dino_prior loss 分支。
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

        prompt_c4=96,
        prompt_c8=128,
        prompt_c16=128,
        snake_kernel=7,
        snake_offset=3.0,

        wpda_dim=128,
        wpda_heads=4,
        wpda_window_size=8,
        wpda_num_prototypes=16,
        wpda_layers=(8, 11),
        wpda_alpha_init=0.01,
        wpda_dropout=0.05,

        decoder_c16=256,
        decoder_c8=128,
        decoder_c4=96,
        decoder_c2=64,
        gn_groups=8,
        detail_beta_init=0.05,
        use_presence_gate=True,
        presence_bias_scale=0.5,
        wpda_alpha_max=0.15,
        **kwargs,
    ):
        super().__init__()
        if in_channels is not None:
            n_channels = in_channels
        if num_classes is not None:
            n_classes = num_classes

        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.return_aux = bool(return_aux)
        self.dino_layers = [int(x) for x in dino_layers]
        self.wpda_layers = [int(x) for x in wpda_layers]
        self.dino_patch_size = int(dino_patch_size)
        self.dino_embed_dim = int(dino_embed_dim)
        self.param_group_lrs = kwargs.get("param_group_lrs", {}) if isinstance(kwargs.get("param_group_lrs", {}), dict) else {}
        self.param_group_wds = kwargs.get("param_group_weight_decays", {}) if isinstance(kwargs.get("param_group_weight_decays", {}), dict) else {}

        if 2 not in self.dino_layers:
            raise ValueError("HLD_v1_G 当前 readout 需要 dino_layers 包含 2。")
        for layer in self.wpda_layers:
            if layer not in self.dino_layers:
                raise ValueError(f"wpda_layers 中的 layer={layer} 不在 dino_layers={self.dino_layers} 中。")

        self.road_branch = HighResolutionRoadBranch(n_channels=self.n_channels, pretrained=pretrained)
        self.dino = FrozenDINOv3TokenExtractor(
            dino_model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            out_layers=dino_layers,
            embed_dim=dino_embed_dim,
            patch_size=dino_patch_size,
            dino_normalize=dino_normalize,
            dino_intermediate_norm=dino_intermediate_norm,
        )

        self.prompt_extractor = RoadDSGDPromptExtractorG(
            c4=prompt_c4,
            c8=prompt_c8,
            c16=prompt_c16,
            gn_groups=gn_groups,
            snake_kernel=snake_kernel,
            snake_offset=snake_offset,
        )

        self.wpda_adapters = nn.ModuleDict({
            str(layer): WPDAAdapterG(
                token_dim=dino_embed_dim,
                prompt_dim=prompt_c16,
                inner_dim=wpda_dim,
                heads=wpda_heads,
                window_size=wpda_window_size,
                num_prototypes=wpda_num_prototypes,
                alpha_init=wpda_alpha_init,
                alpha_max=wpda_alpha_max,
                dropout=wpda_dropout,
                gn_groups=gn_groups,
            )
            for layer in self.wpda_layers
        })

        self.readout = HLDReadout(
            embed_dim=dino_embed_dim,
            c16=decoder_c16,
            c8=decoder_c8,
            c4=decoder_c4,
            p8_channels=prompt_c8,
            p4_channels=prompt_c4,
            gn_groups=gn_groups,
        )

        self.decoder = HeterogeneousRoadDecoderG(
            c16=decoder_c16,
            c8=decoder_c8,
            c4=decoder_c4,
            c2=decoder_c2,
            p16=prompt_c16,
            p8=prompt_c8,
            p4=prompt_c4,
            n_classes=self.n_classes,
            gn_groups=gn_groups,
            detail_beta_init=detail_beta_init,
            use_presence_gate=use_presence_gate,
            presence_bias_scale=presence_bias_scale,
        )

        self._print_trainable_summary()

    def _print_trainable_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dino_total = sum(p.numel() for p in self.dino.parameters())
        dino_trainable = sum(p.numel() for p in self.dino.parameters() if p.requires_grad)
        print("--------------------------------------------------")
        print("🔧 HLD_v1_G DINO-safe High-Low Adaptation Network")
        print(f"    - 总参数量:         {total / 1e6:.2f} M")
        print(f"    - 可训练参数量:     {trainable / 1e6:.2f} M")
        print(f"    - DINO总参数量:     {dino_total / 1e6:.2f} M")
        print(f"    - DINO可训练参数量: {dino_trainable / 1e6:.2f} M")
        print("    - 说明: DINOv3 原始主干冻结；训练 DINO-safe Prompt、WPDA-G Adapter、HLD Readout 和受控 detail decoder")
        print("--------------------------------------------------")

    def train(self, mode=True):
        super().train(mode)
        self.dino.train(False)
        return self

    def _adapt_tokens(self, token_dict: Dict[int, torch.Tensor], p16: torch.Tensor, patch_hw: Tuple[int, int]):
        adapted = dict(token_dict)
        for layer in self.wpda_layers:
            key = str(layer)
            adapted[layer] = self.wpda_adapters[key](token_dict[layer], p16, patch_hw=patch_hw)
        return adapted

    def forward_features(self, x, need_aux=False):
        input_size = x.shape[-2:]
        if input_size[0] % self.dino_patch_size != 0 or input_size[1] % self.dino_patch_size != 0:
            raise RuntimeError(
                f"HLD_v1_G 要求输入 H/W 能被 dino_patch_size 整除。当前输入={input_size}, patch_size={self.dino_patch_size}"
            )

        x0, x1, x2, x3, x_in = self.road_branch(x)
        patch_hw = x3.shape[-2:]

        token_dict = self.dino(x_in)
        p4, p8, p16 = self.prompt_extractor(x1, x2, x3)
        adapted_tokens = self._adapt_tokens(token_dict, p16, patch_hw=patch_hw)
        g16, g8, g4 = self.readout(adapted_tokens, p8=p8, p4=p4, patch_hw=patch_hw)

        logits, dec_aux = self.decoder(
            x0=x0,
            x1=x1,
            x2=x2,
            x3=x3,
            p4=p4,
            p8=p8,
            p16=p16,
            g4=g4,
            g8=g8,
            g16=g16,
            input_size=input_size,
        )

        aux = None
        if need_aux:
            aux = {
                "final_logits": logits,
                "x0": x0,
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "P4": p4,
                "P8": p8,
                "P16": p16,
                "G4": g4,
                "G8": g8,
                "G16": g16,
            }
            aux.update(dec_aux)
            for layer, token in token_dict.items():
                aux[f"T{layer}"] = token
            for layer in self.wpda_layers:
                aux[f"T{layer}_adapted"] = adapted_tokens[layer]

        return logits, aux

    def forward(self, x):
        logits, aux = self.forward_features(x, need_aux=self.return_aux)
        if self.return_aux:
            return aux
        return logits

    def forward_train(self, x):
        logits, _ = self.forward_features(x, need_aux=False)
        return {
            "final_logits": logits,
            "base_logits": None,
            "dino_prior_logits": None,
        }


    def get_param_groups(self, base_lr=1e-4, weight_decay=0.01):
        """
        Optimizer param groups for train_HLD_v2.py.
        Priority:
          1) values from JSON model.param_group_lrs / model.param_group_weight_decays;
          2) safe defaults below.
        """
        lr_cfg = self.param_group_lrs or {}
        wd_cfg = self.param_group_wds or {}
        groups = []
        named_modules = [
            (
                "road_branch",
                self.road_branch,
                float(lr_cfg.get("road_branch_lr", 3e-5)),
                float(wd_cfg.get("road_branch_wd", 0.01)),
            ),
            (
                "prompt_extractor",
                self.prompt_extractor,
                float(lr_cfg.get("prompt_extractor_lr", base_lr)),
                float(wd_cfg.get("prompt_extractor_wd", 0.02)),
            ),
            (
                "wpda_adapters",
                self.wpda_adapters,
                float(lr_cfg.get("wpda_lr", base_lr)),
                float(wd_cfg.get("wpda_wd", 0.02)),
            ),
            (
                "readout",
                self.readout,
                float(lr_cfg.get("hld_readout_lr", base_lr)),
                float(wd_cfg.get("hld_readout_wd", weight_decay)),
            ),
            (
                "decoder",
                self.decoder,
                float(lr_cfg.get("decoder_lr", base_lr)),
                float(wd_cfg.get("decoder_wd", weight_decay)),
            ),
        ]
        for name, module, lr, wd in named_modules:
            params = [p for p in module.parameters() if p.requires_grad]
            if len(params) > 0:
                groups.append({"params": params, "lr": lr, "weight_decay": wd, "name": name})
        return groups


if __name__ == "__main__":
    # 仅用于结构检查；实际运行需要提供真实 dino_repo_path / dino_ckpt_path。
    model = HLD_v1_G(
        n_channels=3,
        n_classes=1,
        pretrained=False,
        return_aux=False,
        dino_ckpt_path="",
    )
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("Input :", x.shape)
    if isinstance(y, dict):
        print("Output keys:", y.keys())
    else:
        print("Output:", y.shape)
