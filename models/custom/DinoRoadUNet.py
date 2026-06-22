
# -*- coding: utf-8 -*-
"""
DinoRoadUNet.py

放置位置：
    SEG/models/custom/DinoRoadUNet.py

用途：
    在你现有的道路分割框架中复现 Dino U-Net 思路。
    第一版目标是忠实复现：
        DINOv3 冻结主干 + DINOv3_Adapter + FAPM + U-Net Decoder

注意：
    1. 需要先把原仓库的 dinounet/dinov3 复制到 SEG/dinounet/dinov3。
    2. 需要先安装 MultiScaleDeformableAttention 自定义算子。
    3. 需要下载 DINOv3 的 .pth 权重，并在 config 中填写 pretrained_path。
"""

import os
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dinounet.dinov3.eval.segmentation.models.backbone.dinov3_adapter import DINOv3_Adapter
from dinounet.dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16

try:
    from dinounet.dinov3.hub.backbones import dinov3_vit7b16
except Exception:
    dinov3_vit7b16 = None


DINOv3_MODEL_FACTORIES = {
    "dinounet_s": dinov3_vits16,
    "dinounet_b": dinov3_vitb16,
    "dinounet_l": dinov3_vitl16,
}

if dinov3_vit7b16 is not None:
    DINOv3_MODEL_FACTORIES["dinounet_7b"] = dinov3_vit7b16


DINOv3_INTERACTION_INDEXES = {
    "dinounet_s": [2, 5, 8, 11],
    "dinounet_b": [2, 5, 8, 11],
    "dinounet_l": [4, 11, 17, 23],
    "dinounet_7b": [9, 19, 29, 39],
}


DINOv3_MODEL_INFO = {
    "dinounet_s": {"embed_dim": 384, "depth": 12, "num_heads": 6},
    "dinounet_b": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "dinounet_l": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "dinounet_7b": {"embed_dim": 4096, "depth": 40, "num_heads": 32},
}


def _strip_state_dict_prefix(state_dict):
    """兼容少数权重里带 module. 或 backbone. 前缀的情况。"""
    if not isinstance(state_dict, dict):
        return state_dict

    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("backbone."):
            nk = nk[len("backbone."):]
        new_sd[nk] = v
    return new_sd


def load_dinov3_model(model_name: str, pretrained_path: str):
    """
    加载 DINOv3 主干。
    这里强制使用本地 .pth 权重，避免训练时自动联网。
    """
    if model_name not in DINOv3_MODEL_FACTORIES:
        raise ValueError(f"不支持的 DINOv3 模型: {model_name}. 当前支持: {list(DINOv3_MODEL_FACTORIES.keys())}")

    if pretrained_path is None or not os.path.isfile(pretrained_path):
        raise FileNotFoundError(
            f"找不到 DINOv3 权重: {pretrained_path}\n"
            f"请把 .pth 权重放到 SEG/weight/dinov3/，并在 config 里填写 pretrained_path。"
        )

    print(f"🔧 构建 DINOv3 backbone: {model_name}")
    model = DINOv3_MODEL_FACTORIES[model_name](pretrained=False)

    print(f"📦 加载 DINOv3 权重: {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location="cpu")
    state_dict = _strip_state_dict_prefix(state_dict)
    msg = model.load_state_dict(state_dict, strict=True)
    print(f"✅ DINOv3 权重加载完成: {msg}")

    return model


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bias: bool = False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=bias),
            nn.Conv2d(in_ch, out_ch, 1, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FAPM(nn.Module):
    """
    Fidelity-Aware Projection Module 的道路复现版。

    输入：
        4 个尺度的 DINOv3_Adapter 特征，每个尺度通道数都是 DINO embed_dim。
    输出：
        4 个尺度的低维特征，通道数为 out_ch_list。
    """
    def __init__(self, in_ch: int, rank: int, out_ch_list: List[int], bias: bool = False):
        super().__init__()
        self.out_ch_list = out_ch_list
        self.shared_basis = nn.Conv2d(in_ch, rank, kernel_size=1, bias=bias)

        self.specific_bases = nn.ModuleList([
            nn.Conv2d(in_ch, rank, kernel_size=1, bias=bias)
            for _ in out_ch_list
        ])

        self.film_generators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(rank, rank, 1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(rank, rank * 2, 1, bias=True),
            )
            for _ in out_ch_list
        ])

        self.refine_blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        for oc in out_ch_list:
            self.refine_blocks.append(nn.Sequential(
                nn.Conv2d(rank, oc, 1, bias=bias),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                DepthwiseSeparableConv(oc, oc, bias=bias),
                nn.Conv2d(oc, oc, 1, bias=bias),
                SqueezeExcitation(oc),
            ))

            if rank != oc:
                self.shortcuts.append(nn.Conv2d(rank, oc, 1, bias=bias))
            else:
                self.shortcuts.append(nn.Identity())

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        outs = []
        for i, x in enumerate(x_list):
            z_shared = self.shared_basis(x)
            z_specific = self.specific_bases[i](x)

            gamma_beta = self.film_generators[i](z_shared)
            gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)

            z_mod = gamma * z_specific + beta
            y = self.refine_blocks[i](z_mod) + self.shortcuts[i](z_mod)
            outs.append(y)
        return outs


class DinoEncoderForRoad(nn.Module):
    """
    DINOv3_Adapter + FAPM。
    输出尺度保持为：
        s1: H/4
        s2: H/8
        s3: H/16
        s4: H/32
    这样比把最高层上采样到 H 更省显存，更适合 1024x1024 道路分割。
    """
    def __init__(
        self,
        dinov3_model: str,
        pretrained_path: str,
        out_channels: List[int],
        rank: int = 256,
        img_size: int = 1024,
        freeze_backbone: bool = True,
        conv_inplane: int = 64,
        deform_num_heads: int = 16,
        n_points: int = 4,
        with_cp: bool = False,
    ):
        super().__init__()
        self.dinov3_model = dinov3_model
        self.out_channels = out_channels
        self.freeze_backbone = freeze_backbone
        self.img_size = img_size

        if dinov3_model not in DINOv3_MODEL_INFO:
            raise ValueError(f"未知 dinov3_model: {dinov3_model}")

        model_info = DINOv3_MODEL_INFO[dinov3_model]
        interaction_indexes = DINOv3_INTERACTION_INDEXES[dinov3_model]

        backbone = load_dinov3_model(dinov3_model, pretrained_path)

        if freeze_backbone:
            backbone.requires_grad_(False)
            backbone.eval()

        print("--------------------------------------------------")
        print("DinoRoadUNet Encoder")
        print(f"  DINOv3 model: {dinov3_model}")
        print(f"  embed_dim: {model_info['embed_dim']}")
        print(f"  interaction_indexes: {interaction_indexes}")
        print(f"  out_channels: {out_channels}")
        print(f"  FAPM rank: {rank}")
        print(f"  freeze_backbone: {freeze_backbone}")
        print("--------------------------------------------------")

        self.dinov3_adapter = DINOv3_Adapter(
            backbone=backbone,
            interaction_indexes=interaction_indexes,
            pretrain_size=img_size,
            conv_inplane=conv_inplane,
            n_points=n_points,
            deform_num_heads=deform_num_heads,
            drop_path_rate=0.3,
            init_values=0.0,
            with_cffn=True,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            add_vit_feature=True,
            use_extra_extractor=True,
            with_cp=with_cp,
        )

        if freeze_backbone:
            self.dinov3_adapter.backbone.requires_grad_(False)
            self.dinov3_adapter.backbone.eval()

        self.fapm = FAPM(
            in_ch=model_info["embed_dim"],
            rank=rank,
            out_ch_list=out_channels,
            bias=False,
        )

    def _force_backbone_eval(self):
        if self.freeze_backbone:
            self.dinov3_adapter.backbone.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._force_backbone_eval()

        b, c, h, w = x.shape
        if c == 1:
            x = x.repeat(1, 3, 1, 1)
        elif c > 3:
            x = x[:, :3]
        elif c < 3:
            x = x.repeat(1, 3 // c + 1, 1, 1)[:, :3]

        feats = self.dinov3_adapter(x)
        x_list = [feats["1"], feats["2"], feats["3"], feats["4"]]
        ys = self.fapm(x_list)

        # 对齐到标准道路分割尺度，避免尺寸误差导致 concat 报错。
        targets = [
            (h // 4, w // 4),
            (h // 8, w // 8),
            (h // 16, w // 16),
            (h // 32, w // 32),
        ]

        outs = []
        for y, size in zip(ys, targets):
            if y.shape[-2:] != size:
                y = F.interpolate(y, size=size, mode="bilinear", align_corners=False)
            outs.append(y)

        return outs


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpCatBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBNReLU(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class RoadUNetDecoder(nn.Module):
    """
    输入 encoder 的 4 个尺度：
        s1 H/4, s2 H/8, s3 H/16, s4 H/32
    输出：
        B x num_classes x H x W
    """
    def __init__(self, channels: List[int], num_classes: int = 1):
        super().__init__()
        c1, c2, c3, c4 = channels

        self.up3 = UpCatBlock(c4, c3, c3)
        self.up2 = UpCatBlock(c3, c2, c2)
        self.up1 = UpCatBlock(c2, c1, c1)

        self.hr = nn.Sequential(
            nn.ConvTranspose2d(c1, max(c1 // 2, 32), kernel_size=2, stride=2),
            ConvBNReLU(max(c1 // 2, 32), max(c1 // 2, 32)),
            nn.ConvTranspose2d(max(c1 // 2, 32), 32, kernel_size=2, stride=2),
            ConvBNReLU(32, 32),
        )
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, skips: List[torch.Tensor], out_size: Tuple[int, int]):
        s1, s2, s3, s4 = skips
        x = self.up3(s4, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.hr(x)
        logits = self.seg_head(x)
        if logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits


class DinoRoadUNet(nn.Module):
    """
    道路分割版 Dino U-Net。

    参数示例：
        DinoRoadUNet(
            num_classes=1,
            dinov3_model="dinounet_s",
            pretrained_path="weight/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            out_channels=[64, 128, 256, 512],
            rank=256,
            img_size=1024
        )
    """
    def __init__(
        self,
        num_classes: int = 1,
        dinov3_model: str = "dinounet_s",
        pretrained_path: Optional[str] = None,
        out_channels: Optional[List[int]] = None,
        rank: int = 256,
        img_size: int = 1024,
        freeze_backbone: bool = True,
        imagenet_norm: bool = True,
        input_already_normalized: bool = False,
        conv_inplane: int = 64,
        deform_num_heads: int = 16,
        n_points: int = 4,
        with_cp: bool = False,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = [64, 128, 256, 512]

        self.num_classes = num_classes
        self.imagenet_norm = imagenet_norm
        self.input_already_normalized = input_already_normalized
        self.freeze_backbone = freeze_backbone

        self.encoder = DinoEncoderForRoad(
            dinov3_model=dinov3_model,
            pretrained_path=pretrained_path,
            out_channels=out_channels,
            rank=rank,
            img_size=img_size,
            freeze_backbone=freeze_backbone,
            conv_inplane=conv_inplane,
            deform_num_heads=deform_num_heads,
            n_points=n_points,
            with_cp=with_cp,
        )

        self.decoder = RoadUNetDecoder(out_channels, num_classes=num_classes)

        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.dinov3_adapter.backbone.eval()
        return self

    def _normalize_for_dino(self, x: torch.Tensor) -> torch.Tensor:
        if not self.imagenet_norm or self.input_already_normalized:
            return x

        # 你的 RoadDataset 如果输出是 0~1，这里直接做 ImageNet Normalize。
        # 如果 Dataset 里已经 Normalize，config 中把 input_already_normalized 设为 true。
        return (x - self.img_mean) / self.img_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_size = x.shape[-2:]
        x = self._normalize_for_dino(x)
        skips = self.encoder(x)
        logits = self.decoder(skips, out_size=out_size)
        return logits
