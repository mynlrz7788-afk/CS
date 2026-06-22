# -*- coding: utf-8 -*-
"""
DINOv3OnlySeg.py

放置位置：
    SEG/models/custom/DINOv3OnlySeg.py

作用：
    最朴素的 DINOv3 道路分割对照组。
    不使用 DINOv3 Adapter。
    不使用 FAPM。
    不使用 U-Net decoder。
    不使用高分辨率路径。
    不使用高低分辨率互导。

结构：
    输入图像
    ↓
    DINOv3 backbone 取最后一层 patch feature
    ↓
    1×1 分类头
    ↓
    双线性上采样到原图大小
    ↓
    输出 logits

注意：
    segmentation head 是必须的。没有任何 head，DINOv3 只能输出特征，不能直接输出二值 mask。
    这里的 head 只有 1×1 Conv，相当于最弱线性探针，适合测试 DINOv3 原始密集特征对道路分割是否有用。
"""

from typing import Optional
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from dinounet.dinov3.hub.backbones import dinov3_vits16, dinov3_vitb16, dinov3_vitl16

try:
    from dinounet.dinov3.hub.backbones import dinov3_vit7b16
except Exception:
    dinov3_vit7b16 = None


DINO_FACTORIES = {
    "dinounet_s": dinov3_vits16,
    "dinov3_vits16": dinov3_vits16,
    "dinounet_b": dinov3_vitb16,
    "dinov3_vitb16": dinov3_vitb16,
    "dinounet_l": dinov3_vitl16,
    "dinov3_vitl16": dinov3_vitl16,
}
if dinov3_vit7b16 is not None:
    DINO_FACTORIES["dinounet_7b"] = dinov3_vit7b16
    DINO_FACTORIES["dinov3_vit7b16"] = dinov3_vit7b16


DINO_INFO = {
    "dinounet_s": {"embed_dim": 384, "last_layer": 11, "patch_size": 16},
    "dinov3_vits16": {"embed_dim": 384, "last_layer": 11, "patch_size": 16},
    "dinounet_b": {"embed_dim": 768, "last_layer": 11, "patch_size": 16},
    "dinov3_vitb16": {"embed_dim": 768, "last_layer": 11, "patch_size": 16},
    "dinounet_l": {"embed_dim": 1024, "last_layer": 23, "patch_size": 16},
    "dinov3_vitl16": {"embed_dim": 1024, "last_layer": 23, "patch_size": 16},
    "dinounet_7b": {"embed_dim": 4096, "last_layer": 39, "patch_size": 16},
    "dinov3_vit7b16": {"embed_dim": 4096, "last_layer": 39, "patch_size": 16},
}


def _strip_state_dict_prefix(state_dict):
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


def load_dinov3_backbone(model_name: str, pretrained_path: Optional[str]):
    if model_name not in DINO_FACTORIES:
        raise ValueError(f"不支持的 DINOv3 模型: {model_name}. 当前支持: {list(DINO_FACTORIES.keys())}")

    if pretrained_path is None or not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f"找不到 DINOv3 权重: {pretrained_path}")

    print(f"🔧 构建纯 DINOv3 backbone: {model_name}")
    model = DINO_FACTORIES[model_name](pretrained=False)

    print(f"📦 加载 DINOv3 权重: {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location="cpu")
    state_dict = _strip_state_dict_prefix(state_dict)
    msg = model.load_state_dict(state_dict, strict=True)
    print(f"✅ DINOv3 权重加载完成: {msg}")
    return model


class DINOv3OnlySeg(nn.Module):
    """
    只用 DINOv3 特征做道路分割的最朴素 baseline。

    默认：
        freeze_backbone=True，只训练 1×1 segmentation head。

    如果你想测试 DINOv3 全量微调，可以把 freeze_backbone=False，
    但那就不是“原始 DINOv3 特征适合不适合道路”的线性探针实验了。
    """
    def __init__(
        self,
        num_classes: int = 1,
        dinov3_model: str = "dinounet_s",
        pretrained_path: Optional[str] = None,
        img_size: int = 1024,
        freeze_backbone: bool = True,
        imagenet_norm: bool = True,
        input_already_normalized: bool = False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dinov3_model = dinov3_model
        self.img_size = img_size
        self.freeze_backbone = freeze_backbone
        self.imagenet_norm = imagenet_norm
        self.input_already_normalized = input_already_normalized

        info = DINO_INFO[dinov3_model]
        self.embed_dim = info["embed_dim"]
        self.patch_size = info["patch_size"]
        self.layer_idx = info["last_layer"] if layer_idx is None else layer_idx

        self.backbone = load_dinov3_backbone(dinov3_model, pretrained_path)
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

        # 最小分割头：1×1 线性分类。
        # 不用普通 decoder，不用 U-Net，不做多尺度融合。
        self.seg_head = nn.Conv2d(self.embed_dim, num_classes, kernel_size=1)

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

        print("--------------------------------------------------")
        print("DINOv3OnlySeg")
        print(f"  model: {dinov3_model}")
        print(f"  embed_dim: {self.embed_dim}")
        print(f"  patch_size: {self.patch_size}")
        print(f"  layer_idx: {self.layer_idx}")
        print(f"  freeze_backbone: {self.freeze_backbone}")
        print("  decoder: None")
        print("  adapter: None")
        print("  FAPM: None")
        print("  head: 1x1 Conv only")
        print("--------------------------------------------------")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _normalize(self, x):
        if not self.imagenet_norm or self.input_already_normalized:
            return x
        return (x - self.img_mean) / self.img_std

    def _extract_last_feature(self, x):
        """
        尽量兼容 DINOv3 / DINOv2 风格的 get_intermediate_layers。
        目标输出 B×C×H/16×W/16。
        """
        b, _, h, w = x.shape
        hp = h // self.patch_size
        wp = w // self.patch_size

        # 常见 DINO 接口，优先使用 reshape=True
        try:
            outs = self.backbone.get_intermediate_layers(
                x,
                n=[self.layer_idx],
                reshape=True,
                return_class_token=False,
                norm=True,
            )
        except TypeError:
            try:
                outs = self.backbone.get_intermediate_layers(
                    x,
                    n=[self.layer_idx],
                    reshape=True,
                    return_class_token=False,
                )
            except TypeError:
                outs = self.backbone.get_intermediate_layers(x, n=[self.layer_idx])

        feat = outs[0] if isinstance(outs, (list, tuple)) else outs

        # 有些实现返回 (patch_tokens, cls_token)
        if isinstance(feat, (list, tuple)):
            feat = feat[0]

        # B,C,H,W
        if feat.dim() == 4:
            return feat

        # B,N,C，需要 reshape 成 B,C,H,W
        if feat.dim() == 3:
            # 如果包含 cls token，去掉第一个
            if feat.shape[1] == hp * wp + 1:
                feat = feat[:, 1:, :]
            if feat.shape[1] != hp * wp:
                raise RuntimeError(
                    f"DINO token 数量不匹配: got {feat.shape[1]}, expected {hp * wp}. "
                    f"输入尺寸={h}x{w}, patch_size={self.patch_size}"
                )
            feat = feat.transpose(1, 2).contiguous().view(b, self.embed_dim, hp, wp)
            return feat

        raise RuntimeError(f"无法识别 DINOv3 feature 形状: {feat.shape}")

    def forward(self, x):
        out_size = x.shape[-2:]
        x = self._normalize(x)

        if self.freeze_backbone:
            with torch.no_grad():
                feat = self._extract_last_feature(x)
        else:
            feat = self._extract_last_feature(x)

        logits_low = self.seg_head(feat)
        logits = F.interpolate(logits_low, size=out_size, mode="bilinear", align_corners=False)
        return logits
