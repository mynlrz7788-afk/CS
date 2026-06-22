import os
import sys
import math
import importlib
from typing import Sequence, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DC_v2"]


def _auto_padding(kernel_size, dilation=1):
    if isinstance(kernel_size, tuple):
        if isinstance(dilation, tuple):
            return tuple(((k - 1) // 2) * d for k, d in zip(kernel_size, dilation))
        return tuple(((k - 1) // 2) * dilation for k in kernel_size)
    return ((kernel_size - 1) // 2) * dilation


class ConvBNAct(nn.Module):
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


class ECALayer(nn.Module):
    """
    和 HL_base 一致的 ECA 通道注意力。
    """

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class ResidualDecoderBlock(nn.Module):
    """
    和 HL_base 一致的残差解码块。

    upsample decoder feature
    → concat skip
    → 2×ConvBNAct
    → ECA
    → shortcut
    → residual add
    → ReLU
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.fuse = nn.Sequential(
            ConvBNAct(in_channels + skip_channels, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        )

        self.eca = ECALayer(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        f = torch.cat([x, skip], dim=1)
        out = self.fuse(f)
        out = self.eca(out)
        out = out + self.shortcut(f)
        return self.act(out)


class FrozenDINOv3FeatureExtractor(nn.Module):
    """
    直接从 dinounet/dinov3 构建 DINOv3。

    不依赖 DC_v2_step2.py。
    不使用 torch.hub.load。
    不训练 DINO 原始参数。

    返回多层 DINO feature map。
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
        self.out_layers = list(out_layers)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.dino_normalize = bool(dino_normalize)
        self.dino_intermediate_norm = bool(dino_intermediate_norm)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.backbone = self._build_dino_backbone()
        self._freeze_backbone()

        print("[DC_v2] 使用 dinounet/dinov3 官方 get_intermediate_layers 提取 DINO 特征")

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
                "导入 dinov3.hub.backbones 失败。请确认 dino_repo_path 指向 "
                ".../SEG/dinounet/dinov3。\n"
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

                for prefix in (
                    "module.",
                    "backbone.",
                    "student.",
                    "teacher.",
                    "model.",
                ):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]

                clean_state[nk] = v

            missing, unexpected = model.load_state_dict(clean_state, strict=False)

            print(
                f"[DC_v2] 加载 DINOv3 权重完成: {self.dino_ckpt_path}\n"
                f"        missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
            )

            if len(missing) > 0:
                print(f"        missing 示例: {missing[:10]}")
            if len(unexpected) > 0:
                print(f"        unexpected 示例: {unexpected[:10]}")

        return model

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)

        # 保证 DINO 始终 eval，不受外部 model.train() 影响。
        self.backbone.eval()

        return self

    @staticmethod
    def _tokens_to_map(feat, h, w, patch_size):
        """
        如果官方接口没有 reshape，这里兜底转成 B×C×H×W。
        """

        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if feat.dim() == 4:
            return feat

        if feat.dim() != 3:
            raise RuntimeError(f"DINO 特征维度异常: {feat.shape}")

        b, n, c = feat.shape
        patch_h = h // patch_size
        patch_w = w // patch_size
        patch_n = patch_h * patch_w

        if n > patch_n:
            special_n = n - patch_n
            feat = feat[:, special_n:, :]

        if feat.shape[1] != patch_n:
            raise RuntimeError(
                f"DINO token 无法 reshape。"
                f"当前 token 数={feat.shape[1]}, 预期 patch 数={patch_n}"
            )

        feat = feat.transpose(1, 2).contiguous()
        feat = feat.view(b, c, patch_h, patch_w)
        return feat

    @torch.no_grad()
    def forward(self, x):
        if self.dino_normalize:
            x = (x - self.mean) / self.std

        _, _, h, w = x.shape

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise RuntimeError(
                f"DINO 输入尺寸必须能被 patch_size 整除。"
                f"当前输入: {h}x{w}, patch_size={self.patch_size}"
            )

        if not hasattr(self.backbone, "get_intermediate_layers"):
            raise RuntimeError(
                "当前 DINOv3 backbone 没有 get_intermediate_layers。"
                "请检查 dinounet/dinov3/models/vision_transformer.py。"
            )

        # 优先用官方接口。
        # norm 默认 False，是为了和之前 DC_v1 手写 block 版本更可比。
        try:
            feats = self.backbone.get_intermediate_layers(
                x,
                n=self.out_layers,
                reshape=True,
                return_class_token=False,
                norm=self.dino_intermediate_norm,
            )
        except TypeError:
            # 兼容不支持 norm 参数的版本。
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
            raise RuntimeError(
                f"DINO 输出层数不匹配。期望 {len(self.out_layers)} 层，"
                f"实际得到 {len(feats)} 层。"
            )

        outputs = {}

        for layer, feat in zip(self.out_layers, feats):
            feat = self._tokens_to_map(
                feat,
                h=h,
                w=w,
                patch_size=self.patch_size,
            )
            outputs[layer] = feat

        return outputs


class DinoOutAdapter(nn.Module):
    """
    DINO 每一层输出后的外置 Adapter。

    它不改变 DINO 本体，只负责把 DINO feature map 转换成 decoder 可用特征。
    """

    def __init__(self, in_ch=384, out_ch=128, bottleneck=64):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.local = nn.Sequential(
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                groups=out_ch,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(out_ch, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.proj_in(x)
        y = self.local(x)
        z = self.mlp(x + y)
        return self.act(x + y + z)


class PyramidBlock(nn.Module):
    """
    从 DINO H/16 特征生成不同尺度的 skip feature。
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.proj = nn.Sequential(
            ConvBNAct(in_ch, out_ch, kernel_size=3),
            ConvBNAct(out_ch, out_ch, kernel_size=3),
        )

    def forward(self, x, size):
        if x.shape[-2:] != size:
            x = F.interpolate(
                x,
                size=size,
                mode="bilinear",
                align_corners=False,
            )
        return self.proj(x)


class DC_v2(nn.Module):
    """
    DC_v2: Frozen DINOv3 + HL_base 风格解码器

    目的：
        验证 DC_v1 的瓶颈是不是轻量 decoder。
        不融合 HL_base。
        不使用 CNN 编码器。
        只把 decoder 换成 HL_base 风格的 ResidualDecoderBlock。

    结构：
        输入图像
        -> Frozen DINOv3
        -> 多层 DINO dense features
        -> 外置 Adapter
        -> 融合为 F16
        -> 构造 DINO-only feature pyramid:
            P4: H/32, 512
            P3: H/16, 256
            P2: H/8,  128
            P1: H/4,  64
            P0: H/2,  64
        -> HL_base 同款 decoder:
            d3 = dec3(P4, P3)
            d2 = dec2(d3, P2)
            d1 = dec1(d2, P1)
            d0 = dec0(d1, P0)
        -> H/2 output head
        -> upsample 到原图
    """

    def __init__(
        self,
        num_classes=1,
        n_channels=3,
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

        adapter_channels=128,
        adapter_bottleneck=64,

        fuse_channels=256,

        # 保留这个字段是为了兼容旧配置。当前版本没有 DINO 内部 Adapter。
        train_dino_branch_adapters=False,

        **kwargs,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.n_channels = int(n_channels)
        self.return_aux = bool(return_aux)

        self.dino_layers = list(dino_layers)
        self.dino_embed_dim = int(dino_embed_dim)
        self.dino_patch_size = int(dino_patch_size)
        self.num_dino_levels = len(self.dino_layers)

        self.dino = FrozenDINOv3FeatureExtractor(
            dino_model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            out_layers=dino_layers,
            embed_dim=dino_embed_dim,
            patch_size=dino_patch_size,
            dino_normalize=dino_normalize,
            dino_intermediate_norm=dino_intermediate_norm,
        )

        self.dino_adapters = nn.ModuleList([
            DinoOutAdapter(
                in_ch=dino_embed_dim,
                out_ch=adapter_channels,
                bottleneck=adapter_bottleneck,
            )
            for _ in range(self.num_dino_levels)
        ])

        self.fuse = nn.Sequential(
            ConvBNAct(adapter_channels * self.num_dino_levels, fuse_channels, kernel_size=1, padding=0),
            ConvBNAct(fuse_channels, fuse_channels, kernel_size=3),
        )

        # DINO-only feature pyramid，对齐 HL_base encoder 输出通道和尺度。
        self.p4_proj = PyramidBlock(fuse_channels, 512)  # H/32
        self.p3_proj = PyramidBlock(fuse_channels, 256)  # H/16
        self.p2_proj = PyramidBlock(fuse_channels, 128)  # H/8
        self.p1_proj = PyramidBlock(fuse_channels, 64)   # H/4
        self.p0_proj = PyramidBlock(fuse_channels, 64)   # H/2

        # HL_base 同款 decoder。
        self.dec3 = ResidualDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=64, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=64, out_channels=64)

        # HL_base 同款 H/2 output head。
        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        self._print_trainable_summary()

    def _print_trainable_summary(self):
        total = 0
        trainable = 0
        dino_trainable = 0

        for p in self.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

        for p in self.dino.parameters():
            if p.requires_grad:
                dino_trainable += p.numel()

        print("--------------------------------------------------")
        print("🔧 DC_v2 Frozen DINO + HL-style Decoder")
        print(f"    - 总参数量:       {total / 1e6:.2f} M")
        print(f"    - 可训练参数量:   {trainable / 1e6:.2f} M")
        print(f"    - DINO分支可训练: {dino_trainable / 1e6:.2f} M")
        print("    - 说明: DINOv3 原始参数冻结，decoder 使用 HL_base 风格")
        print("--------------------------------------------------")

    def set_return_aux(self, flag: bool):
        self.return_aux = bool(flag)

    def forward(self, x):
        input_hw = x.shape[-2:]
        _, _, h, w = x.shape

        dino_out = self.dino(x)

        dino_feats = []

        for layer in self.dino_layers:
            if layer not in dino_out:
                raise RuntimeError(
                    f"DINO 输出中找不到 layer={layer}。"
                    f"当前 keys={list(dino_out.keys())}"
                )
            dino_feats.append(dino_out[layer])

        # DINO vits16 在 1024 输入下是 H/16 = 64×64。
        base_hw = (
            max(1, h // self.dino_patch_size),
            max(1, w // self.dino_patch_size),
        )

        adapted = []

        for feat, adapter in zip(dino_feats, self.dino_adapters):
            if feat.shape[-2:] != base_hw:
                feat = F.interpolate(
                    feat,
                    size=base_hw,
                    mode="bilinear",
                    align_corners=False,
                )

            feat = adapter(feat)
            adapted.append(feat)

        fused = torch.cat(adapted, dim=1)
        f16 = self.fuse(fused)

        # 构造与 HL_base 编码器输出一致的多尺度特征。
        p4 = self.p4_proj(
            f16,
            size=(max(1, h // 32), max(1, w // 32)),
        )
        p3 = self.p3_proj(
            f16,
            size=(max(1, h // 16), max(1, w // 16)),
        )
        p2 = self.p2_proj(
            f16,
            size=(max(1, h // 8), max(1, w // 8)),
        )
        p1 = self.p1_proj(
            f16,
            size=(max(1, h // 4), max(1, w // 4)),
        )
        p0 = self.p0_proj(
            f16,
            size=(max(1, h // 2), max(1, w // 2)),
        )

        # HL_base 风格 decoder。
        d3 = self.dec3(p4, p3)
        d2 = self.dec2(d3, p2)
        d1 = self.dec1(d2, p1)
        d0 = self.dec0(d1, p0)

        logits_half = self.out_head(d0)
        logits = F.interpolate(
            logits_half,
            size=input_hw,
            mode="bilinear",
            align_corners=False,
        )

        if self.return_aux:
            return {
                "final_logits": logits,
                "logits": logits,
                "logits_half": logits_half,
                "dino_feats": dino_feats,
                "adapted_feats": adapted,
                "p4": p4,
                "p3": p3,
                "p2": p2,
                "p1": p1,
                "p0": p0,
                "d3": d3,
                "d2": d2,
                "d1": d1,
                "d0": d0,
            }

        return logits

    def forward_train(self, x):
        logits = self.forward(x)

        if isinstance(logits, dict):
            logits = logits["final_logits"]

        return {
            "final_logits": logits,
            "base_logits": None,
            "dino_prior_logits": None,
        }