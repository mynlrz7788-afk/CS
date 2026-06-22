# -*- coding: utf-8 -*-
"""
DU_v3.py

放置位置：SEG/models/custom/DU_v3.py

主模块：DU_v3

说明：
    前面基本沿用 DU_v4 的大模型适配思路：
    冻结 DINOv3 + Adapter + FAPM，道路高分辨率路径，道路任务适配路径，高低分辨率互导。

    后面的 decoder 改成和你的高指标 UNet 一样的 decoder：
    up1/up2/up3/up4 + conv_up1/conv_up2/conv_up3/conv_up4 + outc。
    不加 sigmoid，直接输出 logits。
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.custom.DinoRoadUNet import DinoEncoderForRoad


class DoubleConv(nn.Module):
    """(卷积 => BatchNorm => ReLU) * 2，和你上传的 Unet.py 保持一致。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None, act="relu"):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        act_layer = nn.LeakyReLU(0.1, inplace=True) if act == "leaky" else nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            act_layer,
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, act="relu"):
        super().__init__()
        act_layer = nn.LeakyReLU(0.1, inplace=True) if act == "leaky" else nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            act_layer,
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            act_layer,
        )

    def forward(self, x):
        return self.block(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class DirectionalConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.h = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels, bias=False)
        self.v = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels, bias=False)
        self.local = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.h(x), self.v(x), self.local(x)], dim=1))


class RoadOCL(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.main = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, stride=stride, act="leaky"),
            DepthwiseSeparableConv(out_ch, out_ch, stride=1, act="leaky"),
        )
        self.dir = DirectionalConv(out_ch)
        self.se = SqueezeExcitation(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.out_act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        y = self.main(x)
        y = y + self.dir(y)
        y = self.se(y)
        y = y + self.shortcut(x)
        return self.out_act(y)


class HighResolutionRoadPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.oc0 = RoadOCL(3, 32, stride=2)
        self.oc1 = RoadOCL(32, 64, stride=2)
        self.oc2 = RoadOCL(64, 128, stride=2)
        self.oc3 = RoadOCL(128, 256, stride=2)

    def forward(self, x):
        r0 = self.oc0(x)   # H/2, 32
        r1 = self.oc1(r0)  # H/4, 64
        r2 = self.oc2(r1)  # H/8, 128
        r3 = self.oc3(r2)  # H/16, 256
        return r0, r1, r2, r3


class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        b = edge.shape[0]
        flat = edge.view(b, -1)
        e_min = flat.min(dim=1)[0].view(b, 1, 1, 1)
        e_max = flat.max(dim=1)[0].view(b, 1, 1, 1)
        return (edge - e_min) / (e_max - e_min + 1e-6)


class PromptPyramid(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.s0 = ConvBNAct(in_ch, 32, 3, stride=2, act="leaky")
        self.s1 = ConvBNAct(32, 64, 3, stride=2, act="leaky")
        self.s2 = ConvBNAct(64, 128, 3, stride=2, act="leaky")
        self.s3 = ConvBNAct(128, 256, 3, stride=2, act="leaky")
        self.s4 = ConvBNAct(256, 512, 3, stride=2, act="leaky")

    def forward(self, x):
        x = self.s0(x)
        a1 = self.s1(x)
        a2 = self.s2(a1)
        a3 = self.s3(a2)
        a4 = self.s4(a3)
        return a1, a2, a3, a4


class RoadAdapterPath(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        self.edge = SobelEdge()
        self.rgb_pyr = PromptPyramid(in_ch=3)
        self.edge_pyr = PromptPyramid(in_ch=1)
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ) for c in channels
        ])

    def forward(self, x_raw):
        edge = self.edge(x_raw)
        rgb_feats = self.rgb_pyr(x_raw)
        edge_feats = self.edge_pyr(edge)
        return [self.fuse[i](torch.cat([r, e], dim=1)) for i, (r, e) in enumerate(zip(rgb_feats, edge_feats))]


class AdapterModulation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=True), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, f, a):
        if a.shape[-2:] != f.shape[-2:]:
            a = F.interpolate(a, size=f.shape[-2:], mode="bilinear", align_corners=False)
        return f * (1.0 + self.gate(a)) + self.proj(a)


class FuseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            SqueezeExcitation(channels)
        )

    def forward(self, f, r):
        if r.shape[-2:] != f.shape[-2:]:
            r = F.interpolate(r, size=f.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([f, r], dim=1)) + f


class LowFuseBlock(nn.Module):
    def __init__(self, high_ch, low_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(high_ch + low_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SqueezeExcitation(out_ch)
        )

    def forward(self, high_feat, low_feat):
        low_up = F.interpolate(low_feat, size=high_feat.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([high_feat, low_up], dim=1)) + high_feat


class LocalCrossAttention2D(nn.Module):
    def __init__(self, q_ch, kv_ch, out_ch=None, attn_dim=32, window_size=3):
        super().__init__()
        if out_ch is None:
            out_ch = q_ch
        self.attn_dim = attn_dim
        self.out_ch = out_ch
        self.window_size = window_size
        self.padding = window_size // 2
        self.q_proj = nn.Conv2d(q_ch, attn_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(kv_ch, attn_dim, 1, bias=False)
        self.v_proj = nn.Conv2d(kv_ch, out_ch, 1, bias=False)
        self.out_proj = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.scale = attn_dim ** -0.5

    def forward(self, q, kv):
        if kv.shape[-2:] != q.shape[-2:]:
            kv = F.interpolate(kv, size=q.shape[-2:], mode="bilinear", align_corners=False)
        b, _, h, w = q.shape
        ksize = self.window_size
        q_map = self.q_proj(q)
        k_map = self.k_proj(kv)
        v_map = self.v_proj(kv)
        k_unfold = F.unfold(k_map, kernel_size=ksize, padding=self.padding).view(b, self.attn_dim, ksize * ksize, h * w)
        v_unfold = F.unfold(v_map, kernel_size=ksize, padding=self.padding).view(b, self.out_ch, ksize * ksize, h * w)
        q_flat = q_map.view(b, self.attn_dim, 1, h * w)
        attn = (q_flat * k_unfold).sum(dim=1) * self.scale
        attn = torch.softmax(attn, dim=1)
        out = (v_unfold * attn.unsqueeze(1)).sum(dim=2).view(b, self.out_ch, h, w)
        return self.out_proj(out)


class RoadHLMGBlock(nn.Module):
    def __init__(self, low_ch, high_ch, attn_dim=32, window_size=3):
        super().__init__()
        self.low_to_high_proj = nn.Sequential(
            nn.Conv2d(low_ch, high_ch, 1, bias=False),
            nn.BatchNorm2d(high_ch),
            nn.ReLU(inplace=True),
        )
        self.high_gate = nn.Conv2d(high_ch, 1, 1)
        self.low_to_high_attn = LocalCrossAttention2D(high_ch, high_ch, high_ch, attn_dim, window_size)
        self.high_down_proj = nn.Sequential(
            nn.Conv2d(high_ch, low_ch, 1, bias=False),
            nn.BatchNorm2d(low_ch),
            nn.ReLU(inplace=True),
        )
        self.low_pred = nn.Conv2d(low_ch, 1, 1)
        self.high_to_low_attn = LocalCrossAttention2D(low_ch, low_ch, low_ch, attn_dim, window_size)
        self.low_refine = nn.Sequential(
            nn.Conv2d(low_ch * 2, low_ch, 1, bias=False),
            nn.BatchNorm2d(low_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_ch, low_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(low_ch),
            nn.ReLU(inplace=True),
            SqueezeExcitation(low_ch)
        )
        self.high_out = nn.Sequential(nn.Conv2d(high_ch, high_ch, 3, padding=1, bias=False), nn.BatchNorm2d(high_ch), nn.ReLU(inplace=True))
        self.low_out = nn.Sequential(nn.Conv2d(low_ch, low_ch, 3, padding=1, bias=False), nn.BatchNorm2d(low_ch), nn.ReLU(inplace=True))

    def forward(self, low, high):
        low_up = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
        low_up = self.low_to_high_proj(low_up)
        gate = torch.sigmoid(self.high_gate(low_up))
        high_gate = high * (1.0 + gate)
        high_delta = self.low_to_high_attn(high_gate, low_up)
        high_ref = self.high_out(high_gate + high_delta)
        p_low = torch.sigmoid(self.low_pred(low))
        uncert = 4.0 * p_low * (1.0 - p_low)
        high_down = F.interpolate(high_ref, size=low.shape[-2:], mode="bilinear", align_corners=False)
        high_down = self.high_down_proj(high_down)
        low_delta_attn = self.high_to_low_attn(low, high_down)
        low_delta = self.low_refine(torch.cat([low, low_delta_attn], dim=1))
        low_ref = self.low_out(low + uncert * low_delta)
        return low_ref, high_ref, p_low, uncert


class RoadHLMGNeck(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), attn_dim=32, window_size=3):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.low_fuse3 = LowFuseBlock(c3, c4, c3)
        self.low_fuse2 = LowFuseBlock(c2, c3, c2)
        self.block3 = RoadHLMGBlock(c4, c3, attn_dim, window_size)
        self.block2 = RoadHLMGBlock(c3, c2, attn_dim, window_size)
        self.block1 = RoadHLMGBlock(c2, c1, attn_dim, window_size)
        self.fuse_f3 = nn.Sequential(
            nn.Conv2d(c3 * 2, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
        )

    def forward(self, h1, h2, h3, l4, f2a, f3a):
        l3 = self.low_fuse3(f3a, l4)
        l2 = self.low_fuse2(f2a, l3)
        l4_ref, h3_ref, p4, u4 = self.block3(l4, h3)
        l3_ref, h2_ref, p3, u3 = self.block2(l3, h2)
        l2_ref, h1_ref, p2, u2 = self.block1(l2, h1)
        f1_final = h1_ref
        f2_final = h2_ref
        f3_final = self.fuse_f3(torch.cat([h3_ref, l3_ref], dim=1))
        f4_final = l4_ref
        return [f1_final, f2_final, f3_final, f4_final], {"p4": p4, "p3": p3, "p2": p2, "u4": u4, "u3": u3, "u2": u2}


class UNetSkipAdapter(nn.Module):
    """
    将 DU_v3 前面得到的四尺度特征转为标准 UNet decoder 需要的五级 skip。
    x1: 64,H  x2:128,H/2  x3:256,H/4  x4:512,H/8  x5:1024,H/16
    """
    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.x1_stem = DoubleConv(3, 64)
        self.f1_to_x1 = nn.Sequential(nn.Conv2d(c1, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.x1_fuse = DoubleConv(128, 64)
        self.r0_to_x2 = nn.Sequential(nn.Conv2d(32, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.f1_to_x2 = nn.Sequential(nn.Conv2d(c1, 128, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.x2_fuse = DoubleConv(256, 128)
        self.f1_to_x3 = nn.Sequential(nn.Conv2d(c1, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.r1_to_x3 = nn.Sequential(nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.x3_fuse = DoubleConv(512, 256)
        self.f2_to_x4 = nn.Sequential(nn.Conv2d(c2, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.r2_to_x4 = nn.Sequential(nn.Conv2d(128, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.x4_fuse = DoubleConv(1024, 512)
        self.f3_to_x5 = nn.Sequential(nn.Conv2d(c3, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.f4_to_x5 = nn.Sequential(nn.Conv2d(c4, 512, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.r3_to_x5 = nn.Sequential(nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.x5_fuse = DoubleConv(1280, 1024)

    def forward(self, x_raw, feats, high_feats):
        f1, f2, f3, f4 = feats
        r0, r1, r2, r3 = high_feats
        out_size = x_raw.shape[-2:]
        x1_base = self.x1_stem(x_raw)
        f1_h = F.interpolate(self.f1_to_x1(f1), size=out_size, mode="bilinear", align_corners=False)
        x1 = self.x1_fuse(torch.cat([x1_base, f1_h], dim=1))
        size_x2 = (out_size[0] // 2, out_size[1] // 2)
        r0_x2 = self.r0_to_x2(r0)
        f1_x2 = F.interpolate(self.f1_to_x2(f1), size=size_x2, mode="bilinear", align_corners=False)
        x2 = self.x2_fuse(torch.cat([r0_x2, f1_x2], dim=1))
        x3 = self.x3_fuse(torch.cat([self.f1_to_x3(f1), self.r1_to_x3(r1)], dim=1))
        x4 = self.x4_fuse(torch.cat([self.f2_to_x4(f2), self.r2_to_x4(r2)], dim=1))
        f3_x5 = self.f3_to_x5(f3)
        f4_x5 = F.interpolate(self.f4_to_x5(f4), size=f3_x5.shape[-2:], mode="bilinear", align_corners=False)
        r3_x5 = self.r3_to_x5(r3)
        x5 = self.x5_fuse(torch.cat([f3_x5, f4_x5, r3_x5], dim=1))
        return x1, x2, x3, x4, x5


class SameUNetDecoder(nn.Module):
    """和你上传的 Unet.py 的 decoder 保持同样结构。"""
    def __init__(self, n_classes=1):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x1, x2, x3, x4, x5):
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv_up1(x)
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up2(x)
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up3(x)
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up4(x)
        logits = self.outc(x)
        return logits, x


class EdgeHead(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, 1, 1)
        )

    def forward(self, x):
        return self.head(x)


class DU_v3(nn.Module):
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
        deform_num_heads: int = 12,
        n_points: int = 4,
        with_cp: bool = False,
        deep_supervision: bool = True,
        attn_dim: int = 32,
        window_size: int = 3,
    ):
        super().__init__()
        if out_channels is None:
            out_channels = [64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.imagenet_norm = imagenet_norm
        self.input_already_normalized = input_already_normalized
        self.dino_encoder = DinoEncoderForRoad(
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
        self.high_path = HighResolutionRoadPath()
        self.adapter_path = RoadAdapterPath(channels=tuple(out_channels))
        self.adapter_mods = nn.ModuleList([AdapterModulation(c) for c in out_channels])
        self.fuse1 = FuseBlock(out_channels[0])
        self.fuse2 = FuseBlock(out_channels[1])
        self.fuse3 = FuseBlock(out_channels[2])
        self.hl_neck = RoadHLMGNeck(tuple(out_channels), attn_dim=attn_dim, window_size=window_size)
        self.skip_adapter = UNetSkipAdapter(tuple(out_channels))
        self.decoder = SameUNetDecoder(n_classes=num_classes)
        self.low_head = nn.Conv2d(out_channels[3], num_classes, 1)
        self.edge_head = EdgeHead(in_ch=64)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def _normalize_for_dino(self, x):
        if not self.imagenet_norm or self.input_already_normalized:
            return x
        return (x - self.img_mean) / self.img_std

    def forward(self, x):
        x_raw = x
        x_dino = self._normalize_for_dino(x)
        f1, f2, f3, f4 = self.dino_encoder(x_dino)
        r0, r1, r2, r3 = self.high_path(x_raw)
        a1, a2, a3, a4 = self.adapter_path(x_raw)
        f1a = self.adapter_mods[0](f1, a1)
        f2a = self.adapter_mods[1](f2, a2)
        f3a = self.adapter_mods[2](f3, a3)
        f4a = self.adapter_mods[3](f4, a4)
        h1 = self.fuse1(f1a, r1)
        h2 = self.fuse2(f2a, r2)
        h3 = self.fuse3(f3a, r3)
        feats, aux = self.hl_neck(h1, h2, h3, f4a, f2a=f2a, f3a=f3a)
        f1_final, f2_final, f3_final, f4_final = feats
        x1, x2, x3, x4, x5 = self.skip_adapter(x_raw, feats, [r0, r1, r2, r3])
        final_logits, decoder_feat = self.decoder(x1, x2, x3, x4, x5)
        low_logits = self.low_head(f4_final)
        edge_logits = self.edge_head(x1)
        if (not self.training) or (not self.deep_supervision):
            return final_logits
        return {
            "final": final_logits,
            "low": low_logits,
            "edge": edge_logits,
            "p4": aux["p4"],
            "p3": aux["p3"],
            "p2": aux["p2"],
        }
