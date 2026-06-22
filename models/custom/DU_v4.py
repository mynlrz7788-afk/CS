
# -*- coding: utf-8 -*-
"""
SEG/models/custom/DU_v4.py
主模块：DU_v4

冻结 DINOv3 + 道路高分辨率路径 + 道路适配路径 + 高低分辨率互导 + 正负残差纠偏。
该版本优先追求效果，不做轻量化。
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.custom.DinoRoadUNet import DinoEncoderForRoad


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, act='relu'):
        super().__init__()
        if p is None:
            p = k // 2
        if act == 'leaky':
            a = nn.LeakyReLU(0.1, inplace=True)
        elif act == 'gelu':
            a = nn.GELU()
        else:
            a = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            a
        )
    def forward(self, x):
        return self.block(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        hidden = max(4, ch // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, ch, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.net(x)


class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, s=1, act='relu'):
        super().__init__()
        a = nn.LeakyReLU(0.1, inplace=True) if act == 'leaky' else nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=s, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            a,
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            a
        )
    def forward(self, x):
        return self.net(x)


class DirectionalConv(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.h = nn.Conv2d(ch, ch, (1, 7), padding=(0, 3), groups=ch, bias=False)
        self.v = nn.Conv2d(ch, ch, (7, 1), padding=(3, 0), groups=ch, bias=False)
        self.l = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(ch * 3, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x):
        return self.fuse(torch.cat([self.h(x), self.v(x), self.l(x)], dim=1))


class RoadOCL(nn.Module):
    """道路版 OCL，提取边界、细路和方向结构。"""
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.main = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, stride, act='leaky'),
            DSConv(out_ch, out_ch, 1, act='leaky')
        )
        self.dir = DirectionalConv(out_ch)
        self.se = SqueezeExcitation(out_ch)
        self.short = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch)
        ) if in_ch != out_ch or stride != 1 else nn.Identity()
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        y = self.main(x)
        y = self.se(y + self.dir(y))
        return self.act(y + self.short(x))


class HighResolutionRoadPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.oc0 = RoadOCL(3, 32, 2)      # H/2
        self.oc1 = RoadOCL(32, 64, 2)     # H/4
        self.oc2 = RoadOCL(64, 128, 2)    # H/8
        self.oc3 = RoadOCL(128, 256, 2)   # H/16
    def forward(self, x):
        r0 = self.oc0(x)
        r1 = self.oc1(r0)
        r2 = self.oc2(r1)
        r3 = self.oc3(r2)
        return r0, r1, r2, r3


class SobelEdge(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('kx', kx, persistent=False)
        self.register_buffer('ky', ky, persistent=False)
    def forward(self, x):
        g = x.mean(dim=1, keepdim=True)
        gx = F.conv2d(g, self.kx, padding=1)
        gy = F.conv2d(g, self.ky, padding=1)
        e = torch.sqrt(gx * gx + gy * gy + 1e-6)
        b = e.shape[0]
        flat = e.view(b, -1)
        mn = flat.min(dim=1)[0].view(b, 1, 1, 1)
        mx = flat.max(dim=1)[0].view(b, 1, 1, 1)
        return (e - mn) / (mx - mn + 1e-6)


class PromptPyramid(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.s0 = ConvBNAct(in_ch, 32, 3, 2, act='leaky')
        self.s1 = ConvBNAct(32, 64, 3, 2, act='leaky')
        self.s2 = ConvBNAct(64, 128, 3, 2, act='leaky')
        self.s3 = ConvBNAct(128, 256, 3, 2, act='leaky')
        self.s4 = ConvBNAct(256, 512, 3, 2, act='leaky')
    def forward(self, x):
        x = self.s0(x)
        a1 = self.s1(x)
        a2 = self.s2(a1)
        a3 = self.s3(a2)
        a4 = self.s4(a3)
        return a1, a2, a3, a4


class RoadAdapterPath(nn.Module):
    """RGB prompt + 边缘 prompt，生成道路任务适配特征 A1-A4。"""
    def __init__(self, channels=(64, 128, 256, 512)):
        super().__init__()
        self.edge = SobelEdge()
        self.rgb = PromptPyramid(3)
        self.edg = PromptPyramid(1)
        self.fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c * 2, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True)
            ) for c in channels
        ])
    def forward(self, x):
        e = self.edge(x)
        rf = self.rgb(x)
        ef = self.edg(e)
        return [self.fuse[i](torch.cat([rf[i], ef[i]], dim=1)) for i in range(4)]


class AdapterModulation(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(ch, ch, 1), nn.Sigmoid())
        self.proj = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, f, a):
        if a.shape[-2:] != f.shape[-2:]:
            a = F.interpolate(a, size=f.shape[-2:], mode='bilinear', align_corners=False)
        return f * (1.0 + self.gate(a)) + self.proj(a)


class FuseBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            SqueezeExcitation(ch)
        )
    def forward(self, f, r):
        if r.shape[-2:] != f.shape[-2:]:
            r = F.interpolate(r, size=f.shape[-2:], mode='bilinear', align_corners=False)
        return self.net(torch.cat([f, r], dim=1)) + f


class LowFuseBlock(nn.Module):
    def __init__(self, high_ch, low_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(high_ch + low_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            SqueezeExcitation(out_ch)
        )
    def forward(self, h, l):
        l = F.interpolate(l, size=h.shape[-2:], mode='bilinear', align_corners=False)
        return self.net(torch.cat([h, l], dim=1)) + h


class LocalCrossAttention2D(nn.Module):
    """局部交叉注意力，默认 3×3 邻域。"""
    def __init__(self, q_ch, kv_ch, out_ch=None, attn_dim=32, window_size=3):
        super().__init__()
        out_ch = q_ch if out_ch is None else out_ch
        self.attn_dim = attn_dim
        self.out_ch = out_ch
        self.k = window_size
        self.pad = window_size // 2
        self.qp = nn.Conv2d(q_ch, attn_dim, 1, bias=False)
        self.kp = nn.Conv2d(kv_ch, attn_dim, 1, bias=False)
        self.vp = nn.Conv2d(kv_ch, out_ch, 1, bias=False)
        self.op = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.scale = attn_dim ** -0.5
    def forward(self, q, kv):
        if kv.shape[-2:] != q.shape[-2:]:
            kv = F.interpolate(kv, size=q.shape[-2:], mode='bilinear', align_corners=False)
        b, _, h, w = q.shape
        qmap = self.qp(q)
        kmap = self.kp(kv)
        vmap = self.vp(kv)
        ku = F.unfold(kmap, kernel_size=self.k, padding=self.pad).view(b, self.attn_dim, self.k * self.k, h * w)
        vu = F.unfold(vmap, kernel_size=self.k, padding=self.pad).view(b, self.out_ch, self.k * self.k, h * w)
        qf = qmap.view(b, self.attn_dim, 1, h * w)
        attn = (qf * ku).sum(dim=1) * self.scale
        attn = torch.softmax(attn, dim=1)
        out = (vu * attn.unsqueeze(1)).sum(dim=2).view(b, self.out_ch, h, w)
        return self.op(out)


class RoadHLMGBlock(nn.Module):
    """低到高强指导，高到低不确定性弱纠偏。"""
    def __init__(self, low_ch, high_ch, attn_dim=32, window_size=3):
        super().__init__()
        self.low_up = nn.Sequential(
            nn.Conv2d(low_ch, high_ch, 1, bias=False),
            nn.BatchNorm2d(high_ch),
            nn.ReLU(inplace=True)
        )
        self.gate = nn.Conv2d(high_ch, 1, 1)
        self.l2h = LocalCrossAttention2D(high_ch, high_ch, high_ch, attn_dim, window_size)
        self.h2l_proj = nn.Sequential(
            nn.Conv2d(high_ch, low_ch, 1, bias=False),
            nn.BatchNorm2d(low_ch),
            nn.ReLU(inplace=True)
        )
        self.low_pred = nn.Conv2d(low_ch, 1, 1)
        self.h2l = LocalCrossAttention2D(low_ch, low_ch, low_ch, attn_dim, window_size)
        self.low_ref = nn.Sequential(
            nn.Conv2d(low_ch * 2, low_ch, 1, bias=False),
            nn.BatchNorm2d(low_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(low_ch, low_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(low_ch),
            nn.ReLU(inplace=True),
            SqueezeExcitation(low_ch)
        )
        self.hout = nn.Sequential(nn.Conv2d(high_ch, high_ch, 3, padding=1, bias=False), nn.BatchNorm2d(high_ch), nn.ReLU(inplace=True))
        self.lout = nn.Sequential(nn.Conv2d(low_ch, low_ch, 3, padding=1, bias=False), nn.BatchNorm2d(low_ch), nn.ReLU(inplace=True))
    def forward(self, low, high):
        low_up = F.interpolate(low, size=high.shape[-2:], mode='bilinear', align_corners=False)
        low_up = self.low_up(low_up)
        g = torch.sigmoid(self.gate(low_up))
        high_g = high * (1.0 + g)
        high_ref = self.hout(high_g + self.l2h(high_g, low_up))
        p_low = torch.sigmoid(self.low_pred(low))
        u = 4.0 * p_low * (1.0 - p_low)
        hd = F.interpolate(high_ref, size=low.shape[-2:], mode='bilinear', align_corners=False)
        hd = self.h2l_proj(hd)
        low_delta = self.h2l(low, hd)
        low_delta = self.low_ref(torch.cat([low, low_delta], dim=1))
        low_ref = self.lout(low + u * low_delta)
        return low_ref, high_ref, p_low, u


class RoadHLMGNeck(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), attn_dim=32, window_size=3):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.lf3 = LowFuseBlock(c3, c4, c3)
        self.lf2 = LowFuseBlock(c2, c3, c2)
        self.b3 = RoadHLMGBlock(c4, c3, attn_dim, window_size)
        self.b2 = RoadHLMGBlock(c3, c2, attn_dim, window_size)
        self.b1 = RoadHLMGBlock(c2, c1, attn_dim, window_size)
        self.f3 = nn.Sequential(
            nn.Conv2d(c3 * 2, c3, 1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c3, 3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True)
        )
    def forward(self, h1, h2, h3, l4, f2a, f3a):
        l3 = self.lf3(f3a, l4)
        l2 = self.lf2(f2a, l3)
        l4r, h3r, p4, u4 = self.b3(l4, h3)
        l3r, h2r, p3, u3 = self.b2(l3, h2)
        l2r, h1r, p2, u2 = self.b1(l2, h1)
        f1 = h1r
        f2 = h2r
        f3 = self.f3(torch.cat([h3r, l3r], dim=1))
        f4 = l4r
        return [f1, f2, f3, f4], {'p4': p4, 'p3': p3, 'p2': p2, 'u4': u4, 'u3': u3, 'u2': u2}


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(in_ch, out_ch), ConvBNAct(out_ch, out_ch))
    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class DecoderWithFeature(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), num_classes=1):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.hr1 = nn.Sequential(nn.ConvTranspose2d(c1, 32, 2, stride=2), DoubleConv(32, 32))
        self.hr2 = nn.Sequential(nn.ConvTranspose2d(32, 32, 2, stride=2), DoubleConv(32, 32))
        self.head = nn.Conv2d(32, num_classes, 1)
    def forward(self, feats, out_size):
        f1, f2, f3, f4 = feats
        d3 = self.up3(f4, f3)
        d2 = self.up2(d3, f2)
        d1 = self.up1(d2, f1)
        d0 = self.hr1(d1)
        dout = self.hr2(d0)
        logits = self.head(dout)
        if logits.shape[-2:] != out_size:
            logits = F.interpolate(logits, size=out_size, mode='bilinear', align_corners=False)
            dout = F.interpolate(dout, size=out_size, mode='bilinear', align_corners=False)
        return logits, d1, dout


class PNRefinementHead(nn.Module):
    """正负残差纠偏头。"""
    def __init__(self, c1=64, c2=128, c4=512, num_classes=1):
        super().__init__()
        self.f2p = nn.Sequential(nn.Conv2d(c2, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True))
        self.q = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            SqueezeExcitation(c1)
        )
        self.pos = nn.Sequential(nn.Conv2d(c1, c1 // 2, 3, padding=1, bias=False), nn.BatchNorm2d(c1 // 2), nn.ReLU(inplace=True), nn.Conv2d(c1 // 2, num_classes, 1))
        self.neg = nn.Sequential(nn.Conv2d(c1, c1 // 2, 3, padding=1, bias=False), nn.BatchNorm2d(c1 // 2), nn.ReLU(inplace=True), nn.Conv2d(c1 // 2, num_classes, 1))
        self.gate = nn.Sequential(nn.Conv2d(c4, c4 // 4, 1, bias=False), nn.BatchNorm2d(c4 // 4), nn.ReLU(inplace=True), nn.Conv2d(c4 // 4, num_classes, 1))
    def forward(self, f1, f2, coarse, f4):
        f2u = F.interpolate(self.f2p(f2), size=f1.shape[-2:], mode='bilinear', align_corners=False)
        q = self.q(torch.cat([f1, f2u], dim=1))
        dpos = F.interpolate(self.pos(q), size=coarse.shape[-2:], mode='bilinear', align_corners=False)
        dneg = F.interpolate(self.neg(q), size=coarse.shape[-2:], mode='bilinear', align_corners=False)
        p = torch.sigmoid(coarse)
        u = 4.0 * p * (1.0 - p)
        g = torch.sigmoid(self.gate(f4))
        g = F.interpolate(g, size=coarse.shape[-2:], mode='bilinear', align_corners=False)
        final = coarse + u * g * dpos - p * (1.0 - g) * dneg
        return final, dpos, dneg, u, g


class EdgeHead(nn.Module):
    def __init__(self, in_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, 1, 1)
        )
    def forward(self, x):
        return self.net(x)


class DU_v4(nn.Module):
    def __init__(self, num_classes=1, dinov3_model='dinounet_s', pretrained_path=None,
                 out_channels: Optional[List[int]] = None, rank=256, img_size=1024,
                 freeze_backbone=True, imagenet_norm=True, input_already_normalized=False,
                 conv_inplane=64, deform_num_heads=12, n_points=4, with_cp=False,
                 deep_supervision=True, attn_dim=32, window_size=3):
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
            with_cp=with_cp
        )
        self.high_path = HighResolutionRoadPath()
        self.adapter_path = RoadAdapterPath(tuple(out_channels))
        self.mods = nn.ModuleList([AdapterModulation(c) for c in out_channels])
        self.fuse1 = FuseBlock(out_channels[0])
        self.fuse2 = FuseBlock(out_channels[1])
        self.fuse3 = FuseBlock(out_channels[2])
        self.hl_neck = RoadHLMGNeck(tuple(out_channels), attn_dim=attn_dim, window_size=window_size)
        self.decoder = DecoderWithFeature(tuple(out_channels), num_classes=num_classes)
        self.pn = PNRefinementHead(out_channels[0], out_channels[1], out_channels[3], num_classes=num_classes)
        self.low_head = nn.Conv2d(out_channels[3], num_classes, 1)
        self.edge_head = EdgeHead(out_channels[0])
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def _normalize_for_dino(self, x):
        if (not self.imagenet_norm) or self.input_already_normalized:
            return x
        return (x - self.img_mean) / self.img_std

    def forward(self, x):
        out_size = x.shape[-2:]
        x_raw = x
        x_dino = self._normalize_for_dino(x)

        f1, f2, f3, f4 = self.dino_encoder(x_dino)
        _, r1, r2, r3 = self.high_path(x_raw)
        a1, a2, a3, a4 = self.adapter_path(x_raw)

        f1a = self.mods[0](f1, a1)
        f2a = self.mods[1](f2, a2)
        f3a = self.mods[2](f3, a3)
        f4a = self.mods[3](f4, a4)

        h1 = self.fuse1(f1a, r1)
        h2 = self.fuse2(f2a, r2)
        h3 = self.fuse3(f3a, r3)
        l4 = f4a

        feats, aux = self.hl_neck(h1, h2, h3, l4, f2a=f2a, f3a=f3a)
        f1f, f2f, f3f, f4f = feats

        coarse, d1, dout = self.decoder(feats, out_size=out_size)
        final, dpos, dneg, u, groad = self.pn(f1f, f2f, coarse, f4f)

        if (not self.training) or (not self.deep_supervision):
            return final

        return {
            'final': final,
            'coarse': coarse,
            'low': self.low_head(f4f),
            'edge': self.edge_head(f1f),
            'delta_pos': dpos,
            'delta_neg': dneg,
            'u': u,
            'groad': groad,
            'p4': aux['p4'],
            'p3': aux['p3'],
            'p2': aux['p2'],
        }
