import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 基础模块
# =========================================================
class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
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


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# =========================================================
# 轻量 TCB：Topology Context Bridge Lite
# 在 bottleneck 以低维方式生成 topology prior
# =========================================================
class TopologyContextBridgeLite(nn.Module):
    def __init__(self, channels, reduced_ratio=4):
        super().__init__()
        reduced = max(channels // reduced_ratio, 64)

        self.reduce = ConvBNReLU(channels, reduced, kernel_size=1, padding=0)

        # Global continuity branch
        self.global_branch = nn.Sequential(
            ConvBNReLU(reduced, reduced, kernel_size=5, padding=2, groups=reduced),
            ConvBNReLU(reduced, reduced, kernel_size=3, padding=2, groups=reduced, dilation=2),
            ConvBNReLU(reduced, reduced, kernel_size=1, padding=0),
        )

        # Structural localization branch
        half = reduced // 2
        self.struct_h = nn.Sequential(
            nn.Conv2d(reduced, half, kernel_size=(1, 5), padding=(0, 2), bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )
        self.struct_v = nn.Sequential(
            nn.Conv2d(reduced, half, kernel_size=(5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )
        self.struct_fuse = ConvBNReLU(reduced, reduced, kernel_size=1, padding=0)

        # Fuse
        self.fuse = nn.Sequential(
            ConvBNReLU(reduced * 2, reduced, kernel_size=1, padding=0),
            ConvBNReLU(reduced, reduced, kernel_size=3, padding=1),
        )

        # 输出 topology prior
        self.topology_head = nn.Conv2d(reduced, 1, kernel_size=1)

        # 可选轻量残差增强 bottleneck
        self.refine_back = nn.Conv2d(reduced, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        xr = self.reduce(x)

        g_global = self.global_branch(xr)

        s_h = self.struct_h(xr)
        s_v = self.struct_v(xr)
        g_struct = self.struct_fuse(torch.cat([s_h, s_v], dim=1))

        fused = self.fuse(torch.cat([g_global, g_struct], dim=1))

        topo_prior = torch.sigmoid(self.topology_head(fused))

        # 初始接近 identity，更稳
        x = x + self.gamma * self.refine_back(fused)
        return x, topo_prior


# =========================================================
# 深层 TCSS：只在深层 skip 上做完整选择性传递
# =========================================================
class DeepTCSS(nn.Module):
    def __init__(self, skip_channels, guide_channels, reduced_ratio=4):
        super().__init__()
        reduced = max(skip_channels // reduced_ratio, 16)
        hidden = max(skip_channels // 8, 16)

        self.skip_reduce = ConvBNReLU(skip_channels, reduced, kernel_size=1, padding=0)
        self.guide_reduce = ConvBNReLU(guide_channels, reduced, kernel_size=1, padding=0)

        # semantic spatial gate: 输出 1 通道空间门
        self.sem_gate = nn.Sequential(
            ConvBNReLU(reduced * 2, reduced, kernel_size=3, padding=1),
            nn.Conv2d(reduced, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # topology spatial gate: 输出 1 通道空间门
        self.top_gate = nn.Sequential(
            ConvBNReLU(reduced + 1, reduced, kernel_size=3, padding=1),
            nn.Conv2d(reduced, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # channel gate: 输出 skip_channels 通道门
        self.channel_fc1 = nn.Conv2d(reduced * 2 + 1, hidden, kernel_size=1, bias=True)
        self.channel_fc2 = nn.Conv2d(hidden, skip_channels, kernel_size=1, bias=True)

        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, skip, guide, topo):
        """
        skip: [B, C_skip, H, W]
        guide: [B, C_guide, H, W]
        topo: [B, 1, H, W]
        """
        if topo.shape[-2:] != skip.shape[-2:]:
            topo = F.interpolate(topo, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        s_r = self.skip_reduce(skip)
        g_r = self.guide_reduce(guide)

        g_sem = self.sem_gate(torch.cat([s_r, g_r], dim=1))          # [B,1,H,W]
        g_top = self.top_gate(torch.cat([s_r, topo], dim=1))         # [B,1,H,W]

        pooled = torch.cat([s_r, g_r, topo], dim=1)
        pooled = F.adaptive_avg_pool2d(pooled, 1)
        g_ch = F.relu(self.channel_fc1(pooled), inplace=True)
        g_ch = torch.sigmoid(self.channel_fc2(g_ch))                 # [B,C_skip,1,1]

        selected = skip * g_sem * g_top * g_ch
        out = skip + self.alpha * selected
        return out


# =========================================================
# 中层轻量 topology gate
# =========================================================
class TopologySkipGateLite(nn.Module):
    def __init__(self, skip_channels, reduced_ratio=8):
        super().__init__()
        reduced = max(skip_channels // reduced_ratio, 16)

        self.skip_reduce = ConvBNReLU(skip_channels, reduced, kernel_size=1, padding=0)
        self.top_gate = nn.Sequential(
            ConvBNReLU(reduced + 1, reduced, kernel_size=3, padding=1),
            nn.Conv2d(reduced, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, skip, topo):
        if topo.shape[-2:] != skip.shape[-2:]:
            topo = F.interpolate(topo, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        s_r = self.skip_reduce(skip)
        g_top = self.top_gate(torch.cat([s_r, topo], dim=1))
        out = skip + self.beta * (skip * g_top)
        return out


# =========================================================
# Decoder block + DeepTCSS
# =========================================================
class UpDeepTCSS(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.tcss = DeepTCSS(skip_channels=skip_channels, guide_channels=out_channels)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip, topo):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        skip = self.tcss(skip, x, topo)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# =========================================================
# Decoder block + Lite Topology Gate
# =========================================================
class UpTopoLite(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.top_gate = TopologySkipGateLite(skip_channels=skip_channels)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip, topo):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        skip = self.top_gate(skip, topo)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# =========================================================
# 普通 U-Net up block
# =========================================================
class UpPlain(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


# =========================================================
# 主模型
# =========================================================
class Unet_v1(nn.Module):
    """
    轻量版 U-Net + TCB + 深层 TCSS
    - 文件名不变：Unet_v1.py
    - 模块名不变：Unet_v1
    - 默认只返回 logits
    """
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 通道改轻
        c1, c2, c3, c4, c5 = 48, 96, 192, 384, 768

        # Encoder
        self.inc = DoubleConv(n_channels, c1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c4, c5))

        # Bottleneck bridge
        self.tcb = TopologyContextBridgeLite(c5)

        # Decoder
        # 深层两级用完整 TCSS
        self.up1 = UpDeepTCSS(c5, c4, c4)
        self.up2 = UpDeepTCSS(c4, c3, c3)

        # 中层只用轻量 topology gate
        self.up3 = UpTopoLite(c3, c2, c2)

        # 浅层回归普通 U-Net
        self.up4 = UpPlain(c2, c1, c1)

        self.outc = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x):
        original_size = x.shape[-2:]

        # Encoder
        x1 = self.inc(x)      # [B,48,H,W]
        x2 = self.down1(x1)   # [B,96,H/2,W/2]
        x3 = self.down2(x2)   # [B,192,H/4,W/4]
        x4 = self.down3(x3)   # [B,384,H/8,W/8]
        x5 = self.down4(x4)   # [B,768,H/16,W/16]

        # TCB
        x5, topo_prior = self.tcb(x5)

        # Decoder
        x = self.up1(x5, x4, topo_prior)
        x = self.up2(x, x3, topo_prior)
        x = self.up3(x, x2, topo_prior)
        x = self.up4(x, x1)

        logits = self.outc(x)

        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)

        return logits


if __name__ == "__main__":
    model = Unet_v1(n_channels=3, n_classes=1)
    x = torch.randn(1, 3, 1024, 1024)
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)