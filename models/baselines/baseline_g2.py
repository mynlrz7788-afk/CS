import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -------------------------
# Basic blocks
# -------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, dilation=1, groups=1):
        super().__init__()
        if padding is None:
            padding = ((kernel_size - 1) // 2) * dilation
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
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
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


class AsymmetricDepthwiseConv(nn.Module):
    def __init__(self, channels, kernel_size=(1, 7)):
        super().__init__()
        pad_h = (kernel_size[0] - 1) // 2
        pad_w = (kernel_size[1] - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=(pad_h, pad_w),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 4

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.reduce(x)
        x = self.up(x)
        x = self.project(x)
        return x


class AuxHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.head(x)


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class LiteACFA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        assert channels % 4 == 0, "LiteACFA expects channels divisible by 4"
        branch_channels = channels // 4

        self.channel_gate = ChannelGate(channels)
        self.spatial_gate = SpatialGate()
        self.pre = ConvBNReLU(channels, channels, kernel_size=1)

        self.branch_h = AsymmetricDepthwiseConv(branch_channels, kernel_size=(1, 7))
        self.branch_v = AsymmetricDepthwiseConv(branch_channels, kernel_size=(7, 1))
        self.branch_d = DepthwiseSeparableConv(branch_channels, branch_channels, kernel_size=3, dilation=2)
        self.branch_g = DepthwiseSeparableConv(branch_channels, branch_channels, kernel_size=5, dilation=1)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x0 = self.channel_gate(x)
        x0 = self.spatial_gate(x0)
        x0 = self.pre(x0)

        xh, xv, xd, xg = torch.chunk(x0, 4, dim=1)

        bh = self.branch_h(xh)
        bv = self.branch_v(xv)
        bd = self.branch_d(xd)
        bg = self.branch_g(xg)

        out = torch.cat([bh, bv, bd, bg], dim=1)
        out = self.fuse(out)
        out = self.relu(out + x)
        return out


class MultiScaleStructExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre = ConvBNReLU(channels, channels, kernel_size=1)

        self.s1_k3 = DepthwiseSeparableConv(channels, channels, kernel_size=3)
        self.s1_k5 = DepthwiseSeparableConv(channels, channels, kernel_size=5)
        self.s1_d2 = DepthwiseSeparableConv(channels, channels, kernel_size=3, dilation=2)

        self.mix = ConvBNReLU(channels * 3, channels, kernel_size=1)

        self.s2_k3 = DepthwiseSeparableConv(channels, channels, kernel_size=3)
        self.s2_k5 = DepthwiseSeparableConv(channels, channels, kernel_size=5)
        self.s2_d2 = DepthwiseSeparableConv(channels, channels, kernel_size=3, dilation=2)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.pre(x)

        s1_3 = self.s1_k3(x)
        s1_5 = self.s1_k5(x)
        s1_d = self.s1_d2(x)

        mix = self.mix(torch.cat([s1_3, s1_5, s1_d], dim=1))

        s2_3 = self.s2_k3(mix)
        s2_5 = self.s2_k5(mix)
        s2_d = self.s2_d2(mix)

        out = self.fuse(torch.cat([s1_3 + s2_3, s1_5 + s2_5, s1_d + s2_d], dim=1))
        return out


class LinePriorGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.line_h = AsymmetricDepthwiseConv(channels, kernel_size=(1, 5))
        self.line_v = AsymmetricDepthwiseConv(channels, kernel_size=(5, 1))
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.line_h(x)
        v = self.line_v(x)
        return self.fuse(torch.cat([h, v], dim=1))


class SpatialSoftGate(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dw_local = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dw_context = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=2,
                dilation=2,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x0 = self.project(x)
        xl = self.dw_local(x0)
        xc = self.dw_context(x0)
        return self.fuse(torch.cat([x0, xl, xc], dim=1))


class RSSFBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.align_dec = ConvBNReLU(channels, channels, kernel_size=1)
        self.align_enc = ConvBNReLU(channels, channels, kernel_size=1)

        self.dec_ms = MultiScaleStructExtractor(channels)
        self.enc_ms = MultiScaleStructExtractor(channels)

        self.soft_gate = SpatialSoftGate(in_channels=channels * 4, out_channels=channels)
        self.line_gate = LinePriorGate(channels)

        self.post_fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, dec_feat, enc_feat):
        d0 = self.align_dec(dec_feat)
        e0 = self.align_enc(enc_feat)

        dm = self.dec_ms(d0)
        em = self.enc_ms(e0)

        gate_in = torch.cat([e0, d0, torch.abs(e0 - d0), e0 * d0], dim=1)
        soft_gate = self.soft_gate(gate_in)
        line_gate = self.line_gate(e0)
        gate = 0.7 * soft_gate + 0.3 * line_gate

        e_sel = em * gate

        out = self.post_fuse(torch.cat([dm, e_sel, torch.abs(dm - e_sel)], dim=1))
        out = self.relu(out + d0)
        return out


class FinalRefineHead(nn.Module):
    def __init__(self, in_channels=64, mid_channels=32, num_classes=1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x):
        x = self.up(x)
        x = self.refine(x)
        return x


class baseline_g2(nn.Module):
    def __init__(
        self,
        img_size=None,
        num_classes=1,
        use_imagenet_pretrain=True,
        deep_supervision=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes

        if use_imagenet_pretrain:
            try:
                backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            except Exception:
                backbone = models.resnet34(pretrained=True)
        else:
            try:
                backbone = models.resnet34(weights=None)
            except Exception:
                backbone = models.resnet34(pretrained=False)

        self.stem_conv = backbone.conv1
        self.stem_bn = backbone.bn1
        self.stem_relu = backbone.relu
        self.stem_pool = backbone.maxpool

        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.bottleneck_enhance = LiteACFA(512)

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        self.fuse3 = RSSFBlock(256)
        self.fuse2 = RSSFBlock(128)
        self.fuse1 = RSSFBlock(64)

        self.final_head = FinalRefineHead(in_channels=64, mid_channels=32, num_classes=num_classes)

        if self.deep_supervision:
            self.aux_head_d1 = AuxHead(64, num_classes=num_classes)
            self.aux_head_d2 = AuxHead(64, num_classes=num_classes)
            self.aux_head_d3 = AuxHead(128, num_classes=num_classes)

    def forward(self, x):
        input_size = x.shape[-2:]

        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        z4 = self.bottleneck_enhance(e4)

        x3 = self.decoder4(z4)
        x3 = self.fuse3(x3, e3)

        x2 = self.decoder3(x3)
        x2 = self.fuse2(x2, e2)

        x1 = self.decoder2(x2)
        x1 = self.fuse1(x1, e1)

        d1 = self.decoder1(x1)

        out = self.final_head(d1)
        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        if self.deep_supervision and self.training:
            aux_d1 = self.aux_head_d1(d1)
            aux_d2 = self.aux_head_d2(x1)
            aux_d3 = self.aux_head_d3(x2)

            if aux_d1.shape[-2:] != input_size:
                aux_d1 = F.interpolate(aux_d1, size=input_size, mode="bilinear", align_corners=False)
            if aux_d2.shape[-2:] != input_size:
                aux_d2 = F.interpolate(aux_d2, size=input_size, mode="bilinear", align_corners=False)
            if aux_d3.shape[-2:] != input_size:
                aux_d3 = F.interpolate(aux_d3, size=input_size, mode="bilinear", align_corners=False)

            return {
                "main": out,
                "aux_d1": aux_d1,
                "aux_d2": aux_d2,
                "aux_d3": aux_d3,
            }

        return out


if __name__ == "__main__":
    model = baseline_g2(
        img_size=1024,
        num_classes=1,
        use_imagenet_pretrain=False,
        deep_supervision=True,
    )

    x = torch.randn(2, 3, 1024, 1024)

    model.train()
    outputs = model(x)
    print("=== Train Mode ===")
    for k, v in outputs.items():
        print(k, v.shape)

    model.eval()
    with torch.no_grad():
        y = model(x)
    print("\n=== Eval Mode ===")
    print("main:", y.shape)
