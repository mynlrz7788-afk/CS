import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange
from torchvision import models

nonlinearity = partial(F.relu, inplace=True)


# =========================
# 普通解码器：替换原 SGADecoder
# =========================
class _DecoderBlock(nn.Module):
    """
    LinkNet-style DecoderBlock（普通解码器）
    输入:  [B, Cin, H, W]
    输出:  [B, Cout, 2H, 2W]
    """
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


# =========================
# 以下全部保留 AFDANet 原始 SEAF 相关模块
# =========================
class ECA(nn.Module):
    def __init__(self, channel, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cfusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat([
            x[:, :, 0::2, 0::2],
            x[:, :, 1::2, 0::2],
            x[:, :, 0::2, 1::2],
            x[:, :, 1::2, 1::2]
        ], dim=1)
        x = self.cfusion(x)
        x = self.bn(x)
        return x


class SADBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Focus_downsample = Focus(in_channels=in_channels, out_channels=out_channels)
        self.dwconv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            groups=in_channels
        )
        self.dwconv_downsample = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=7, stride=2, padding=3,
            groups=out_channels
        )
        self.dwconv_act = nn.GELU()
        self.dwconv_bn = nn.BatchNorm2d(out_channels)
        self.maxpool_downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool_bn = nn.BatchNorm2d(out_channels)
        self.Aggregation = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        c = x
        x = self.dwconv(x)
        m = x

        c = self.Focus_downsample(c)

        x = self.dwconv_downsample(x)
        x = self.dwconv_act(x)
        x = self.dwconv_bn(x)

        m = self.maxpool_downsample(m)
        m = self.maxpool_bn(m)

        x = torch.cat([c, x, m], dim=1)
        x = self.Aggregation(x)
        return x


class PCSA(nn.Module):
    def __init__(self, dim, head_num, patch_size=8, kernel_sizes=[3, 5, 7, 9]):
        super().__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_channels = self.dim // 4

        gc = self.group_channels
        self.dwc1 = nn.Conv1d(gc, gc, kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2, groups=gc)
        self.dwc2 = nn.Conv1d(gc, gc, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2, groups=gc)
        self.dwc3 = nn.Conv1d(gc, gc, kernel_size=kernel_sizes[2], padding=kernel_sizes[2] // 2, groups=gc)
        self.dwc4 = nn.Conv1d(gc, gc, kernel_size=kernel_sizes[3], padding=kernel_sizes[3] // 2, groups=gc)

        self.sa_gate = nn.Sigmoid()
        self.gn_h = nn.GroupNorm(4, dim)
        self.gn_w = nn.GroupNorm(4, dim)

        self.identity = nn.Identity()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False, groups=dim)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=False, groups=dim)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=False, groups=dim)

        self.sig = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(kernel_size=(patch_size, patch_size), stride=patch_size)

    def forward(self, x):
        # Directional Spatial Attention
        b, c, h_, w_ = x.size()
        x2 = x.clone()

        x_h = x.mean(dim=3)
        feat1_h, feat2_h, feat3_h, feat4_h = torch.split(x_h, self.group_channels, dim=1)

        x_w = x.mean(dim=2)
        feat1_w, feat2_w, feat3_w, feat4_w = torch.split(x_w, self.group_channels, dim=1)

        x_h_att = self.sa_gate(self.gn_h(torch.cat((
            self.dwc1(feat1_h),
            self.dwc2(feat2_h),
            self.dwc3(feat3_h),
            self.dwc4(feat4_h),
        ), dim=1)))
        x_h_att = x_h_att.view(b, c, h_, 1)

        x_w_att = self.sa_gate(self.gn_w(torch.cat((
            self.dwc1(feat1_w),
            self.dwc2(feat2_w),
            self.dwc3(feat3_w),
            self.dwc4(feat4_w),
        ), dim=1)))
        x_w_att = x_w_att.view(b, c, 1, w_)

        x = x * x_h_att * x_w_att

        # Multi-Head Contextual Channel Attention
        y = self.avgpool(x2)
        y = self.identity(y)
        _, _, h_, w_ = y.size()

        y = self.norm(y)
        q = self.q(y)
        k = self.k(y)
        v = self.v(y)

        q = rearrange(
            q, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
            head_num=int(self.head_num), head_dim=int(self.head_dim)
        )
        k = rearrange(
            k, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
            head_num=int(self.head_num), head_dim=int(self.head_dim)
        )
        v = rearrange(
            v, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
            head_num=int(self.head_num), head_dim=int(self.head_dim)
        )

        c_att = q @ k.transpose(-2, -1) * self.scaler
        c_att = c_att.softmax(dim=-1)
        c_att = c_att @ v
        c_att = rearrange(
            c_att, 'b head_num head_dim (h w) -> b (head_num head_dim) h w',
            h=int(h_), w=int(w_)
        )

        c_att = c_att.mean((2, 3), keepdim=True)
        c_att = self.sig(c_att)
        return x + c_att * x2


class MS_DWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(
                channels, channels,
                kernel_size=scale[i],
                padding=scale[i] // 2,
                groups=channels
            )
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features),
        )
        self.dwconv = MS_DWConv(hidden_features)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(hidden_features)

        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) + x
        x = self.norm(self.act(x))
        x = self.fc2(x)
        return x


class SEAFModule(nn.Module):
    def __init__(self, in_channels, out_channels, mlp_ratio=4):
        super().__init__()
        self.sadb = SADBlock(in_channels, out_channels)
        self.norm1 = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()
        self.p1 = nn.Parameter(torch.tensor(1.0))
        self.p2 = nn.Parameter(torch.tensor(1.0))
        self.psca = PCSA(out_channels, head_num=8, patch_size=8)
        self.norm2 = nn.GroupNorm(1, out_channels)
        self.mlp = MLP(
            in_features=out_channels,
            hidden_features=int(out_channels * mlp_ratio)
        )

    def forward(self, x1, x2):
        probs = torch.softmax(torch.stack([self.p1, self.p2]), dim=0)
        p1, p2 = probs[0], probs[1]

        x1 = self.act(self.norm1(self.sadb(x1)))
        x1 = p1 * x1
        x2 = p2 * x2

        x = x1 + x2
        shortcut = x.clone()
        x = self.psca(x) + shortcut
        x = self.mlp(self.norm2(x)) + x
        return x


# =========================
# 新主模型：保留 SEAF，换普通解码器
# =========================
class Baseline_v(nn.Module):
    """
    文件名固定: Baseline_v.py
    主类名固定: Baseline_v

    当前版本:
    普通 CNN encoder（ResNet34） + 原始 SEAF + 普通 DecoderBlock
    """
    def __init__(self, img_size=1024, num_classes=1, use_imagenet_pretrain=True):
        super().__init__()
        self.img_size = img_size
        filters = [64, 128, 256, 512]

        if use_imagenet_pretrain:
            try:
                resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            except Exception:
                resnet = models.resnet34(pretrained=True)
        else:
            try:
                resnet = models.resnet34(weights=None)
            except Exception:
                resnet = models.resnet34(pretrained=False)

        # Encoder：保持普通 ResNet34
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # 保留原始 SEAF
        self.seaf1 = SEAFModule(filters[0], filters[0])
        self.seaf2 = SEAFModule(filters[0], filters[1])
        self.seaf3 = SEAFModule(filters[1], filters[2])
        self.seaf4 = SEAFModule(filters[2], filters[3])

        # 普通 decoder：替换原 SGADecoder
        self.decoder4 = _DecoderBlock(filters[3], filters[2])  # 512 -> 256
        self.decoder3 = _DecoderBlock(filters[2], filters[1])  # 256 -> 128
        self.decoder2 = _DecoderBlock(filters[1], filters[0])  # 128 -> 64
        self.decoder1 = _DecoderBlock(filters[0], filters[0])  # 64 -> 64

        # 输出头保持简单
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        if self.img_size != 1024:
            x = F.interpolate(x, size=1024, mode='bilinear', align_corners=True)

        # Encoder
        x = self.firstconv(x)     # 64, 512x512
        x = self.firstbn(x)
        x = self.firstrelu(x)

        x1 = self.firstmaxpool(x)     # 64, 256x256
        encoder1 = self.encoder1(x1)  # 64, 256x256
        f1 = self.seaf1(x, encoder1)  # 64, 256x256

        encoder2 = self.encoder2(encoder1)  # 128, 128x128
        f2 = self.seaf2(f1, encoder2)       # 128, 128x128

        encoder3 = self.encoder3(encoder2)  # 256, 64x64
        f3 = self.seaf3(f2, encoder3)       # 256, 64x64

        encoder4 = self.encoder4(encoder3)  # 512, 32x32
        f4 = self.seaf4(f3, encoder4)       # 512, 32x32
        f4 = encoder4 + f4

        # 普通 Decoder
        d4 = self.decoder4(f4)   # 256, 64x64
        d4 = d4 + f3

        d3 = self.decoder3(d4)   # 128, 128x128
        d3 = d3 + f2

        d2 = self.decoder2(d3)   # 64, 256x256
        d2 = d2 + f1

        d1 = self.decoder1(d2)   # 64, 512x512

        out = self.finaldeconv1(d1)  # 32, 1024x1024
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        if self.img_size != 1024:
            out = F.interpolate(out, size=self.img_size, mode='bilinear', align_corners=True)

        return out


if __name__ == '__main__':
    model = Baseline_v(img_size=1024, num_classes=1, use_imagenet_pretrain=False)
    x = torch.randn(2, 3, 1024, 1024)
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)