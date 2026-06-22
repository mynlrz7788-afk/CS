import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class DecoderBlock(nn.Module):
    """
    普通 DecoderBlock
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


class Baseline(nn.Module):
    """
    AFDANet-style baseline:
    - 保留 AFDANet 的 ResNet34 encoder 和输入输出流程
    - 去掉 SEAF
    - 去掉 SGADecoder
    - 用普通 DecoderBlock 做恢复

    输出 logits（不做 sigmoid）
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

        # Encoder：与 AFDANet 保持一致
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1   # 64
        self.encoder2 = resnet.layer2   # 128
        self.encoder3 = resnet.layer3   # 256
        self.encoder4 = resnet.layer4   # 512

        # 普通 Decoder：替换原 SGADecoder
        self.decoder4 = DecoderBlock(filters[3], filters[2])  # 512 -> 256
        self.decoder3 = DecoderBlock(filters[2], filters[1])  # 256 -> 128
        self.decoder2 = DecoderBlock(filters[1], filters[0])  # 128 -> 64
        self.decoder1 = DecoderBlock(filters[0], filters[0])  # 64 -> 64

        # 输出头：尽量保持 AFDANet 风格
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        original_size = x.shape[-2:]

        # 与 AFDANet 一样：如果不是 1024，就先缩放到 1024
        if self.img_size != 1024:
            x = F.interpolate(x, size=1024, mode='bilinear', align_corners=True)

        # Encoder
        x = self.firstconv(x)      # [B,64,512,512]
        x = self.firstbn(x)
        x = self.firstrelu(x)

        x1 = self.firstmaxpool(x)  # [B,64,256,256]
        e1 = self.encoder1(x1)     # [B,64,256,256]
        e2 = self.encoder2(e1)     # [B,128,128,128]
        e3 = self.encoder3(e2)     # [B,256,64,64]
        e4 = self.encoder4(e3)     # [B,512,32,32]

        # Decoder：直接用普通 skip 恢复
        d4 = self.decoder4(e4)     # [B,256,64,64]
        d4 = d4 + e3

        d3 = self.decoder3(d4)     # [B,128,128,128]
        d3 = d3 + e2

        d2 = self.decoder2(d3)     # [B,64,256,256]
        d2 = d2 + e1

        d1 = self.decoder1(d2)     # [B,64,512,512]

        # Head
        out = self.finaldeconv1(d1)   # [B,32,1024,1024]
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        # 如果原输入不是 1024，再恢复回去
        if self.img_size != 1024:
            out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=True)

        return out


if __name__ == "__main__":
    model = Baseline(img_size=1024, num_classes=1, use_imagenet_pretrain=False)
    x = torch.randn(2, 3, 1024, 1024)
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)