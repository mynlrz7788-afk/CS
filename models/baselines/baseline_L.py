import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["baseline_L"]


class ConvBNReLU(nn.Module):
    """卷积 + BN + ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：DW 3x3 + PW 1x1"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
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


class DecoderBlock(nn.Module):
    """
    轻量解码块：
    上采样 -> 当前特征1x1压缩 -> skip特征1x1压缩 -> 拼接 -> 3x3卷积 -> 深度可分离卷积
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.in_proj = ConvBNReLU(in_channels, out_channels, kernel_size=1, padding=0)
        self.skip_proj = ConvBNReLU(skip_channels, out_channels, kernel_size=1, padding=0)

        self.fuse = nn.Sequential(
            ConvBNReLU(out_channels * 2, out_channels, kernel_size=3, padding=1),
            DepthwiseSeparableConv(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = self.in_proj(x)
        skip = self.skip_proj(skip)

        # 防止输入不是32倍数时出现1个像素的尺寸偏差
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([skip, x], dim=1)
        x = self.fuse(x)
        return x


def _get_resnet34(pretrained=True):
    """
    兼容不同 torchvision 版本：
    - 新版：weights=ResNet34_Weights.IMAGENET1K_V1
    - 旧版：pretrained=True
    """
    try:
        from torchvision.models import resnet34, ResNet34_Weights
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet34(weights=weights)
        return model
    except Exception:
        from torchvision import models
        try:
            model = models.resnet34(pretrained=pretrained)
        except TypeError:
            model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        return model


class baseline_L(nn.Module):
    """
    ResNet-34 encoder + Lite U-Net decoder

    设计目标：
    1. 比标准UNet更轻
    2. 比原始LinkNet解码器更强一些
    3. forward输出logits，不在模型内部做sigmoid
    4. 兼容你现有实验框架常见参数名：num_classes / n_classes / in_channels / n_channels
    """
    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        num_classes=None,
        in_channels=None,
        pretrained=True,
        **kwargs,
    ):
        super(baseline_L, self).__init__()

        # 兼容不同框架的参数命名
        if in_channels is not None:
            n_channels = in_channels
        if num_classes is not None:
            n_classes = num_classes

        self.n_channels = n_channels
        self.n_classes = n_classes

        encoder = _get_resnet34(pretrained=pretrained)

        # 如果不是3通道，先映射到3通道，方便继续使用ImageNet预训练权重
        if n_channels != 3:
            self.input_adapter = nn.Conv2d(n_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # ResNet-34 编码器
        self.stem = nn.Sequential(
            encoder.conv1,  # /2
            encoder.bn1,
            encoder.relu,
        )
        self.maxpool = encoder.maxpool  # /4
        self.layer1 = encoder.layer1    # 64,  /4
        self.layer2 = encoder.layer2    # 128, /8
        self.layer3 = encoder.layer3    # 256, /16
        self.layer4 = encoder.layer4    # 512, /32

        # 轻量 U-Net 风格解码器
        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)  # /32 -> /16
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)  # /16 -> /8
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=64,  out_channels=64)   # /8  -> /4
        self.dec1 = DecoderBlock(in_channels=64,  skip_channels=64,  out_channels=32)   # /4  -> /2

        # 最后恢复到原图大小
        self.final_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.seg_head = nn.Sequential(
            ConvBNReLU(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, n_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.input_adapter(x)

        # 编码器
        x0 = self.stem(x)                   # [B,  64, H/2,  W/2]
        x1 = self.layer1(self.maxpool(x0))  # [B,  64, H/4,  W/4]
        x2 = self.layer2(x1)                # [B, 128, H/8,  W/8]
        x3 = self.layer3(x2)                # [B, 256, H/16, W/16]
        x4 = self.layer4(x3)                # [B, 512, H/32, W/32]

        # 解码器
        d4 = self.dec4(x4, x3)              # [B, 256, H/16, W/16]
        d3 = self.dec3(d4, x2)              # [B, 128, H/8,  W/8]
        d2 = self.dec2(d3, x1)              # [B,  64, H/4,  W/4]
        d1 = self.dec1(d2, x0)              # [B,  32, H/2,  W/2]

        out = self.final_up(d1)             # [B,  32, H,    W]
        logits = self.seg_head(out)         # [B, n_classes, H, W]

        # 不进行 sigmoid，保持和你现有训练/测试框架一致
        return logits


if __name__ == '__main__':
    model = baseline_L(n_channels=3, n_classes=1, pretrained=False)
    x = torch.randn(2, 3, 1024, 1024)
    y = model(x)
    print('Input :', x.shape)
    print('Output:', y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_params / 1e6:.2f} M')
    print(f'Trainable params: {trainable_params / 1e6:.2f} M')
