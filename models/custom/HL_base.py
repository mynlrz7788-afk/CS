import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["HL_base"]


def _auto_padding(kernel_size, dilation=1):
    """支持 int / tuple kernel 的 same padding。"""
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
    """Efficient Channel Attention（高效通道注意力）"""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class ResidualDecoderBlock(nn.Module):
    """
    LinkNet/UNet 风格残差解码块：
    upsample decoder feature → concat skip → 2×ConvBNAct → ECA → shortcut → ReLU

    这是 HL_v2 中 decoder 的原始结构。HL_base 只保留这个编解码结构，
    不使用 MHR / high-low routing / expert bank。
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvBNAct(in_channels + skip_channels, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        )
        self.eca = ECALayer(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([x, skip], dim=1)
        out = self.fuse(f)
        out = self.eca(out)
        out = out + self.shortcut(f)
        return self.act(out)


class HL_base(nn.Module):
    """
    HL_base / R34-ED Baseline:
    纯编解码道路分割基线，用于验证 HL_v2 中 MHR-MoE 的真实贡献。

    设计原则：
    1) 保留 HL_v2 的 ResNet34 encoder；
    2) 保留 HL_v2 的 ResidualDecoderBlock 解码器；
    3) 直接使用原始 skip connection: x3, x2, x1, x0；
    4) 不使用 MHR、MutualGuidance、Expert Bank、Router；
    5) 输出头仍然在 H/2 上预测 logits，再上采样到原图，保持和 HL_v2 公平对比。

    Forward 尺度：
        x0: H/2,  64
        x1: H/4,  64
        x2: H/8,  128
        x3: H/16, 256
        x4: H/32, 512

        d3 = dec3(x4, x3)  # H/16, 256
        d2 = dec2(d3, x2)  # H/8,  128
        d1 = dec1(d2, x1)  # H/4,   96
        d0 = dec0(d1, x0)  # H/2,   64
    """
    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        num_classes=None,
        in_channels=None,
        pretrained=True,
        return_aux=False,
        **kwargs,
    ):
        super(HL_base, self).__init__()

        if in_channels is not None:
            n_channels = in_channels
        if num_classes is not None:
            n_classes = num_classes

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.return_aux = return_aux

        encoder = self._get_resnet34(pretrained=pretrained)

        if n_channels != 3:
            self.input_adapter = nn.Conv2d(n_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # Encoder: 与 HL_v2 保持一致。
        self.stem = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
        )
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        # Decoder: 与 HL_v2 完全一致，只是 skip 全部使用原始 encoder feature。
        self.dec3 = ResidualDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=64, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=64, out_channels=64)

        # Lightweight H/2 output head: 与 HL_v2 保持一致。
        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    @staticmethod
    def _get_resnet34(pretrained=True):
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
                model = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            return model

    def forward_features(self, x):
        input_size = x.shape[-2:]
        x_in = self.input_adapter(x)

        # Encoder
        x0 = self.stem(x_in)                    # H/2,  64
        x1 = self.layer1(self.maxpool(x0))      # H/4,  64
        x2 = self.layer2(x1)                    # H/8,  128
        x3 = self.layer3(x2)                    # H/16, 256
        x4 = self.layer4(x3)                    # H/32, 512

        # Decoder with direct skip connections.
        d3 = self.dec3(x4, x3)                  # H/16, 256
        d2 = self.dec2(d3, x2)                  # H/8,  128
        d1 = self.dec1(d2, x1)                  # H/4,   96
        d0 = self.dec0(d1, x0)                  # H/2,   64

        logits_half = self.out_head(d0)         # H/2, n_classes
        logits = F.interpolate(logits_half, size=input_size, mode="bilinear", align_corners=False)

        aux = {
            "logits_half": logits_half,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "d3": d3,
            "d2": d2,
            "d1": d1,
            "d0": d0,
        }

        return logits, aux

    def forward(self, x):
        logits, aux = self.forward_features(x)
        if self.return_aux:
            aux["fused_logits"] = logits
            return aux
        return logits


if __name__ == "__main__":
    model = HL_base(n_channels=3, n_classes=1, pretrained=False, return_aux=False)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("Input :", x.shape)
    if isinstance(y, dict):
        print("Output keys:", y.keys())
    else:
        print("Output:", y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Trainable params: {trainable_params / 1e6:.2f} M")
