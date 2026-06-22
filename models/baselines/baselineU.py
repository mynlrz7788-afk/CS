import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


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
    """
    轻量辅助监督头（auxiliary head，辅助头）
    """
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


class baselineU(nn.Module):
    """
    baselineU:
    - ResNet34 encoder（编码器）
    - LinkNet-style decoder（解码器）
    - 只增加三个 deep supervision（深监督）分支：d1 / d2 / d3
    - 输出 logits（不做 sigmoid）

    训练时:
        model.train()
        outputs = model(imgs)
        outputs = {
            "main": main_logits,
            "aux_d1": aux_logits_d1,
            "aux_d2": aux_logits_d2,
            "aux_d3": aux_logits_d3
        }

    验证/测试时:
        model.eval()
        preds = model(imgs)
        preds 直接是 main logits（tensor）
    """

    def __init__(self, num_classes=1, use_imagenet_pretrain=True, deep_supervision=True):
        super().__init__()
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

        # Encoder（编码器）
        self.stem_conv = backbone.conv1
        self.stem_bn = backbone.bn1
        self.stem_relu = backbone.relu
        self.stem_pool = backbone.maxpool

        self.encoder1 = backbone.layer1   # 64, 1/4
        self.encoder2 = backbone.layer2   # 128, 1/8
        self.encoder3 = backbone.layer3   # 256, 1/16
        self.encoder4 = backbone.layer4   # 512, 1/32

        # Decoder（解码器）
        self.decoder4 = DecoderBlock(512, 256)  # 1/32 -> 1/16
        self.decoder3 = DecoderBlock(256, 128)  # 1/16 -> 1/8
        self.decoder2 = DecoderBlock(128, 64)   # 1/8  -> 1/4
        self.decoder1 = DecoderBlock(64, 64)    # 1/4  -> 1/2

        # Final head（主输出头）
        self.final_deconv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final_bn = nn.BatchNorm2d(32)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

        # 三个 deep supervision（深监督）分支：d1 / d2 / d3
        if self.deep_supervision:
            self.aux_head_d1 = AuxHead(64, num_classes=num_classes)   # 1/2
            self.aux_head_d2 = AuxHead(64, num_classes=num_classes)   # 1/4
            self.aux_head_d3 = AuxHead(128, num_classes=num_classes)  # 1/8

    def forward(self, x):
        input_size = x.shape[-2:]

        # Encoder
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = self.stem_relu(x)
        x = self.stem_pool(x)

        e1 = self.encoder1(x)   # [B, 64,  H/4,  W/4]
        e2 = self.encoder2(e1)  # [B, 128, H/8,  W/8]
        e3 = self.encoder3(e2)  # [B, 256, H/16, W/16]
        e4 = self.encoder4(e3)  # [B, 512, H/32, W/32]

        # Decoder + skip connection（跳跃连接）
        d4 = self.decoder4(e4) + e3   # [B, 256, H/16, W/16]
        d3 = self.decoder3(d4) + e2   # [B, 128, H/8,  W/8]
        d2 = self.decoder2(d3) + e1   # [B, 64,  H/4,  W/4]
        d1 = self.decoder1(d2)        # [B, 64,  H/2,  W/2]

        # Main output（主输出）
        out = self.final_deconv(d1)
        out = self.final_bn(out)
        out = self.final_relu(out)
        out = self.final_conv(out)

        if out.shape[-2:] != input_size:
            out = F.interpolate(out, size=input_size, mode="bilinear", align_corners=False)

        # 训练时：返回主输出 + 三个辅助输出
        if self.deep_supervision and self.training:
            aux_d1 = self.aux_head_d1(d1)
            aux_d2 = self.aux_head_d2(d2)
            aux_d3 = self.aux_head_d3(d3)

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
                "aux_d3": aux_d3
            }

        # 验证/测试时：只返回主输出
        return out


if __name__ == "__main__":
    model = baselineU(
        num_classes=1,
        use_imagenet_pretrain=False,
        deep_supervision=True
    )

    # train mode（训练模式）
    model.train()
    x = torch.randn(2, 3, 1024, 1024)
    outputs = model(x)

    print("=== Train Mode ===")
    for k, v in outputs.items():
        print(k, v.shape)

    # eval mode（验证/测试模式）
    model.eval()
    with torch.no_grad():
        y = model(x)

    print("\n=== Eval Mode ===")
    print("main:", y.shape)