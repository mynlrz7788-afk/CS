"""
baselineU0L.py

Low-FLOPs baseline derived from the current medium baselineU0.

Key changes relative to the medium baselineU0:
1) Keep encoder width unchanged: 44 -> 88 -> 176 -> 352 -> 704
2) Compress all skip features with 1x1 conv before concatenation
3) Use depthwise separable conv blocks in high-resolution decoder stages d1 / d0
4) Use a single conv block at d0 to further reduce FLOPs
5) No deep supervision

Main module:
    baselineU0L
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSConv(nn.Module):
    """Depthwise separable conv: DW 3x3 + PW 1x1"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, padding=1,
                groups=in_channels, bias=bias
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSDoubleConv(nn.Module):
    """(Depthwise separable conv) * 2"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            DSConv(in_channels, out_channels, bias=bias),
            DSConv(out_channels, out_channels, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSSingleConv(nn.Module):
    """Single depthwise separable conv"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.block = DSConv(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool + DoubleConv"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SkipCompress(nn.Module):
    """1x1 conv to compress skip channels before concatenation"""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = False) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """
    ConvTranspose2d upsample + compressed skip + concat + custom conv block
    """
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        skip_compress_channels: int,
        out_channels: int,
        conv_block: nn.Module,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, bias=bias
        )
        self.skip_compress = SkipCompress(skip_channels, skip_compress_channels, bias=bias)
        self.conv = conv_block(out_channels + skip_compress_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        skip = self.skip_compress(skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutHead(nn.Module):
    """Final 1x1 segmentation head"""
    def __init__(self, in_channels: int, num_classes: int = 1) -> None:
        super().__init__()
        self.out = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(x)


class baselineU0(nn.Module):
    """
    Low-FLOPs baseline.

    Encoder width:
        44 -> 88 -> 176 -> 352 -> 704

    Decoder policy:
        d3 / d2 : standard DoubleConv
        d1      : depthwise separable double conv
        d0      : depthwise separable single conv

    Skip compression:
        e3: 352 -> 176
        e2: 176 -> 88
        e1:  88 -> 44
        e0:  44 -> 22
    """
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 44,
        return_features: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.return_features = return_features

        c0 = base_channels
        c1 = c0 * 2
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2

        s0 = c0 // 2
        s1 = c1 // 2
        s2 = c2 // 2
        s3 = c3 // 2

        self.enc0 = DoubleConv(in_channels, c0, bias=bias)
        self.enc1 = DownBlock(c0, c1, bias=bias)
        self.enc2 = DownBlock(c1, c2, bias=bias)
        self.enc3 = DownBlock(c2, c3, bias=bias)

        self.bottleneck = DownBlock(c3, c4, bias=bias)

        self.up3 = UpBlock(c4, c3, s3, c3, DoubleConv, bias=bias)
        self.up2 = UpBlock(c3, c2, s2, c2, DoubleConv, bias=bias)
        self.up1 = UpBlock(c2, c1, s1, c1, DSDoubleConv, bias=bias)
        self.up0 = UpBlock(c1, c0, s0, c0, DSSingleConv, bias=bias)

        self.head_main = OutHead(c0, num_classes=num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _upsample_to_input(x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)
        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:
        input_hw = x.shape[-2:]

        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        b = self.bottleneck(e3)

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, e0)

        main = self.head_main(d0)
        main = self._upsample_to_input(main, input_hw)

        if not self.return_features:
            return main

        return {
            "main": main,
            "features": {
                "e0": e0,
                "e1": e1,
                "e2": e2,
                "e3": e3,
                "b": b,
                "d3": d3,
                "d2": d2,
                "d1": d1,
                "d0": d0,
            }
        }


if __name__ == "__main__":
    model = baselineU0(
        in_channels=3,
        num_classes=1,
        base_channels=44,
        return_features=True,
    )

    x = torch.randn(2, 3, 512, 512)
    y = model(x)

    print("=== baselineU sanity check ===")
    if isinstance(y, dict):
        for k, v in y.items():
            if isinstance(v, dict):
                print(f"{k}:")
                for kk, vv in v.items():
                    print(f"  {kk}: {tuple(vv.shape)}")
            else:
                print(f"{k}: {tuple(v.shape)}")
    else:
        print(tuple(y.shape))
