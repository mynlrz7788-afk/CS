import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["HL_v1"]


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


class MultiScaleStripBlock(nn.Module):
    """
    Multi-scale Strip Direction Block（多尺度条带方向块）

    只保留在 MutualGuidance 内部使用，用于提取道路的方向结构。
    这里不是专家模块，而是高分辨率到低分辨率反向补偿时的结构提取器。
    """
    def __init__(self, channels, kernels=(3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(
                nn.Sequential(
                    ConvBNAct(channels, channels, kernel_size=(1, k), padding=(0, k // 2), groups=channels),
                    ConvBNAct(channels, channels, kernel_size=(k, 1), padding=(k // 2, 0), groups=channels),
                )
            )
        self.fuse = nn.Sequential(
            ConvBNAct(channels * len(kernels), channels, kernel_size=1, padding=0),
            ECALayer(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        out = self.fuse(torch.cat(outs, dim=1))
        return self.act(out + x)


class ResidualDecoderBlock(nn.Module):
    """
    LinkNet/UNet 风格残差解码块。

    该部分与 HL_base 保持一致：
    upsample decoder feature -> concat skip -> 2 个 ConvBNAct -> ECA -> shortcut -> ReLU。
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


class MutualGuidance(nn.Module):
    """
    High-Low Mutual Guidance（高低分辨率相互指导）

    1. Low -> High：
       低分辨率语义生成道路感知门控，筛选高分辨率细节。

    2. High -> Low：
       高分辨率道路结构反向补偿低分辨率语义。

    3. Mutual Fusion：
       融合 h_g、l_g、差异特征和交互特征，得到相互指导后的特征。
    """
    def __init__(self, channels):
        super().__init__()
        self.low_gate = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.high_structure = MultiScaleStripBlock(channels, kernels=(3, 5, 7))

        self.high_gate = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1, padding=0),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.fuse = ConvBNAct(channels * 4, channels, kernel_size=1, padding=0)

    def forward(self, h, l):
        # Low -> High Guidance
        g_l = self.low_gate(l)
        h_g = h + h * g_l

        # High -> Low Guidance
        s_h = self.high_structure(h_g)
        g_h = self.high_gate(torch.cat([s_h, l, torch.abs(s_h - l)], dim=1))
        l_g = l + s_h * g_h

        # Mutual Fusion
        f_mg = self.fuse(torch.cat([h_g, l_g, torch.abs(h_g - l_g), h_g * l_g], dim=1))
        return f_mg, h_g, l_g, g_l, g_h


class HLMGSkipBlock(nn.Module):
    """
    High-Low Mutual Guidance Skip Block（高低分辨率相互指导跳跃连接块）

    本模块用于替换普通 skip connection，但不使用专家库、不使用路由器、不使用 Top-k。

    输入：
        base_skip：当前阶段原始 skip，例如 x2 或 x1。
        x0, x1：高分辨率浅层特征。
        x2, x3：低分辨率深层特征。

    输出：
        adaptive_skip：在原始 skip 基础上加入高低互导后的残差增强。

    重要设计：
        输出采用 base_skip + alpha * delta 的形式。
        这样训练初期接近 HL_base，不会直接破坏原始稳定路径。
    """
    def __init__(self, out_channels, guide_channels=64, init_alpha=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.guide_channels = guide_channels

        # High features: x0(H/2,64) + x1(H/4,64)
        self.high_proj = ConvBNAct(64 + 64, guide_channels, kernel_size=1, padding=0)

        # Low features: x2(H/8,128) + x3(H/16,256)
        self.low_proj = ConvBNAct(128 + 256, guide_channels, kernel_size=1, padding=0)

        self.mutual_guidance = MutualGuidance(guide_channels)

        self.delta_proj = nn.Sequential(
            ConvBNAct(guide_channels, out_channels, kernel_size=1, padding=0),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
            ECALayer(out_channels),
        )

        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    @staticmethod
    def _resize(x, size):
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, base_skip, x0, x1, x2, x3, target_size=None, return_info=False):
        if target_size is None:
            target_size = base_skip.shape[-2:]

        base = self._resize(base_skip, target_size)

        x0_r = self._resize(x0, target_size)
        x1_r = self._resize(x1, target_size)
        x2_r = self._resize(x2, target_size)
        x3_r = self._resize(x3, target_size)

        h = self.high_proj(torch.cat([x0_r, x1_r], dim=1))
        l = self.low_proj(torch.cat([x2_r, x3_r], dim=1))

        f_mg, h_g, l_g, gate_l, gate_h = self.mutual_guidance(h, l)
        delta = self.delta_proj(f_mg)

        adaptive_skip = base + self.alpha * delta

        if return_info:
            info = {
                "alpha": self.alpha,
                "gate_low_to_high": gate_l,
                "gate_high_to_low": gate_h,
                "mutual_feature": f_mg,
                "high_guided": h_g,
                "low_guided": l_g,
            }
            return adaptive_skip, info
        return adaptive_skip


class HL_v1(nn.Module):
    """
    HL_v1：HL_base + High-Low Mutual Guidance

    这一版只保留：
    1. ResNet34 编码器；
    2. HL_base 的残差解码器；
    3. H/8 和 H/4 两个阶段的高低分辨率相互指导 skip；
    4. H/2 输出头。

    删除：
    1. 专家模块；
    2. 专家库；
    3. 路由器；
    4. Top-k 选择；
    5. 加权专家混合。

    默认启用：
        use_hlg8=True   替换 x2 skip。
        use_hlg4=True   替换 x1 skip。

    默认关闭：
        use_hlg16=False 不替换 x3，保证深层语义稳定。
        use_hlg2=False  不替换 x0，避免过多浅层纹理噪声。
    """
    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        num_classes=None,
        in_channels=None,
        pretrained=True,
        return_aux=False,
        guide_channels=64,
        init_alpha=0.1,
        use_hlg16=False,
        use_hlg8=True,
        use_hlg4=True,
        use_hlg2=False,
        **kwargs,
    ):
        super(HL_v1, self).__init__()

        if in_channels is not None:
            n_channels = in_channels
        if num_classes is not None:
            n_classes = num_classes

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.return_aux = return_aux

        self.use_hlg16 = use_hlg16
        self.use_hlg8 = use_hlg8
        self.use_hlg4 = use_hlg4
        self.use_hlg2 = use_hlg2

        encoder = self._get_resnet34(pretrained=pretrained)

        if n_channels != 3:
            self.input_adapter = nn.Conv2d(n_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # Encoder: 与 HL_base 保持一致。
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

        # Decoder: 与 HL_base 保持一致。
        self.dec3 = ResidualDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=64, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=64, out_channels=64)

        # High-Low Mutual Guidance Skip Blocks
        self.hlg16 = HLMGSkipBlock(
            out_channels=256,
            guide_channels=guide_channels,
            init_alpha=init_alpha,
        ) if use_hlg16 else None

        self.hlg8 = HLMGSkipBlock(
            out_channels=128,
            guide_channels=guide_channels,
            init_alpha=init_alpha,
        ) if use_hlg8 else None

        self.hlg4 = HLMGSkipBlock(
            out_channels=64,
            guide_channels=guide_channels,
            init_alpha=init_alpha,
        ) if use_hlg4 else None

        self.hlg2 = HLMGSkipBlock(
            out_channels=64,
            guide_channels=guide_channels,
            init_alpha=init_alpha,
        ) if use_hlg2 else None

        # Lightweight H/2 output head: 与 HL_base 保持一致。
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

        aux = {}

        # Decoder stage H/16
        if self.hlg16 is not None:
            skip3, info16 = self.hlg16(x3, x0, x1, x2, x3, target_size=x3.shape[-2:], return_info=True)
            aux["hlg16"] = info16
        else:
            skip3 = x3
        d3 = self.dec3(x4, skip3)               # H/16, 256

        # Decoder stage H/8
        if self.hlg8 is not None:
            skip2, info8 = self.hlg8(x2, x0, x1, x2, x3, target_size=x2.shape[-2:], return_info=True)
            aux["hlg8"] = info8
        else:
            skip2 = x2
        d2 = self.dec2(d3, skip2)               # H/8, 128

        # Decoder stage H/4
        if self.hlg4 is not None:
            skip1, info4 = self.hlg4(x1, x0, x1, x2, x3, target_size=x1.shape[-2:], return_info=True)
            aux["hlg4"] = info4
        else:
            skip1 = x1
        d1 = self.dec1(d2, skip1)               # H/4, 96

        # Decoder stage H/2
        if self.hlg2 is not None:
            skip0, info2 = self.hlg2(x0, x0, x1, x2, x3, target_size=x0.shape[-2:], return_info=True)
            aux["hlg2"] = info2
        else:
            skip0 = x0
        d0 = self.dec0(d1, skip0)               # H/2, 64

        logits_half = self.out_head(d0)         # H/2, n_classes
        logits = F.interpolate(logits_half, size=input_size, mode="bilinear", align_corners=False)

        aux.update({
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
        })

        return logits, aux

    def forward(self, x):
        logits, aux = self.forward_features(x)
        if self.return_aux:
            aux["fused_logits"] = logits
            return aux
        return logits


if __name__ == "__main__":
    model = HL_v1(n_channels=3, n_classes=1, pretrained=False, return_aux=False)
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
