import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DecoderBlock(nn.Module):
    """
    LinkNet-style decoder block（LinkNet风格解码块）
    输入:  [B, Cin, H, W]
    输出:  [B, Cout, 2H, 2W]
    """
    def __init__(self, in_channels: int, out_channels: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.up(x)
        x = self.project(x)
        return x


class AuxHead(nn.Module):
    """
    Auxiliary head（辅助监督头）
    """
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class ECA(nn.Module):
    """
    Efficient Channel Attention（高效通道注意力）
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=k_size,
            padding=(k_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ChannelSemanticGate(nn.Module):
    """
    只保留通道语义门（只做语义粗筛选）
    guide feature（引导特征）生成通道权重，对 shallow feature（浅层特征）做轻量筛选。
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, guide: torch.Tensor) -> torch.Tensor:
        weight = self.mlp(self.pool(guide))
        return weight


class LiteStructurePreserve(nn.Module):
    """
    极轻量结构保真分支（只保一条最轻的结构支路）
    目标不是重建，而是避免语义筛选时把细长道路完全压没。
    """
    def __init__(self, channels: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            ECA(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class SSRMDelta(nn.Module):
    """
    SSRM-Delta（更保守的语义-结构跳连精炼模块）

    当前版本的目标不是“替代原始 skip”，而是“预测一个小的修正量 delta（增量修正）”。

    设计原则：
    1. 只保留通道语义门，不再使用空间门，减少对弱细路的误伤；
    2. 结构分支极轻量，只做结构保真，不做强重建；
    3. 输出的是 delta，而不是 refined skip；
    4. 外部连接方式使用: out = original_skip + gamma * delta。
    """
    def __init__(self, shallow_channels: int, guide_channels: int, inner_channels: int = None):
        super().__init__()
        inner_channels = inner_channels or shallow_channels

        self.shallow_proj = nn.Sequential(
            nn.Conv2d(shallow_channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.guide_proj = nn.Sequential(
            nn.Conv2d(guide_channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        self.pre_clean = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1, groups=inner_channels, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        self.channel_gate = ChannelSemanticGate(inner_channels)
        self.structure_preserve = LiteStructurePreserve(inner_channels)

        self.delta_head = nn.Sequential(
            nn.Conv2d(inner_channels * 2, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            ECA(inner_channels),
            nn.Conv2d(inner_channels, shallow_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(shallow_channels),
        )

    def forward(self, shallow: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        if guide.shape[-2:] != shallow.shape[-2:]:
            guide = F.interpolate(guide, size=shallow.shape[-2:], mode="bilinear", align_corners=False)

        s0 = self.shallow_proj(shallow)
        g0 = self.guide_proj(guide)

        s_pre = self.pre_clean(s0)

        # 只做通道语义筛选
        wc = self.channel_gate(g0)
        s_sem = s_pre * wc

        # 极轻量结构保真
        s_str = self.structure_preserve(s_sem)

        # 只输出修正量 delta，而不是直接重写 skip
        delta = self.delta_head(torch.cat([s_sem, s_str], dim=1))
        return delta


class baseline_g1(nn.Module):
    """
    baseline_g1（新调整版本）

    核心改动：
    1. 保留 ResNet34 encoder（编码器） + LinkNet-style decoder（解码器）;
    2. 保留三个 deep supervision（深监督）头;
    3. 只在 e2 -> d3 这一层加入更保守的 SSRM-Delta;
    4. 模块只预测 delta（修正量），不替代原始 skip;
    5. skip 融合方式改为: d3 = u3 + e2 + gamma * delta2。
    """
    def __init__(
        self,
        num_classes: int = 1,
        use_imagenet_pretrain: bool = True,
        deep_supervision: bool = True,
    ):
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

        self.encoder1 = backbone.layer1   # 64,  1/4
        self.encoder2 = backbone.layer2   # 128, 1/8
        self.encoder3 = backbone.layer3   # 256, 1/16
        self.encoder4 = backbone.layer4   # 512, 1/32

        # Decoder（解码器）
        self.decoder4 = DecoderBlock(512, 256)  # 1/32 -> 1/16
        self.decoder3 = DecoderBlock(256, 128)  # 1/16 -> 1/8
        self.decoder2 = DecoderBlock(128, 64)   # 1/8  -> 1/4
        self.decoder1 = DecoderBlock(64, 64)    # 1/4  -> 1/2

        # 只在 e2 -> u3 这一层插入更保守的 skip 修正模块
        self.ssrm2 = SSRMDelta(shallow_channels=128, guide_channels=128, inner_channels=128)

        # gamma（修正强度）
        # 初始化为 0，让模型先从原始 baselineU 开始学，再逐步学习是否需要 skip 修正。
        self.gamma2 = nn.Parameter(torch.tensor(0.0))

        # Final head（主输出头）
        self.final_deconv = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.final_bn = nn.BatchNorm2d(32)
        self.final_relu = nn.ReLU(inplace=True)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=3, padding=1)

        # 三个 deep supervision（深监督）分支
        if self.deep_supervision:
            self.aux_head_d1 = AuxHead(64, num_classes=num_classes)   # 1/2
            self.aux_head_d2 = AuxHead(64, num_classes=num_classes)   # 1/4
            self.aux_head_d3 = AuxHead(128, num_classes=num_classes)  # 1/8

    def forward(self, x: torch.Tensor):
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

        # Decoder stage 4
        u4 = self.decoder4(e4)      # [B,256,H/16,W/16]
        d4 = u4 + e3

        # Decoder stage 3 + conservative delta refinement on e2
        u3 = self.decoder3(d4)      # [B,128,H/8,W/8]
        delta2 = self.ssrm2(e2, u3) # [B,128,H/8,W/8]
        d3 = u3 + e2 + self.gamma2 * delta2

        # Decoder stage 2（恢复原始 baselineU skip）
        u2 = self.decoder2(d3)      # [B,64,H/4,W/4]
        d2 = u2 + e1

        # Decoder stage 1
        d1 = self.decoder1(d2)      # [B,64,H/2,W/2]

        # Main output
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
                "aux_d3": aux_d3,
            }

        # 验证/测试时：只返回主输出
        return out


if __name__ == "__main__":
    model = baseline_g1(
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
    print("gamma2:", model.gamma2.item())

    model.eval()
    with torch.no_grad():
        y = model(x)
    print("\n=== Eval Mode ===")
    print("main:", y.shape)
