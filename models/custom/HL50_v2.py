import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["HL50_v2"]


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


class DirectionalConvBlock(nn.Module):
    """
    道路友好的方向卷积块：
    1×5 → 5×1 与 5×1 → 1×5 两条方向路径。
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        mid = max(out_channels // 2, 8)
        self.branch_hv = nn.Sequential(
            ConvBNAct(in_channels, mid, kernel_size=(1, 5), stride=(1, stride), padding=(0, 2)),
            ConvBNAct(mid, out_channels, kernel_size=(5, 1), stride=(stride, 1), padding=(2, 0)),
        )
        self.branch_vh = nn.Sequential(
            ConvBNAct(in_channels, mid, kernel_size=(5, 1), stride=(stride, 1), padding=(2, 0)),
            ConvBNAct(mid, out_channels, kernel_size=(1, 5), stride=(1, stride), padding=(0, 2)),
        )
        self.fuse = ConvBNAct(out_channels * 2, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        a = self.branch_hv(x)
        b = self.branch_vh(x)
        return self.fuse(torch.cat([a, b], dim=1))


class MultiScaleStripBlock(nn.Module):
    """
    Multi-scale Strip Direction Block（多尺度条带方向块）
    用于建模道路的长条状方向连通性。
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
    LinkNet/UNet 风格残差解码块：
    upsample decoder feature → concat skip → 2×ConvBNAct → ECA → shortcut → ReLU
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


class FrequencyDetailExpert(nn.Module):
    """
    Frequency-enhanced Fine Detail Expert（频率增强细道路细节专家）

    目标：
    - 强化低对比度细道路、弱边界和细路纹理；
    - 使用低频估计残差 F_high = F - Up(AvgPool(F)) 提取高频；
    - 使用可学习 Laplacian-style depthwise conv 强化边缘。
    """
    def __init__(self, channels):
        super().__init__()
        self.high_proj = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )
        self.lap = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self._init_laplacian()

        self.lap_proj = ConvBNAct(channels, channels, kernel_size=1, padding=0)
        self.local = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1, padding=0),
            ECALayer(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def _init_laplacian(self):
        kernel = torch.tensor(
            [[0.0, -1.0, 0.0],
             [-1.0, 4.0, -1.0],
             [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        )
        with torch.no_grad():
            self.lap.weight.zero_()
            for c in range(self.lap.weight.shape[0]):
                self.lap.weight[c, 0, :, :] = kernel
        self.lap.weight.requires_grad = True

    def forward(self, f, h_g=None, l_g=None):
        low = F.avg_pool2d(f, kernel_size=2, stride=2, ceil_mode=False)
        low = F.interpolate(low, size=f.shape[-2:], mode="bilinear", align_corners=False)
        high = f - low

        y_local = self.local(f)
        y_high = self.high_proj(high)
        y_lap = self.lap_proj(self.lap(f))

        out = self.fuse(torch.cat([y_local, y_high, y_lap], dim=1))
        return self.act(f + out)


class StripDirectionExpert(nn.Module):
    """
    Strip Directional Connectivity Expert（条带方向连通专家）

    目标：
    - 使用 1×3/3×1, 1×5/5×1, 1×7/7×1 多尺度条带卷积；
    - 强化道路方向延展、交叉口和断裂连接。
    """
    def __init__(self, channels, kernels=(3, 5, 7), use_dilation_branch=True):
        super().__init__()
        self.strip = MultiScaleStripBlock(channels, kernels=kernels)
        self.use_dilation_branch = use_dilation_branch
        if use_dilation_branch:
            self.dilated = nn.Sequential(
                ConvBNAct(channels, channels, kernel_size=3, dilation=2, groups=channels),
                ConvBNAct(channels, channels, kernel_size=1, padding=0),
            )
            fuse_in = channels * 2
        else:
            fuse_in = channels

        self.fuse = nn.Sequential(
            ConvBNAct(fuse_in, channels, kernel_size=1, padding=0),
            ECALayer(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, f, h_g=None, l_g=None):
        y_strip = self.strip(f)
        if self.use_dilation_branch:
            y_dil = self.dilated(f)
            y = self.fuse(torch.cat([y_strip, y_dil], dim=1))
        else:
            y = self.fuse(y_strip)
        return self.act(f + y)


class ContextCompletionExpert(nn.Module):
    """
    Context Completion Expert（上下文补全专家）

    目标：
    - 通过空洞卷积和全局门控补充遮挡区域、大范围道路上下文。
    """
    def __init__(self, channels):
        super().__init__()
        self.dil2 = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, dilation=2, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )
        self.dil4 = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, dilation=4, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
        )
        self.global_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1, padding=0),
            ECALayer(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, f, h_g=None, l_g=None):
        y = self.fuse(torch.cat([self.dil2(f), self.dil4(f)], dim=1))
        y = y * self.global_gate(f)
        return self.act(f + y)


class FalsePositiveSuppressionExpert(nn.Module):
    """
    False-positive Suppression Expert（误检抑制专家）

    目标：
    - 当高分辨率细节强但低分辨率道路语义不支持时，降低非道路细线响应；
    - 显式利用 H_g, L_g 和 |H_g-L_g|。
    """
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            ConvBNAct(channels * 3, channels, kernel_size=1, padding=0),
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.fuse = nn.Sequential(
            ConvBNAct(channels * 2, channels, kernel_size=1, padding=0),
            ECALayer(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, f, h_g=None, l_g=None):
        if h_g is None or l_g is None:
            return f
        gate = self.gate(torch.cat([h_g, l_g, torch.abs(h_g - l_g)], dim=1))
        y = h_g * gate + l_g * (1.0 - gate)
        out = self.fuse(torch.cat([f, y], dim=1))
        return self.act(f + out)


class MutualGuidance(nn.Module):
    """
    High-Low Mutual Guidance（高低分辨率相互指导）

    1) Low -> High:
       低分辨率道路语义生成 road-aware gate，筛选高分辨率细节。

    2) High -> Low:
       被筛选后的高分辨率结构通过方向卷积反向补偿低分辨率语义。

    3) Mutual Fusion:
       concat(H_g, L_g, |H_g-L_g|, H_g*L_g) 得到 mutual-guided feature。
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


class MHRMoEBlock(nn.Module):
    """
    MHR-MoE:
    Mutual-guided High-low Routing Mixture of Experts
    相互指导高低分辨率路由专家混合模块

    输入:
    - decoder feature D_i
    - encoder features x0, x1, x2, x3
    输出:
    - adaptive skip_i，用于替代固定 skip connection。
    """
    def __init__(
        self,
        decoder_channels,
        out_channels,
        expert_channels=64,
        topk=2,
        use_context=True,
        use_suppression=True,
    ):
        super().__init__()
        self.expert_channels = expert_channels
        self.out_channels = out_channels
        self.topk = topk
        self.use_context = use_context
        self.use_suppression = use_suppression

        # High features: x0 + x1
        self.high_proj = ConvBNAct(64 + 64, expert_channels, kernel_size=1, padding=0)

        # Low features: x2 + x3
        self.low_proj = ConvBNAct(128 + 256, expert_channels, kernel_size=1, padding=0)

        self.mutual_guidance = MutualGuidance(expert_channels)

        self.experts = nn.ModuleList()
        self.experts.append(FrequencyDetailExpert(expert_channels))
        self.experts.append(StripDirectionExpert(expert_channels, kernels=(3, 5, 7), use_dilation_branch=True))
        if use_context:
            self.experts.append(ContextCompletionExpert(expert_channels))
        if use_suppression:
            self.experts.append(FalsePositiveSuppressionExpert(expert_channels))

        self.num_experts = len(self.experts)

        self.decoder_proj = ConvBNAct(decoder_channels, expert_channels, kernel_size=1, padding=0)
        hidden = max(expert_channels // 2, 16)
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expert_channels * 2, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, self.num_experts, kernel_size=1, bias=True),
        )

        self.mix_fuse = nn.Sequential(
            ConvBNAct(expert_channels, expert_channels, kernel_size=3),
            ECALayer(expert_channels),
        )

        self.docker = ConvBNAct(expert_channels, out_channels, kernel_size=1, padding=0)

    @staticmethod
    def _resize(x, size):
        if x.shape[-2:] == size:
            return x
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def _masked_topk_softmax(self, logits):
        """
        logits: [B, num_experts, 1, 1]
        返回经过 top-k mask 后重新归一化的权重。
        """
        weights = torch.softmax(logits, dim=1)

        if self.topk is None or self.topk <= 0 or self.topk >= self.num_experts:
            return weights

        k = min(self.topk, self.num_experts)
        topk_idx = torch.topk(weights, k=k, dim=1).indices
        mask = torch.zeros_like(weights)
        mask.scatter_(1, topk_idx, 1.0)

        weights = weights * mask
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        return weights

    def forward(self, d, x0, x1, x2, x3, target_size=None, return_info=False):
        if target_size is None:
            target_size = d.shape[-2:]

        # 1) Scale alignment（尺度对齐）
        x0_r = self._resize(x0, target_size)
        x1_r = self._resize(x1, target_size)
        x2_r = self._resize(x2, target_size)
        x3_r = self._resize(x3, target_size)

        h = self.high_proj(torch.cat([x0_r, x1_r], dim=1))
        l = self.low_proj(torch.cat([x2_r, x3_r], dim=1))

        # 2) High-Low Mutual Guidance（高低相互指导）
        f_mg, h_g, l_g, gate_l, gate_h = self.mutual_guidance(h, l)

        # 3) Expert Bank（专家库）
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(f_mg, h_g, l_g))

        expert_stack = torch.stack(expert_outputs, dim=1)  # [B, E, C, H, W]

        # 4) Stage-specific Router（阶段特异路由）
        d_proj = self._resize(self.decoder_proj(d), target_size)
        router_input = torch.cat([d_proj, f_mg], dim=1)
        logits = self.router(router_input)                 # [B, E, 1, 1]
        weights = self._masked_topk_softmax(logits)        # [B, E, 1, 1]

        # 5) Weighted Expert Mixture（加权专家混合）
        weights_view = weights.unsqueeze(-1)               # [B, E, 1, 1, 1]
        mixed = (expert_stack * weights_view).sum(dim=1)   # [B, C, H, W]
        mixed = self.mix_fuse(mixed)

        # 6) Docker（对接器）生成 adaptive skip
        skip = self.docker(mixed)

        if return_info:
            info = {
                "router_logits": logits,
                "router_weights": weights,
                "gate_low_to_high": gate_l,
                "gate_high_to_low": gate_h,
                "mutual_feature": f_mg,
            }
            return skip, info
        return skip


class HL50_v2(nn.Module):
    """
    HL50_v2 / R50Proj-MHRNet:
    ResNet50-projected Road-aware Mutual-guided High-low Routing Network
    ResNet50 投影版道路感知相互指导高低分辨率路由网络

    设计思路：
    1) 将 ResNet34 encoder 替换为 ResNet50 encoder；
    2) 保留 ResNet50 内部原始高通道 r1/r2/r3/r4 继续前向；
    3) 使用 1×1 projection 将 r1/r2/r3/r4 压回 HL_v2 兼容通道：64/128/256/512；
    4) 后续 MHR-MoE decoder 完全沿用 HL_v2_best：只启用 mhr8 和 mhr4，topk=2；
    5) 用于验证更强 ResNet 系列表征是否可以提升 Massachusetts Roads 的 IoU 上限。
    """
    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        num_classes=None,
        in_channels=None,
        pretrained=True,
        return_aux=False,
        expert_channels=64,
        topk=2,
        use_context=True,
        use_suppression=True,
        use_mhr_stage3=False,
        use_mhr_stage0=False,
        **kwargs,
    ):
        super(HL50_v2, self).__init__()

        if in_channels is not None:
            n_channels = in_channels
        if num_classes is not None:
            n_classes = num_classes

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.return_aux = return_aux
        self.use_mhr_stage3 = use_mhr_stage3
        self.use_mhr_stage0 = use_mhr_stage0

        encoder = self._get_resnet50(pretrained=pretrained)

        if n_channels != 3:
            self.input_adapter = nn.Conv2d(n_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # Encoder
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

        # ResNet50 raw features are high-dimensional:
        # r1/r2/r3/r4 = 256/512/1024/2048.
        # Project them back to ResNet34-style channels so the original HL_v2 decoder can be reused.
        self.proj1 = self._make_projection(256, 64)
        self.proj2 = self._make_projection(512, 128)
        self.proj3 = self._make_projection(1024, 256)
        self.proj4 = self._make_projection(2048, 512)

        # Decoder
        self.dec3 = ResidualDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=64, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=64, out_channels=64)

        # MHR-MoE adaptive skip generators
        # 默认只替换 H/8 和 H/4 两个关键 skip，更稳、更省算力。
        if use_mhr_stage3:
            self.mhr16 = MHRMoEBlock(
                decoder_channels=512,
                out_channels=256,
                expert_channels=expert_channels,
                topk=topk,
                use_context=use_context,
                use_suppression=use_suppression,
            )
        else:
            self.mhr16 = None

        self.mhr8 = MHRMoEBlock(
            decoder_channels=256,
            out_channels=128,
            expert_channels=expert_channels,
            topk=topk,
            use_context=use_context,
            use_suppression=use_suppression,
        )

        self.mhr4 = MHRMoEBlock(
            decoder_channels=128,
            out_channels=64,
            expert_channels=expert_channels,
            topk=topk,
            use_context=use_context,
            use_suppression=use_suppression,
        )

        if use_mhr_stage0:
            self.mhr2 = MHRMoEBlock(
                decoder_channels=96,
                out_channels=64,
                expert_channels=expert_channels,
                topk=topk,
                use_context=use_context,
                use_suppression=use_suppression,
            )
        else:
            self.mhr2 = None

        # Lightweight H/2 output head
        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    @staticmethod
    def _make_projection(in_channels, out_channels):
        """
        1×1 projection（投影层）:
        将 ResNet50 的高通道特征压回 HL_v2 decoder 兼容通道。
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _get_resnet50(pretrained=True):
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            model = resnet50(weights=weights)
            return model
        except Exception:
            from torchvision import models
            try:
                model = models.resnet50(pretrained=pretrained)
            except TypeError:
                model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            return model

    def forward_features(self, x):
        input_size = x.shape[-2:]
        x_in = self.input_adapter(x)

        # Encoder stem
        x0 = self.stem(x_in)                    # H/2, 64

        # ResNet50 raw features keep their original channels inside the backbone.
        r1 = self.layer1(self.maxpool(x0))      # H/4, 256
        r2 = self.layer2(r1)                    # H/8, 512
        r3 = self.layer3(r2)                    # H/16, 1024
        r4 = self.layer4(r3)                    # H/32, 2048

        # Project raw ResNet50 features to HL_v2-compatible channels.
        x1 = self.proj1(r1)                     # H/4, 64
        x2 = self.proj2(r2)                     # H/8, 128
        x3 = self.proj3(r3)                     # H/16, 256
        x4 = self.proj4(r4)                     # H/32, 512

        aux = {}

        # Decoder stage H/16
        if self.mhr16 is not None:
            skip3, info16 = self.mhr16(x4, x0, x1, x2, x3, target_size=x3.shape[-2:], return_info=True)
            aux["mhr16"] = info16
        else:
            skip3 = x3
        d3 = self.dec3(x4, skip3)               # H/16, 256

        # Decoder stage H/8
        skip2, info8 = self.mhr8(d3, x0, x1, x2, x3, target_size=x2.shape[-2:], return_info=True)
        aux["mhr8"] = info8
        d2 = self.dec2(d3, skip2)               # H/8, 128

        # Decoder stage H/4
        skip1, info4 = self.mhr4(d2, x0, x1, x2, x3, target_size=x1.shape[-2:], return_info=True)
        aux["mhr4"] = info4
        d1 = self.dec1(d2, skip1)               # H/4, 96

        # Decoder stage H/2
        if self.mhr2 is not None:
            skip0, info2 = self.mhr2(d1, x0, x1, x2, x3, target_size=x0.shape[-2:], return_info=True)
            aux["mhr2"] = info2
        else:
            skip0 = x0
        d0 = self.dec0(d1, skip0)               # H/2, 64

        logits_half = self.out_head(d0)         # H/2
        logits = F.interpolate(logits_half, size=input_size, mode="bilinear", align_corners=False)

        aux.update({
            "logits_half": logits_half,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "r4": r4,
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
    model = HL50_v2(n_channels=3, n_classes=1, pretrained=False, return_aux=False)
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
