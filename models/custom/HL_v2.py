import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["HL_v2"]


def _auto_padding(kernel_size, dilation=1):
    """支持 int / tuple kernel 的 same padding。"""
    if isinstance(kernel_size, tuple):
        if isinstance(dilation, tuple):
            return tuple(((k - 1) // 2) * d for k, d in zip(kernel_size, dilation))
        return tuple(((k - 1) // 2) * dilation for k in kernel_size)
    return ((kernel_size - 1) // 2) * dilation


def _init_last_conv_to_zero(module):
    if isinstance(module, nn.Conv2d):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _bounded_gate(param, max_value=0.5):
    return max_value * torch.sigmoid(param)


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
    """Efficient Channel Attention（高效通道注意力）。"""
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
    """与 HL_base 保持一致的残差解码块。"""
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


class SpatialTransformer2D(nn.Module):
    """
    2D 空间变换器。
    flow 的单位是 feature map 上的像素位移。
    flow[:, 0] 是 x 方向位移，flow[:, 1] 是 y 方向位移。
    """
    def __init__(self, padding_mode="border", align_corners=True):
        super().__init__()
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    @staticmethod
    def _base_grid(batch, height, width, device, dtype, align_corners=True):
        if align_corners:
            ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
            xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
        else:
            ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * 2.0 / height - 1.0
            xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * 2.0 / width - 1.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return grid.unsqueeze(0).expand(batch, height, width, 2)

    def forward(self, src, flow):
        b, _, h, w = src.shape
        grid = self._base_grid(b, h, w, src.device, src.dtype, self.align_corners)

        if self.align_corners:
            norm_x = flow[:, 0] * 2.0 / max(w - 1, 1)
            norm_y = flow[:, 1] * 2.0 / max(h - 1, 1)
        else:
            norm_x = flow[:, 0] * 2.0 / max(w, 1)
            norm_y = flow[:, 1] * 2.0 / max(h, 1)

        disp = torch.stack([norm_x, norm_y], dim=-1)
        new_grid = grid + disp
        return F.grid_sample(
            src,
            new_grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )


class VecInt2D(nn.Module):
    """
    速度场积分模块。
    使用 scaling and squaring，把 velocity field 积分成平滑变形场。
    这是从 MPT 论文和开源实现里迁移过来的关键部分。
    """
    def __init__(self, nsteps=4):
        super().__init__()
        self.nsteps = int(nsteps)
        self.transformer = SpatialTransformer2D(padding_mode="border", align_corners=True)
        self.scale = 1.0 / (2 ** self.nsteps)

    def forward(self, velocity):
        flow = velocity * self.scale
        for _ in range(self.nsteps):
            flow = flow + self.transformer(flow, flow)
        return flow


class MorphPatchSampler2D(nn.Module):
    """
    形态图像块采样。
    不是只做一次普通 warp，而是在 velocity field 得到的变形场周围采样多个邻域点。
    为了适配 1024 道路图像，默认使用 star13 采样点，兼顾效果和显存。
    """
    def __init__(self, channels, patch_size=5, groups=8, sample_mode="star13"):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        self.groups = groups if channels % groups == 0 else 1
        self.sample_mode = sample_mode
        self.transformer = SpatialTransformer2D(padding_mode="border", align_corners=True)

        offsets = self._build_offsets(patch_size, sample_mode)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.float32), persistent=False)
        self.group_weights = nn.Parameter(torch.zeros(self.groups, len(offsets)))

        self.fuse = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=1),
            ECALayer(channels),
        )

    @staticmethod
    def _build_offsets(patch_size, sample_mode):
        r = patch_size // 2
        if sample_mode == "full":
            return [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        if sample_mode == "star13" and r >= 2:
            return [
                (0, 0),
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-2, 0), (2, 0), (0, -2), (0, 2),
                (-1, -1), (1, -1), (-1, 1), (1, 1),
            ]
        return [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]

    def forward(self, x, flow):
        b, c, h, w = x.shape
        weights = torch.softmax(self.group_weights, dim=1)
        channels_per_group = c // self.groups

        out = None
        for idx in range(self.offsets.shape[0]):
            dx = self.offsets[idx, 0].to(dtype=x.dtype, device=x.device)
            dy = self.offsets[idx, 1].to(dtype=x.dtype, device=x.device)
            offset_flow = flow.clone()
            offset_flow[:, 0] = offset_flow[:, 0] + dx
            offset_flow[:, 1] = offset_flow[:, 1] + dy
            sampled = self.transformer(x, offset_flow)

            wi = weights[:, idx].repeat_interleave(channels_per_group).view(1, c, 1, 1)
            sampled = sampled * wi
            out = sampled if out is None else out + sampled

        return self.fuse(out)


class VelocityPredictor2D(nn.Module):
    """由双分辨率特征和道路先验预测速度场。"""
    def __init__(self, in_channels, hidden_channels=128, max_disp=3.0):
        super().__init__()
        self.max_disp = float(max_disp)
        self.body = nn.Sequential(
            ConvBNAct(in_channels, hidden_channels, kernel_size=3),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, groups=1),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.flow_head = nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        _init_last_conv_to_zero(self.flow_head)

    def forward(self, x):
        f = self.body(x)
        raw = self.flow_head(f)
        gate = self.gate_head(f)
        velocity = self.max_disp * torch.tanh(raw) * gate
        return velocity, gate


class RoadPriorBuilder(nn.Module):
    """根据粗预测生成道路概率、不确定区域、候选区域、边界提示。"""
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, logits, size):
        p = torch.sigmoid(logits)
        p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
        u = 4.0 * p * (1.0 - p)
        r = F.max_pool2d(p, kernel_size=7, stride=1, padding=3)
        gx = F.conv2d(p, self.sobel_x.to(dtype=p.dtype), padding=1)
        gy = F.conv2d(p, self.sobel_y.to(dtype=p.dtype), padding=1)
        e = torch.sqrt(gx * gx + gy * gy + 1e-6)
        e = e / (e.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return torch.cat([p, u, r, e], dim=1)


class SoftClusterCrossAttention2D(nn.Module):
    """
    道路语义聚类注意力。
    先用 soft assignment 构建语义中心，再让 query 特征和语义中心做交叉注意力。
    """
    def __init__(self, channels=128, num_clusters=32, num_heads=4, prior_channels=4):
        super().__init__()
        assert channels % num_heads == 0, "channels 必须能被 num_heads 整除。"
        self.channels = channels
        self.num_clusters = num_clusters
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.pre = ConvBNAct(channels, channels, kernel_size=1)
        self.cluster_weights = nn.Parameter(torch.randn(channels, num_clusters) * 0.02)
        self.cluster_bias = nn.Parameter(torch.zeros(num_clusters))
        self.prior_proj = nn.Conv2d(prior_channels, num_clusters, kernel_size=1)

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def build_prototypes(self, feat, priors=None):
        feat = self.pre(feat)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2)  # B, N, C
        logits = torch.matmul(tokens, self.cluster_weights) + self.cluster_bias
        if priors is not None:
            prior_logits = self.prior_proj(priors).flatten(2).transpose(1, 2)
            logits = logits + prior_logits
        # 对空间位置归一化，使每个 cluster 从全图中聚合自己的语义中心。
        assign = torch.softmax(logits, dim=1)  # B, N, K
        prototypes = torch.einsum("bnk,bnc->bkc", assign, tokens)
        return prototypes

    def attend(self, query_feat, prototypes):
        b, c, h, w = query_feat.shape
        q = self.q_proj(query_feat).flatten(2).transpose(1, 2)  # B, N, C
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)

        k = self.k_proj(prototypes).view(b, self.num_clusters, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(prototypes).view(b, self.num_clusters, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, h * w, c)
        out = out.transpose(1, 2).view(b, c, h, w)
        return self.out_proj(out)

    def forward(self, query_feat, proto_feat, priors=None):
        prototypes = self.build_prototypes(proto_feat, priors=priors)
        return self.attend(query_feat, prototypes), prototypes


class BMMGBlock(nn.Module):
    """
    Bi-resolution Morph-Patch Mutual Guidance，双分辨率形态互导模块。

    high 分支：1/4 分辨率，保留道路边界和细节。
    low 分支：1/8 分辨率，提供更稳定的道路语义。
    两个方向都使用 velocity field -> VecInt -> MorphPatch，不做普通单向融合。
    """
    def __init__(
        self,
        high_channels=96,
        low_channels=128,
        embed_channels=128,
        patch_size=5,
        sample_mode="star13",
        num_clusters=32,
        num_heads=4,
        vecint_steps=4,
        max_disp_high=3.0,
        max_disp_low=2.0,
    ):
        super().__init__()
        c = embed_channels
        self.c = c
        self.high_proj = ConvBNAct(high_channels, c, kernel_size=1)
        self.low_proj = ConvBNAct(low_channels, c, kernel_size=1)
        self.prior_builder = RoadPriorBuilder()

        self.vel_high = VelocityPredictor2D(2 * c + 4, hidden_channels=c, max_disp=max_disp_high)
        self.vel_low = VelocityPredictor2D(3 * c + 4, hidden_channels=c, max_disp=max_disp_low)
        self.vecint_high = VecInt2D(nsteps=vecint_steps)
        self.vecint_low = VecInt2D(nsteps=vecint_steps)

        self.morph_high = MorphPatchSampler2D(c, patch_size=patch_size, groups=8, sample_mode=sample_mode)
        self.morph_low = MorphPatchSampler2D(c, patch_size=patch_size, groups=8, sample_mode=sample_mode)

        self.detail_high = nn.Sequential(
            ConvBNAct(c + 4, c, kernel_size=3),
            ConvBNAct(c, c, kernel_size=3),
        )

        self.proto_fuse = nn.Sequential(
            ConvBNAct(2 * c + 4, c, kernel_size=3),
            ConvBNAct(c, c, kernel_size=3),
        )
        self.sca = SoftClusterCrossAttention2D(c, num_clusters=num_clusters, num_heads=num_heads, prior_channels=4)

        self.update_high = nn.Sequential(
            ConvBNAct(4 * c + 4, c, kernel_size=3),
            ConvBNAct(c, c, kernel_size=3),
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.update_low = nn.Sequential(
            ConvBNAct(4 * c + 4, c, kernel_size=3),
            ConvBNAct(c, c, kernel_size=3),
            nn.Conv2d(c, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
        )

        self.confuse_head = nn.Sequential(
            ConvBNAct(2 * c + 4, c // 2, kernel_size=3),
            nn.Conv2d(c // 2, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 初始不要过度破坏 HL_base，但也不能太小。max=0.5，初始约 0.10。
        self.alpha_h = nn.Parameter(torch.tensor(-1.3862944))
        self.alpha_l = nn.Parameter(torch.tensor(-1.3862944))

    def forward(self, high_feat, low_feat, base_logits_half):
        fh = self.high_proj(high_feat)  # H/4
        fl = self.low_proj(low_feat)    # H/8
        high_size = fh.shape[-2:]
        low_size = fl.shape[-2:]

        pri_h = self.prior_builder(base_logits_half, high_size)
        pri_l = self.prior_builder(base_logits_half, low_size)
        fl_up = F.interpolate(fl, size=high_size, mode="bilinear", align_corners=False)

        # low -> high：低分辨率语义指导高分辨率形态采样。
        v_h_in = torch.cat([fh, fl_up, pri_h], dim=1)
        v_h, gate_h = self.vel_high(v_h_in)
        phi_h = self.vecint_high(v_h)
        fh_morph = self.morph_high(fh, phi_h)

        # high -> low：高分辨率细节反向指导低分辨率语义采样。
        detail_h = self.detail_high(torch.cat([fh, pri_h], dim=1))
        fh_down = F.interpolate(fh, size=low_size, mode="bilinear", align_corners=False)
        detail_l = F.interpolate(detail_h, size=low_size, mode="bilinear", align_corners=False)
        v_l_in = torch.cat([fl, fh_down, detail_l, pri_l], dim=1)
        v_l, gate_l = self.vel_low(v_l_in)
        phi_l = self.vecint_low(v_l)
        fl_morph = self.morph_low(fl, phi_l)

        # 双向语义聚类注意力：用两个分支共同生成道路语义中心。
        fl_morph_up = F.interpolate(fl_morph, size=high_size, mode="bilinear", align_corners=False)
        proto_feat = self.proto_fuse(torch.cat([fh_morph, fl_morph_up, pri_h], dim=1))
        fh_sca, prototypes = self.sca(query_feat=fh_morph, proto_feat=proto_feat, priors=pri_h)
        fl_sca = self.sca.attend(fl_morph, prototypes)

        # 双向残差更新。
        fl_sca_up = F.interpolate(fl_sca, size=high_size, mode="bilinear", align_corners=False)
        high_delta = self.update_high(torch.cat([fh, fh_morph, fh_sca, fl_sca_up, pri_h], dim=1))
        fh_out = fh + _bounded_gate(self.alpha_h, max_value=0.5) * high_delta

        fh_sca_down = F.interpolate(fh_sca, size=low_size, mode="bilinear", align_corners=False)
        low_delta = self.update_low(torch.cat([fl, fl_morph, fl_sca, fh_sca_down, pri_l], dim=1))
        fl_out = fl + _bounded_gate(self.alpha_l, max_value=0.5) * low_delta

        confuse = self.confuse_head(torch.cat([fh_sca, fh_morph, pri_h], dim=1))

        aux = {
            "v_h": v_h,
            "v_l": v_l,
            "phi_h": phi_h,
            "phi_l": phi_l,
            "gate_h": gate_h,
            "gate_l": gate_l,
            "confuse_h": confuse,
            "fh_morph": fh_morph,
            "fl_morph": fl_morph,
            "fh_sca": fh_sca,
            "fl_sca": fl_sca,
        }
        return fh_out, fl_out, aux


class HL_v2(nn.Module):
    """
    HL_v2：面向道路分割的双分辨率形态互导网络。

    主体保留 HL_base 的 ResNet34 编码器和残差解码器。
    新增 BMMG：
      1) 低分辨率语义指导高分辨率 Morph-Patch；
      2) 高分辨率细节反向指导低分辨率 Morph-Patch；
      3) 双向语义聚类注意力；
      4) 正负残差预测，兼顾补道路和抑制类道路误检。
    """
    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        num_classes=None,
        in_channels=None,
        pretrained=True,
        return_aux=False,
        bmmg_channels=128,
        morph_patch_size=5,
        morph_sample_mode="star13",
        num_clusters=32,
        sca_heads=4,
        vecint_steps=4,
        max_disp_high=3.0,
        max_disp_low=2.0,
        **kwargs,
    ):
        super().__init__()
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

        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        self.dec3 = ResidualDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=64, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=64, out_channels=64)

        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

        self.bmmg = BMMGBlock(
            high_channels=96,
            low_channels=128,
            embed_channels=bmmg_channels,
            patch_size=morph_patch_size,
            sample_mode=morph_sample_mode,
            num_clusters=num_clusters,
            num_heads=sca_heads,
            vecint_steps=vecint_steps,
            max_disp_high=max_disp_high,
            max_disp_low=max_disp_low,
        )

        refine_in = 64 + bmmg_channels + bmmg_channels + 3
        self.refine_fuse = nn.Sequential(
            ConvBNAct(refine_in, 128, kernel_size=3),
            ConvBNAct(128, 96, kernel_size=3),
            ConvBNAct(96, 64, kernel_size=3),
        )
        self.delta_pos_head = nn.Conv2d(64, n_classes, kernel_size=1)
        self.delta_neg_head = nn.Conv2d(64, n_classes, kernel_size=1)
        nn.init.zeros_(self.delta_pos_head.weight)
        nn.init.constant_(self.delta_pos_head.bias, -3.0)
        nn.init.zeros_(self.delta_neg_head.weight)
        nn.init.constant_(self.delta_neg_head.bias, -3.0)

        # beta/gamma 最大 0.8，初始约 0.16，第一阶段不会毁掉 base logits。
        self.beta_pos = nn.Parameter(torch.tensor(-1.3862944))
        self.gamma_neg = nn.Parameter(torch.tensor(-1.3862944))

    @staticmethod
    def _get_resnet34(pretrained=True):
        try:
            from torchvision.models import resnet34, ResNet34_Weights
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            return resnet34(weights=weights)
        except Exception:
            from torchvision import models
            try:
                return models.resnet34(pretrained=pretrained)
            except TypeError:
                return models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)

    @staticmethod
    def _prior_half(base_logits_half):
        p = torch.sigmoid(base_logits_half)
        u = 4.0 * p * (1.0 - p)
        r = F.max_pool2d(p, kernel_size=7, stride=1, padding=3)
        return p, u, r

    def forward_features(self, x):
        input_size = x.shape[-2:]
        x_in = self.input_adapter(x)

        x0 = self.stem(x_in)                    # H/2,  64
        x1 = self.layer1(self.maxpool(x0))      # H/4,  64
        x2 = self.layer2(x1)                    # H/8,  128
        x3 = self.layer3(x2)                    # H/16, 256
        x4 = self.layer4(x3)                    # H/32, 512

        d3 = self.dec3(x4, x3)                  # H/16, 256
        d2 = self.dec2(d3, x2)                  # H/8,  128
        d1 = self.dec1(d2, x1)                  # H/4,   96
        d0 = self.dec0(d1, x0)                  # H/2,   64

        base_logits_half = self.out_head(d0)    # H/2, n_classes
        fh_out, fl_out, bmmg_aux = self.bmmg(d1, d2, base_logits_half)

        half_size = d0.shape[-2:]
        fh_half = F.interpolate(fh_out, size=half_size, mode="bilinear", align_corners=False)
        fl_half = F.interpolate(fl_out, size=half_size, mode="bilinear", align_corners=False)
        confuse_half = F.interpolate(bmmg_aux["confuse_h"], size=half_size, mode="bilinear", align_corners=False)
        p_half, u_half, r_half = self._prior_half(base_logits_half)

        refine_feat = self.refine_fuse(torch.cat([d0, fh_half, fl_half, p_half, u_half, confuse_half], dim=1))
        delta_pos = F.softplus(self.delta_pos_head(refine_feat))
        delta_neg = F.softplus(self.delta_neg_head(refine_feat))

        gate_pos = torch.clamp(u_half * r_half, 0.0, 1.0)
        gate_neg = torch.clamp(confuse_half * (0.25 + 0.75 * p_half), 0.0, 1.0)
        beta = _bounded_gate(self.beta_pos, max_value=0.8)
        gamma = _bounded_gate(self.gamma_neg, max_value=0.8)

        final_logits_half = base_logits_half + beta * gate_pos * delta_pos - gamma * gate_neg * delta_neg
        logits = F.interpolate(final_logits_half, size=input_size, mode="bilinear", align_corners=False)
        base_logits = F.interpolate(base_logits_half, size=input_size, mode="bilinear", align_corners=False)

        aux = {
            "logits": logits,
            "base_logits": base_logits,
            "logits_half": final_logits_half,
            "base_logits_half": base_logits_half,
            "delta_pos": delta_pos,
            "delta_neg": delta_neg,
            "gate_pos": gate_pos,
            "gate_neg": gate_neg,
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
        aux.update(bmmg_aux)
        return logits, aux

    def forward(self, x, return_aux=None):
        logits, aux = self.forward_features(x)
        use_aux = self.return_aux if return_aux is None else return_aux
        if use_aux:
            return aux
        return logits


if __name__ == "__main__":
    model = HL_v2(n_channels=3, n_classes=1, pretrained=False, return_aux=True)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)
    print("Output:", out["logits"].shape)
    print("Aux keys:", sorted(out.keys()))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Trainable params: {trainable_params / 1e6:.2f} M")
