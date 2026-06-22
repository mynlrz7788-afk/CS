import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["HL_v3"]


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
    """HL_base 原始残差解码块。"""
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


class AddExpert(nn.Module):
    """补全专家：只在 M_add 区域补细路、断裂和边界。"""
    def __init__(self, in_channels, mid_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, kernel_size=1),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, groups=mid_channels),
            ConvBNAct(mid_channels, mid_channels, kernel_size=(1, 7), groups=mid_channels),
            ConvBNAct(mid_channels, mid_channels, kernel_size=(7, 1), groups=mid_channels),
            ConvBNAct(mid_channels, mid_channels, kernel_size=1),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class SuppressExpert(nn.Module):
    """抑制专家：只在 M_sup 区域压制类道路误检。"""
    def __init__(self, in_channels, mid_channels=32):
        super().__init__()
        self.pre = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, kernel_size=1),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, groups=mid_channels),
            ConvBNAct(mid_channels, mid_channels, kernel_size=1),
        )
        self.eca = ECALayer(mid_channels)
        self.head = nn.Conv2d(mid_channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.pre(x)
        x = self.eca(x)
        return self.head(x)


class ShrinkExpert(nn.Module):
    """收缩专家：处理低分辨率粗预测导致的边界外扩。"""
    def __init__(self, in_channels=4, mid_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, kernel_size=1),
            ConvBNAct(mid_channels, mid_channels, kernel_size=3, groups=mid_channels),
            ConvBNAct(mid_channels, mid_channels, kernel_size=1),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def _inverse_sigmoid(x):
    x = float(x)
    x = min(max(x, 1e-4), 1.0 - 1e-4)
    return math.log(x / (1.0 - x))


class DRRModule(nn.Module):
    """
    双分辨率分歧路由纠偏模块（DRR）。

    输入：
        h4: 1/4 高分辨率 skip，来自 x1，默认 64 通道。
        l8: 1/8 低分辨率语义，来自 d2，默认 128 通道。

    输出：
        h4_ref: 修正后的 1/4 skip，通道与 h4 相同。
        l8_ref: 修正后的 1/8 语义，通道与 l8 相同。
        aux:    训练阶段用于辅助监督的中间结果。

    设计目标：
        1. P_H 和 P_L 先分别判断道路。
        2. 用二者分歧构造 M_add、M_sup、M_shrink。
        3. M_add 只补靠近低分辨率道路主体的高分辨率细节。
        4. M_sup 只压远离低分辨率道路主体的高分辨率误检。
        5. M_shrink 收缩低分辨率边界外扩。
    """
    def __init__(
        self,
        h_in=64,
        l_in=128,
        h_mid=32,
        l_mid=64,
        add_mid=32,
        sup_mid=32,
        shrink_mid=16,
        n_classes=1,
        alpha_init=0.8,
        beta_init=0.55,
        gamma_init=0.35,
        delta_init=0.3,
    ):
        super().__init__()
        self.n_classes = n_classes

        self.h_reduce = ConvBNAct(h_in, h_mid, kernel_size=1)
        self.l_reduce = ConvBNAct(l_in, l_mid, kernel_size=1)

        # 高分辨率道路证据头：轻量条带结构，偏向细路和边界。
        self.high_feat = nn.Sequential(
            ConvBNAct(h_mid, h_mid, kernel_size=3, groups=h_mid),
            ConvBNAct(h_mid, h_mid, kernel_size=(1, 5), groups=h_mid),
            ConvBNAct(h_mid, h_mid, kernel_size=(5, 1), groups=h_mid),
            ConvBNAct(h_mid, h_mid, kernel_size=1),
        )
        self.high_head = nn.Conv2d(h_mid, n_classes, kernel_size=1)

        # 低分辨率道路主体头：稳定语义判断。
        self.low_head = nn.Sequential(
            ConvBNAct(l_mid, l_mid, kernel_size=3, groups=l_mid),
            ConvBNAct(l_mid, l_mid, kernel_size=1),
            nn.Conv2d(l_mid, n_classes, kernel_size=1),
        )

        expert_in = h_mid + l_mid + 3 * n_classes
        self.add_expert = AddExpert(expert_in, add_mid)
        self.sup_expert = SuppressExpert(expert_in, sup_mid)
        self.shrink_expert = ShrinkExpert(4 * n_classes, shrink_mid)

        # 把路由结果写回 h4。最后一层零初始化，让模型初始接近 HL_base，训练更稳。
        self.rmap_to_h = nn.Sequential(
            ConvBNAct(4 * n_classes, h_mid, kernel_size=1),
            nn.Conv2d(h_mid, h_in, kernel_size=1, bias=False),
        )

        # 把分歧反馈写回 l8。最后一层零初始化，让模型初始接近 HL_base。
        self.fb_to_l = nn.Sequential(
            ConvBNAct(6 * n_classes, l_mid, kernel_size=1),
            nn.Conv2d(l_mid, l_in, kernel_size=1, bias=False),
        )

        nn.init.zeros_(self.rmap_to_h[-1].weight)
        nn.init.zeros_(self.fb_to_l[-1].weight)

        # 用 sigmoid 约束到 [0, 1.5]，避免前期修正过强。
        self.alpha_raw = nn.Parameter(torch.tensor(_inverse_sigmoid(alpha_init / 1.5), dtype=torch.float32))
        self.beta_raw = nn.Parameter(torch.tensor(_inverse_sigmoid(beta_init / 1.5), dtype=torch.float32))
        self.gamma_raw = nn.Parameter(torch.tensor(_inverse_sigmoid(gamma_init / 1.5), dtype=torch.float32))
        self.delta_raw = nn.Parameter(torch.tensor(_inverse_sigmoid(delta_init / 1.0), dtype=torch.float32))

    @staticmethod
    def _safe_logit(p):
        p = p.clamp(1e-4, 1.0 - 1e-4)
        return torch.log(p / (1.0 - p))

    @staticmethod
    def _morph_gradient(p, kernel_size=5):
        pad = kernel_size // 2
        dilate = F.max_pool2d(p, kernel_size=kernel_size, stride=1, padding=pad)
        erode = -F.max_pool2d(-p, kernel_size=kernel_size, stride=1, padding=pad)
        return (dilate - erode).clamp(0.0, 1.0)

    def _scale_alpha(self):
        return 1.5 * torch.sigmoid(self.alpha_raw)

    def _scale_beta(self):
        return 1.5 * torch.sigmoid(self.beta_raw)

    def _scale_gamma(self):
        return 1.5 * torch.sigmoid(self.gamma_raw)

    def _scale_delta(self):
        return torch.sigmoid(self.delta_raw)

    def forward(self, h4, l8):
        h = self.h_reduce(h4)  # B,32,H/4,W/4
        l = self.l_reduce(l8)  # B,64,H/8,W/8
        h_size = h.shape[-2:]
        l_size = l.shape[-2:]
        l_up = F.interpolate(l, size=h_size, mode="bilinear", align_corners=False)

        # 1. 双分辨率分别判断道路。
        h_feat = self.high_feat(h)
        logits_h = self.high_head(h_feat)
        p_h = torch.sigmoid(logits_h)

        logits_l8 = self.low_head(l)
        p_l = F.interpolate(torch.sigmoid(logits_l8), size=h_size, mode="bilinear", align_corners=False)

        # 2. 构造分歧区域。
        m_keep = p_h * p_l
        m_honly = p_h * (1.0 - p_l)
        m_lonly = p_l * (1.0 - p_h)

        u_h = 4.0 * p_h * (1.0 - p_h)
        u_l = 4.0 * p_l * (1.0 - p_l)
        m_unc = (0.5 * (u_h + u_l)).clamp(0.0, 1.0)

        # 3. 低分辨率主体邻域。fix1 版本弱化抑制、扩大可补区域，避免 Recall 掉太多。
        n_l_small = F.max_pool2d(p_l, kernel_size=5, stride=1, padding=2)
        n_l_mid = F.max_pool2d(p_l, kernel_size=15, stride=1, padding=7)
        n_l = torch.clamp(torch.maximum(n_l_small, 0.65 * n_l_mid), 0.0, 1.0)

        m_add = m_honly * n_l * (0.7 + 0.3 * m_unc)
        m_sup = m_honly * (1.0 - n_l) * (1.0 - 0.5 * m_unc)

        b_l = self._morph_gradient(p_l, kernel_size=5)
        m_shrink = m_lonly * b_l

        # 4. 三个路由专家。
        add_in = torch.cat([h, l_up, p_l, p_h, m_add], dim=1)
        sup_in = torch.cat([h, l_up, p_l, p_h, m_sup], dim=1)
        shrink_in = torch.cat([p_l, p_h, b_l, m_shrink], dim=1)

        add_logit = self.add_expert(add_in)
        sup_logit = self.sup_expert(sup_in)
        shrink_logit = self.shrink_expert(shrink_in)

        r_add = F.softplus(add_logit) * m_add
        r_sup = F.softplus(sup_logit) * m_sup
        r_shrink = F.softplus(shrink_logit) * m_shrink

        alpha = self._scale_alpha()
        beta = self._scale_beta()
        gamma = self._scale_gamma()
        delta = self._scale_delta()

        aux_logit_4 = self._safe_logit(p_l) + alpha * r_add - beta * r_sup - gamma * r_shrink
        aux_prob_4 = torch.sigmoid(aux_logit_4)

        # 5. 修正高分辨率 skip。
        # fix1 版本降低抑制权重，避免 Recall 被压得过低。
        w_h = 1.0 + 0.15 * m_keep + 0.35 * m_add - 0.30 * m_sup - 0.20 * m_shrink
        w_h = w_h.clamp(0.45, 1.55)
        r_map = torch.cat([r_add, r_sup, r_shrink, aux_prob_4], dim=1)
        h_delta = self.rmap_to_h(r_map)
        h4_ref = h4 * w_h + h_delta

        # 6. 高分辨率分歧反馈低分辨率。
        fb = torch.cat([m_add, m_sup, m_shrink, r_add, r_sup, r_shrink], dim=1)
        fb = F.interpolate(fb, size=l_size, mode="area")
        l_delta = self.fb_to_l(fb)
        l8_ref = l8 + delta * l_delta

        aux = {
            "p_h_logits": logits_h,
            "p_l_logits8": logits_l8,
            "add_raw": add_logit,
            "sup_raw": sup_logit,
            "shrink_raw": shrink_logit,
            "aux_logits4": aux_logit_4,
            "logits_h": logits_h,
            "logits_l8": logits_l8,
            "p_h": p_h,
            "p_l": p_l,
            "m_keep": m_keep,
            "m_add": m_add,
            "m_sup": m_sup,
            "m_shrink": m_shrink,
            "m_unc": m_unc,
            "b_l": b_l,
            "add_logit": add_logit,
            "sup_logit": sup_logit,
            "shrink_logit": shrink_logit,
            "r_add": r_add,
            "r_sup": r_sup,
            "r_shrink": r_shrink,
            "aux_logit_4": aux_logit_4,
            "alpha": alpha.detach(),
            "beta": beta.detach(),
            "gamma": gamma.detach(),
            "delta": delta.detach(),
        }
        return h4_ref, l8_ref, aux


class HL_v3(nn.Module):
    """
    HL_v3 = HL_base + 双分辨率分歧路由纠偏模块（DRR）。

    相比 HL_base 的唯一结构性改动：
        d2 = dec2(d3, x2) 后，进入 dec1 之前，
        用 1/4 skip x1 和 1/8 decoder feature d2 做分歧路由纠偏。

    原始 HL_base 尺度：
        x0: H/2,  64
        x1: H/4,  64
        x2: H/8,  128
        x3: H/16, 256
        x4: H/32, 512
        d2: H/8,  128
        d1: H/4,   96
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
        super(HL_v3, self).__init__()

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

        self.dec3 = ResidualDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)

        self.drr = DRRModule(
            h_in=64,
            l_in=128,
            h_mid=32,
            l_mid=64,
            add_mid=32,
            sup_mid=32,
            shrink_mid=16,
            n_classes=n_classes,
            alpha_init=0.8,
            beta_init=0.55,
            gamma_init=0.35,
            delta_init=0.3,
        )

        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=64, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=64, out_channels=64)

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

        x0 = self.stem(x_in)                    # H/2,  64
        x1 = self.layer1(self.maxpool(x0))      # H/4,  64
        x2 = self.layer2(x1)                    # H/8,  128
        x3 = self.layer3(x2)                    # H/16, 256
        x4 = self.layer4(x3)                    # H/32, 512

        d3 = self.dec3(x4, x3)                  # H/16, 256
        d2 = self.dec2(d3, x2)                  # H/8,  128

        x1_ref, d2_ref, drr_aux = self.drr(x1, d2)

        d1 = self.dec1(d2_ref, x1_ref)          # H/4,  96
        d0 = self.dec0(d1, x0)                  # H/2,  64

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
            "x1_ref": x1_ref,
            "d2_ref": d2_ref,
            "d1": d1,
            "d0": d0,
            "drr": drr_aux,
        }
        return logits, aux

    def forward(self, x):
        logits, aux = self.forward_features(x)
        if self.training:
            return {"out": logits, "drr": aux["drr"]}
        if self.return_aux:
            aux["fused_logits"] = logits
            return aux
        return logits


if __name__ == "__main__":
    model = HL_v3(n_channels=3, n_classes=1, pretrained=False, return_aux=False)
    x = torch.randn(1, 3, 256, 256)
    model.train()
    y = model(x)
    print("Train output:", y["out"].shape, y["drr"]["aux_logit_4"].shape)
    model.eval()
    with torch.no_grad():
        y_eval = model(x)
    print("Eval output:", y_eval.shape)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Trainable params: {trainable_params / 1e6:.2f} M")
