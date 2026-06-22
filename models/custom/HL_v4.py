import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["HL_v4"]


def _auto_padding(kernel_size, dilation=1):
    """Same padding for int or tuple kernels."""
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
    """Efficient Channel Attention."""
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
    Same decoder block as HL_base.
    This keeps the comparison fair: only skip features are changed by HL_v4.
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


class RoadDirectionalBlock(nn.Module):
    """
    Road-oriented detail block.

    It replaces the plain OCL idea in HL-SAM-Seg with a road-aware structure:
    - 3x3 local branch keeps local edge texture.
    - 1xk and kx1 strip branches capture long thin horizontal/vertical roads.
    - dilated branch enlarges context for small broken road parts.

    The branches are summed rather than concatenated to keep 1024x1024 training affordable.
    """
    def __init__(self, in_channels, out_channels, k=7, dilation=2):
        super().__init__()
        self.local = ConvBNAct(in_channels, out_channels, kernel_size=3)
        self.horiz = ConvBNAct(in_channels, out_channels, kernel_size=(1, k))
        self.vert = ConvBNAct(in_channels, out_channels, kernel_size=(k, 1))
        self.dilated = ConvBNAct(in_channels, out_channels, kernel_size=3, dilation=dilation)
        self.fuse = nn.Sequential(
            ConvBNAct(out_channels, out_channels, kernel_size=1),
            ECALayer(out_channels),
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.local(x) + self.horiz(x) + self.vert(x) + self.dilated(x)
        y = self.fuse(y)
        return self.act(y + self.shortcut(x))


class RoadAwareHLGuidance(nn.Module):
    """
    Road-aware high-low guidance module.

    Compared with the HL module in HL-SAM-Seg, this module is changed for road extraction:
    - low/semantic feature provides a road gate to suppress non-road high-res textures.
    - high/detail feature provides road boundary and thin-road cues back to semantic feature.
    - directional block strengthens road continuity before feedback.

    Input and output keep the same channel number, so it can directly replace skip features.
    """
    def __init__(self, channels, hidden_channels=None, k=7, gamma_init=0.1):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = max(channels // 2, 32)

        self.low_proj = ConvBNAct(channels, hidden_channels, kernel_size=1)
        self.high_proj = ConvBNAct(channels, hidden_channels, kernel_size=1)

        self.semantic_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Sigmoid(),
        )
        self.detail_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, groups=hidden_channels, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.Sigmoid(),
        )

        self.dir_enhance = RoadDirectionalBlock(hidden_channels, hidden_channels, k=k)

        self.fuse = nn.Sequential(
            ConvBNAct(hidden_channels * 3, hidden_channels, kernel_size=1),
            ConvBNAct(hidden_channels, hidden_channels, kernel_size=3),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.out_eca = ECALayer(channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, low_sem, high_detail):
        if high_detail.shape[-2:] != low_sem.shape[-2:]:
            high_detail = F.interpolate(high_detail, size=low_sem.shape[-2:], mode="bilinear", align_corners=False)

        l = self.low_proj(low_sem)
        h = self.high_proj(high_detail)

        # Semantic-to-detail guidance: keep road-like details, suppress texture noise.
        sem_gate = self.semantic_gate(l)
        h_sem = h * (1.0 + sem_gate)

        # Direction-aware detail enhancement for thin and elongated roads.
        h_dir = self.dir_enhance(h_sem)

        # Detail-to-semantic feedback: inject thin road and boundary cues into low-res semantics.
        det_gate = self.detail_gate(h_dir)
        l_refined = l + h_dir * det_gate

        # Stable high-low agreement.
        # The first version used raw multiplication, which may create very large values
        # under AMP on 1024x1024 road images.  tanh keeps the agreement term bounded
        # while still preserving the idea of explicit high-low interaction.
        agreement = torch.tanh(l_refined) * torch.tanh(h_dir)
        fused = torch.cat([l_refined, h_dir, agreement], dim=1)
        delta = self.fuse(fused)
        delta = self.out_proj(delta)
        delta = self.out_eca(delta)

        # Bound the residual strength.  This keeps HL_v4 close to HL_base at the
        # beginning of training and prevents the guidance branch from destroying
        # the encoder skip features.
        gamma = 0.2 * torch.sigmoid(self.gamma)
        return self.act(low_sem + gamma * delta)


class SegHead(nn.Module):
    def __init__(self, in_channels, out_channels=1, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = min(in_channels, 64)
        self.head = nn.Sequential(
            ConvBNAct(in_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class HL_v4(nn.Module):
    """
    HL_v4: Road Structure-aware High-Low Guidance Network.

    This model is built from HL_base and modifies the skip features before decoding.
    It is designed for road extraction on DeepGlobe / Massachusetts Roads.

    Main changes over HL_base:
    1) RoadDirectionalBlock extracts road-oriented high-resolution details.
    2) Top-down semantic prior gives x1/x2 stronger road semantics.
    3) RoadAwareHLGuidance refines x1/x2/x3 skip features before the decoder.
    4) Optional deep supervision and boundary auxiliary output are returned for trainH.py.
    """
    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        num_classes=None,
        in_channels=None,
        pretrained=True,
        return_aux=False,
        deep_supervision=True,
        use_edge=True,
        guidance_gamma_init=0.1,
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
        self.deep_supervision = deep_supervision
        self.use_edge = use_edge

        encoder = self._get_resnet34(pretrained=pretrained)

        if n_channels != 3:
            self.input_adapter = nn.Conv2d(n_channels, 3, kernel_size=1, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # Encoder: same as HL_base.
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.maxpool = encoder.maxpool
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        # Road-oriented high-resolution detail path, corresponding to x1/x2/x3.
        self.rdp1 = RoadDirectionalBlock(64, 64, k=7)
        self.rdp2 = RoadDirectionalBlock(128, 128, k=7)
        self.rdp3 = RoadDirectionalBlock(256, 256, k=7)

        # Top-down semantic prior, making shallow skips less noisy.
        self.td32 = ConvBNAct(256, 128, kernel_size=1)
        self.td21 = ConvBNAct(128, 64, kernel_size=1)

        # Road-aware high-low guidance modules.
        self.rhl3 = RoadAwareHLGuidance(256, hidden_channels=128, k=7, gamma_init=guidance_gamma_init)
        self.rhl2 = RoadAwareHLGuidance(128, hidden_channels=64, k=7, gamma_init=guidance_gamma_init)
        self.rhl1 = RoadAwareHLGuidance(64, hidden_channels=32, k=7, gamma_init=guidance_gamma_init)

        # Decoder: same as HL_base.
        self.dec3 = ResidualDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec2 = ResidualDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec1 = ResidualDecoderBlock(in_channels=128, skip_channels=64, out_channels=96)
        self.dec0 = ResidualDecoderBlock(in_channels=96, skip_channels=64, out_channels=64)

        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

        # Auxiliary heads for road supervision at different decoder depths.
        if deep_supervision:
            self.aux3_head = SegHead(256, n_classes, mid_channels=64)
            self.aux2_head = SegHead(128, n_classes, mid_channels=64)
            self.aux1_head = SegHead(96, n_classes, mid_channels=64)

        # Boundary head. It is only used during training by trainH.py.
        if use_edge:
            self.edge_head = SegHead(64, n_classes, mid_channels=32)

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

    def forward_features(self, x):
        input_size = x.shape[-2:]
        x_in = self.input_adapter(x)

        # Encoder.
        x0 = self.stem(x_in)                    # H/2,  64
        x1 = self.layer1(self.maxpool(x0))      # H/4,  64
        x2 = self.layer2(x1)                    # H/8,  128
        x3 = self.layer3(x2)                    # H/16, 256
        x4 = self.layer4(x3)                    # H/32, 512

        # Road-oriented high-res detail features.
        r1 = self.rdp1(x1)
        r2 = self.rdp2(x2)
        r3 = self.rdp3(x3)

        # Top-down semantic prior. This gives shallow details more reliable road semantics.
        sem3 = x3
        sem2 = x2 + F.interpolate(self.td32(sem3), size=x2.shape[-2:], mode="bilinear", align_corners=False)
        sem1 = x1 + F.interpolate(self.td21(sem2), size=x1.shape[-2:], mode="bilinear", align_corners=False)

        # Road-aware high-low feature guidance.
        x3_r = self.rhl3(sem3, r3)              # H/16, 256
        x2_r = self.rhl2(sem2, r2)              # H/8,  128
        x1_r = self.rhl1(sem1, r1)              # H/4,   64

        # Decoder with refined skip features.
        d3 = self.dec3(x4, x3_r)                # H/16, 256
        d2 = self.dec2(d3, x2_r)                # H/8,  128
        d1 = self.dec1(d2, x1_r)                # H/4,   96
        d0 = self.dec0(d1, x0)                  # H/2,   64

        logits_half = self.out_head(d0)
        logits = F.interpolate(logits_half, size=input_size, mode="bilinear", align_corners=False)

        aux = {
            "logits": logits,
            "logits_half": logits_half,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "r1": r1,
            "r2": r2,
            "r3": r3,
            "x1_refined": x1_r,
            "x2_refined": x2_r,
            "x3_refined": x3_r,
            "d3": d3,
            "d2": d2,
            "d1": d1,
            "d0": d0,
        }

        if self.deep_supervision:
            aux["aux3"] = F.interpolate(self.aux3_head(d3), size=input_size, mode="bilinear", align_corners=False)
            aux["aux2"] = F.interpolate(self.aux2_head(d2), size=input_size, mode="bilinear", align_corners=False)
            aux["aux1"] = F.interpolate(self.aux1_head(d1), size=input_size, mode="bilinear", align_corners=False)

        if self.use_edge:
            edge_half = self.edge_head(d0)
            aux["edge_logits"] = F.interpolate(edge_half, size=input_size, mode="bilinear", align_corners=False)

        return logits, aux

    def forward(self, x):
        logits, aux = self.forward_features(x)
        if self.return_aux:
            return aux
        return logits


if __name__ == "__main__":
    model = HL_v4(n_channels=3, n_classes=1, pretrained=False, return_aux=True)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print("Input:", x.shape)
    print("Output keys:", y.keys())
    print("Logits:", y["logits"].shape)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"Trainable params: {trainable_params / 1e6:.2f} M")
