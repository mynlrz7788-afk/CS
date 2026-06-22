import os
import sys
import importlib
from typing import Dict, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["DC_V3_1"]


def _auto_padding(kernel_size, dilation=1):
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
    """
    Efficient Channel Attention.
    保持 HL_base 风格。
    """

    def __init__(self, channels, k_size=3):
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

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
        return x * y.expand_as(x)


class ResidualDecoderBlock(nn.Module):
    """
    HL_base 风格残差解码块。
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.fuse = nn.Sequential(
            ConvBNAct(in_channels + skip_channels, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3),
        )

        self.eca = ECALayer(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        feat = torch.cat([x, skip], dim=1)

        out = self.fuse(feat)
        out = self.eca(out)
        out = out + self.shortcut(feat)

        return self.act(out)


class FrozenDINOv3FeatureExtractor(nn.Module):
    """
    Frozen DINOv3 特征提取器。

    作用：
    1. 从本地 dinounet/dinov3 构建 DINOv3 backbone。
    2. 加载 DINOv3 预训练权重。
    3. 冻结 DINO 原始参数。
    4. 提取 F2、F5、F8、F11。
    5. 输出 2D feature map。
    """

    def __init__(
        self,
        dino_model_name="dinov3_vits16",
        dino_repo_path="/home/u2508183004/zyn/SEG/dinounet/dinov3",
        dino_ckpt_path="/home/u2508183004/zyn/SEG/weight/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        out_layers=(2, 5, 8, 11),
        embed_dim=384,
        patch_size=16,
        dino_normalize=False,
        dino_intermediate_norm=False,
    ):
        super().__init__()

        self.dino_model_name = dino_model_name
        self.dino_repo_path = dino_repo_path
        self.dino_ckpt_path = dino_ckpt_path
        self.out_layers = list(out_layers)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.dino_normalize = bool(dino_normalize)
        self.dino_intermediate_norm = bool(dino_intermediate_norm)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        self.backbone = self._build_dino_backbone()
        self._freeze_backbone()

        print(
            "[DC_V3_1] Frozen DINOv3 feature extractor ready. "
            f"out_layers={self.out_layers}"
        )

    def _build_dino_backbone(self):
        if not os.path.isdir(self.dino_repo_path):
            raise FileNotFoundError(f"找不到 dino_repo_path: {self.dino_repo_path}")

        dinov3_parent = os.path.dirname(self.dino_repo_path)

        if dinov3_parent not in sys.path:
            sys.path.insert(0, dinov3_parent)

        try:
            backbones = importlib.import_module("dinov3.hub.backbones")
        except Exception as e:
            raise ImportError(
                "导入 dinov3.hub.backbones 失败。请确认 dino_repo_path 指向 "
                ".../SEG/dinounet/dinov3。\n"
                f"原始错误: {repr(e)}"
            )

        if not hasattr(backbones, self.dino_model_name):
            available = [n for n in dir(backbones) if n.startswith("dinov3_")]
            raise AttributeError(
                f"在 dinov3.hub.backbones 中找不到模型: {self.dino_model_name}\n"
                f"可用模型示例: {available[:30]}"
            )

        model = getattr(backbones, self.dino_model_name)(pretrained=False)

        if self.dino_ckpt_path is not None and self.dino_ckpt_path != "":
            if not os.path.isfile(self.dino_ckpt_path):
                raise FileNotFoundError(f"找不到 DINOv3 权重文件: {self.dino_ckpt_path}")

            ckpt = torch.load(self.dino_ckpt_path, map_location="cpu")

            if isinstance(ckpt, dict):
                if "model" in ckpt:
                    state_dict = ckpt["model"]
                elif "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                elif "teacher" in ckpt:
                    state_dict = ckpt["teacher"]
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt

            clean_state = {}

            for k, v in state_dict.items():
                nk = k

                for prefix in (
                    "module.",
                    "backbone.",
                    "student.",
                    "teacher.",
                    "model.",
                ):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]

                clean_state[nk] = v

            missing, unexpected = model.load_state_dict(clean_state, strict=False)

            print(
                f"[DC_V3_1] 加载 DINOv3 权重完成: {self.dino_ckpt_path}\n"
                f"        missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
            )

            if len(missing) > 0:
                print(f"        missing 示例: {missing[:10]}")
            if len(unexpected) > 0:
                print(f"        unexpected 示例: {unexpected[:10]}")

        return model

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @staticmethod
    def _tokens_to_map(feat, h, w, patch_size):
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        if feat.dim() == 4:
            return feat

        if feat.dim() != 3:
            raise RuntimeError(f"DINO 特征维度异常: {feat.shape}")

        b, n, c = feat.shape
        patch_h = h // patch_size
        patch_w = w // patch_size
        patch_n = patch_h * patch_w

        if n > patch_n:
            special_n = n - patch_n
            feat = feat[:, special_n:, :]

        if feat.shape[1] != patch_n:
            raise RuntimeError(
                f"DINO token 无法 reshape。"
                f"当前 token 数={feat.shape[1]}, 预期 patch 数={patch_n}"
            )

        feat = feat.transpose(1, 2).contiguous()
        feat = feat.view(b, c, patch_h, patch_w)

        return feat

    @torch.no_grad()
    def forward(self, x):
        if self.dino_normalize:
            x = (x - self.mean) / self.std

        _, _, h, w = x.shape

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise RuntimeError(
                f"DINO 输入尺寸必须能被 patch_size 整除。"
                f"当前输入: {h}x{w}, patch_size={self.patch_size}"
            )

        if not hasattr(self.backbone, "get_intermediate_layers"):
            raise RuntimeError(
                "当前 DINOv3 backbone 没有 get_intermediate_layers。"
                "请检查 dinounet/dinov3/models/vision_transformer.py。"
            )

        try:
            feats = self.backbone.get_intermediate_layers(
                x,
                n=self.out_layers,
                reshape=True,
                return_class_token=False,
                norm=self.dino_intermediate_norm,
            )
        except TypeError:
            feats = self.backbone.get_intermediate_layers(
                x,
                n=self.out_layers,
                reshape=True,
                return_class_token=False,
            )

        if isinstance(feats, torch.Tensor):
            feats = [feats]

        feats = list(feats)

        if len(feats) != len(self.out_layers):
            raise RuntimeError(
                f"DINO 输出层数不匹配。期望 {len(self.out_layers)} 层，"
                f"实际得到 {len(feats)} 层。"
            )

        outputs = {}

        for layer, feat in zip(self.out_layers, feats):
            feat = self._tokens_to_map(
                feat,
                h=h,
                w=w,
                patch_size=self.patch_size,
            )
            outputs[int(layer)] = feat

        return outputs


class DirectionalSpatialGate(nn.Module):
    """
    轻量方向空间权重。

    不是直接用条形卷积生成道路特征，
    而是根据水平、垂直方向统计生成空间权重。
    """

    def __init__(self, channels, kernel_size=7):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.conv_w = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )

        self.conv_h = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape

        # 沿高度求均值，保留宽度方向响应
        stat_w = x.mean(dim=2)          # B, C, W
        # 沿宽度求均值，保留高度方向响应
        stat_h = x.mean(dim=3)          # B, C, H

        att_w = self.conv_w(stat_w).view(b, c, 1, w)
        att_h = self.conv_h(stat_h).view(b, c, h, 1)

        gate = self.sigmoid(att_w + att_h)

        return gate


class RoadLayerAdapter(nn.Module):
    """
    单个 DINO 层的轻量道路适配器。

    Fi
    ↓
    1×1 Conv 降维
    ↓
    dilation=1 和 dilation=3 的局部上下文
    ↓
    方向空间权重
    ↓
    通道筛选
    ↓
    1×1 Conv 升维
    ↓
    残差回写
    """

    def __init__(
        self,
        in_channels=384,
        bottleneck=64,
        alpha_init=0.01,
    ):
        super().__init__()

        self.reduce = ConvBNAct(
            in_channels,
            bottleneck,
            kernel_size=1,
            padding=0,
        )

        self.dw_d1 = ConvBNAct(
            bottleneck,
            bottleneck,
            kernel_size=3,
            dilation=1,
            groups=bottleneck,
        )

        self.dw_d3 = ConvBNAct(
            bottleneck,
            bottleneck,
            kernel_size=3,
            dilation=3,
            groups=bottleneck,
        )

        self.local_fuse = ConvBNAct(
            bottleneck,
            bottleneck,
            kernel_size=1,
            padding=0,
        )

        self.dir_gate = DirectionalSpatialGate(
            channels=bottleneck,
            kernel_size=7,
        )

        hidden = max(bottleneck // 4, 8)

        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(bottleneck, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, bottleneck, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.expand = nn.Sequential(
            nn.Conv2d(bottleneck, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self._init_weights()

    def _init_weights(self):
        # 让适配器初始阶段尽量接近恒等映射
        last_bn = self.expand[-1]
        nn.init.zeros_(last_bn.weight)
        nn.init.zeros_(last_bn.bias)

    def forward(self, x):
        z = self.reduce(x)

        l1 = self.dw_d1(z)
        l2 = self.dw_d3(z)
        local = self.local_fuse(l1 + l2)

        dir_weight = self.dir_gate(local)
        local = local * dir_weight

        ch_weight = self.channel_gate(local)
        local = local * ch_weight

        delta = self.expand(local)

        out = x + self.alpha * delta

        return out


class AdaptiveDinoLayerFusion(nn.Module):
    """
    F2、F5、F8、F11 的自适应层融合。

    不直接 concat。
    先对每一层做空间权重和通道权重，
    再生成 G16。
    """

    def __init__(
        self,
        in_channels=384,
        proj_channels=128,
        out_channels=256,
        dino_layers=(2, 5, 8, 11),
    ):
        super().__init__()

        self.dino_layers = list(dino_layers)
        self.num_layers = len(self.dino_layers)
        self.proj_channels = int(proj_channels)

        self.proj = nn.ModuleDict()
        self.spatial_score = nn.ModuleDict()

        for layer in self.dino_layers:
            key = str(int(layer))

            self.proj[key] = ConvBNAct(
                in_channels,
                proj_channels,
                kernel_size=1,
                padding=0,
            )

            self.spatial_score[key] = nn.Conv2d(
                proj_channels,
                1,
                kernel_size=1,
                bias=True,
            )

        hidden = max(proj_channels, 32)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(
                proj_channels * self.num_layers,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden,
                proj_channels * self.num_layers,
                kernel_size=1,
                bias=True,
            ),
        )

        self.out_proj = ConvBNAct(
            proj_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

    def forward(self, feats: Dict[int, torch.Tensor], target_size: Tuple[int, int]):
        proj_feats = []
        spatial_scores = []
        channel_descriptors = []

        for layer in self.dino_layers:
            key = str(int(layer))

            if int(layer) not in feats:
                raise KeyError(
                    f"LRD-Adapter 输入中缺少 layer={layer}。"
                    f"当前 keys={list(feats.keys())}"
                )

            f = feats[int(layer)]

            if f.shape[-2:] != target_size:
                f = F.interpolate(
                    f,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            p = self.proj[key](f)

            proj_feats.append(p)
            spatial_scores.append(self.spatial_score[key](p))
            channel_descriptors.append(F.adaptive_avg_pool2d(p, 1))

        # 空间层权重：B, L, 1, H, W
        spatial_scores = torch.cat(spatial_scores, dim=1)
        w_spatial = torch.softmax(spatial_scores, dim=1).unsqueeze(2)

        # 通道层权重：B, L, C, 1, 1
        channel_cat = torch.cat(channel_descriptors, dim=1)
        w_channel = self.channel_mlp(channel_cat)

        b = w_channel.shape[0]
        c = self.proj_channels

        w_channel = w_channel.view(
            b,
            self.num_layers,
            c,
            1,
            1,
        )
        w_channel = torch.softmax(w_channel, dim=1)

        # B, L, C, H, W
        feat_stack = torch.stack(proj_feats, dim=1)

        fused = (feat_stack * w_spatial * w_channel).sum(dim=1)

        g16 = self.out_proj(fused)

        return g16


class LRDAdapter(nn.Module):
    """
    Lightweight Road-aware DINO Adapter.

    输入：
        F2、F5、F8、F11

    输出：
        G16

    过程：
        每层单独 RoadLayerAdapter
        +
        空间层权重
        +
        通道层权重
        +
        自适应融合
    """

    def __init__(
        self,
        in_channels=384,
        bottleneck=64,
        proj_channels=128,
        out_channels=256,
        dino_layers=(2, 5, 8, 11),
        alpha_init=0.01,
    ):
        super().__init__()

        self.dino_layers = list(dino_layers)

        self.layer_adapters = nn.ModuleDict()

        for layer in self.dino_layers:
            key = str(int(layer))
            self.layer_adapters[key] = RoadLayerAdapter(
                in_channels=in_channels,
                bottleneck=bottleneck,
                alpha_init=alpha_init,
            )

        self.fusion = AdaptiveDinoLayerFusion(
            in_channels=in_channels,
            proj_channels=proj_channels,
            out_channels=out_channels,
            dino_layers=dino_layers,
        )

    def forward(self, dino_feats: Dict[int, torch.Tensor], target_size: Tuple[int, int]):
        adapted = {}

        for layer in self.dino_layers:
            layer = int(layer)
            key = str(layer)

            if layer not in dino_feats:
                raise KeyError(
                    f"DINO 输出中找不到 layer={layer}。"
                    f"当前 keys={list(dino_feats.keys())}"
                )

            adapted[layer] = self.layer_adapters[key](dino_feats[layer])

        g16 = self.fusion(adapted, target_size=target_size)

        return g16, adapted


class RoadDeformableCrossAttention(nn.Module):
    """
    PyTorch 版单层可变形交叉注意力。

    Q 来自 CNN feature。
    K/V 来自 DINO feature。

    为了不依赖额外 CUDA op，这里用 grid_sample 实现。
    """

    def __init__(
        self,
        query_channels,
        kv_channels,
        hidden_dim=256,
        num_heads=8,
        num_points=4,
        max_offset=4.0,
    ):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim 必须能被 num_heads 整除。"
                f"当前 hidden_dim={hidden_dim}, num_heads={num_heads}"
            )

        self.query_channels = int(query_channels)
        self.kv_channels = int(kv_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.num_points = int(num_points)
        self.head_dim = self.hidden_dim // self.num_heads
        self.max_offset = float(max_offset)

        self.q_proj = nn.Sequential(
            nn.Conv2d(self.query_channels, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.kv_proj = nn.Sequential(
            nn.Conv2d(self.kv_channels, self.hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.offset_conv = nn.Conv2d(
            self.hidden_dim,
            self.num_heads * self.num_points * 2,
            kernel_size=3,
            padding=1,
        )

        self.attn_conv = nn.Conv2d(
            self.hidden_dim,
            self.num_heads * self.num_points,
            kernel_size=3,
            padding=1,
        )

        self.out_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.query_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.query_channels),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        nn.init.zeros_(self.attn_conv.weight)
        nn.init.zeros_(self.attn_conv.bias)

        last_bn = self.out_proj[-1]
        nn.init.zeros_(last_bn.weight)
        nn.init.zeros_(last_bn.bias)

    @staticmethod
    def _make_base_grid(batch_size, height, width, device, dtype):
        ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)

        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return grid

    def forward(self, query_map, kv_map):
        b, _, hq, wq = query_map.shape
        _, _, hk, wk = kv_map.shape

        q = self.q_proj(query_map)
        v = self.kv_proj(kv_map)

        offsets = self.offset_conv(q)
        attn = self.attn_conv(q)

        offsets = offsets.view(
            b,
            self.num_heads,
            self.num_points,
            2,
            hq,
            wq,
        )
        offsets = offsets.permute(0, 1, 2, 4, 5, 3).contiguous()

        offsets = torch.tanh(offsets)

        if wk > 1:
            offset_x_scale = 2.0 * self.max_offset / float(wk - 1)
        else:
            offset_x_scale = 0.0

        if hk > 1:
            offset_y_scale = 2.0 * self.max_offset / float(hk - 1)
        else:
            offset_y_scale = 0.0

        scale = torch.tensor(
            [offset_x_scale, offset_y_scale],
            device=query_map.device,
            dtype=query_map.dtype,
        ).view(1, 1, 1, 1, 1, 2)

        offsets = offsets * scale

        base_grid = self._make_base_grid(
            batch_size=b,
            height=hq,
            width=wq,
            device=query_map.device,
            dtype=query_map.dtype,
        )
        base_grid = base_grid.view(b, 1, 1, hq, wq, 2)

        sample_grid = base_grid + offsets
        sample_grid = sample_grid.view(
            b * self.num_heads * self.num_points,
            hq,
            wq,
            2,
        )

        v = v.view(
            b,
            self.num_heads,
            self.head_dim,
            hk,
            wk,
        )

        v = v.unsqueeze(2).expand(
            b,
            self.num_heads,
            self.num_points,
            self.head_dim,
            hk,
            wk,
        )
        v = v.contiguous().view(
            b * self.num_heads * self.num_points,
            self.head_dim,
            hk,
            wk,
        )

        sampled = F.grid_sample(
            v,
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        sampled = sampled.view(
            b,
            self.num_heads,
            self.num_points,
            self.head_dim,
            hq,
            wq,
        )

        attn = attn.view(
            b,
            self.num_heads,
            self.num_points,
            hq,
            wq,
        )
        attn = torch.softmax(attn, dim=2)
        attn = attn.unsqueeze(3)

        out = (sampled * attn).sum(dim=2)
        out = out.contiguous().view(
            b,
            self.hidden_dim,
            hq,
            wq,
        )

        out = self.out_proj(out)

        return out


class DC_V3_1(nn.Module):
    """
    DC_V3_1:
    HL_base + Frozen DINO + LRD-Adapter + H/16 Road-DCA。

    版本特点：
    1. HL_base 作为主分割网络。
    2. DINO 原始主干全部冻结。
    3. 提取 F2、F5、F8、F11。
    4. 每个 DINO 层先经过轻量道路适配器。
    5. 通过空间层权重和通道层权重自适应融合为 G16。
    6. x3 作为 Q，G16 作为 K/V 做 Road-DCA16。
    7. 只增强 x3，不接 H/8。
    8. 解码器仍然使用 HL_base 风格。
    """

    def __init__(
        self,
        n_channels=3,
        n_classes=1,
        num_classes=None,
        in_channels=None,
        pretrained=True,
        return_aux=False,

        dino_model_name="dinov3_vits16",
        dino_repo_path="/home/u2508183004/zyn/SEG/dinounet/dinov3",
        dino_ckpt_path="/home/u2508183004/zyn/SEG/weight/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        dino_layers=(2, 5, 8, 11),
        dino_embed_dim=384,
        dino_patch_size=16,
        dino_normalize=False,
        dino_intermediate_norm=False,

        adapter_bottleneck=64,
        adapter_proj_channels=128,
        adapter_out_channels=256,
        adapter_alpha_init=0.01,

        dca_hidden_dim=256,
        dca_heads=8,
        dca_points=4,
        dca_max_offset=4.0,
        dca_init_scale=0.1,

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

        self.dino = FrozenDINOv3FeatureExtractor(
            dino_model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            out_layers=dino_layers,
            embed_dim=dino_embed_dim,
            patch_size=dino_patch_size,
            dino_normalize=dino_normalize,
            dino_intermediate_norm=dino_intermediate_norm,
        )

        self.lrd_adapter = LRDAdapter(
            in_channels=dino_embed_dim,
            bottleneck=adapter_bottleneck,
            proj_channels=adapter_proj_channels,
            out_channels=adapter_out_channels,
            dino_layers=dino_layers,
            alpha_init=adapter_alpha_init,
        )

        self.dca16 = RoadDeformableCrossAttention(
            query_channels=256,
            kv_channels=adapter_out_channels,
            hidden_dim=dca_hidden_dim,
            num_heads=dca_heads,
            num_points=dca_points,
            max_offset=dca_max_offset,
        )

        self.gamma16 = nn.Parameter(torch.tensor(float(dca_init_scale)))

        self.dec3 = ResidualDecoderBlock(
            in_channels=512,
            skip_channels=256,
            out_channels=256,
        )

        self.dec2 = ResidualDecoderBlock(
            in_channels=256,
            skip_channels=128,
            out_channels=128,
        )

        self.dec1 = ResidualDecoderBlock(
            in_channels=128,
            skip_channels=64,
            out_channels=96,
        )

        self.dec0 = ResidualDecoderBlock(
            in_channels=96,
            skip_channels=64,
            out_channels=64,
        )

        self.out_head = nn.Sequential(
            ConvBNAct(64, 64, kernel_size=3),
            ConvBNAct(64, 64, kernel_size=3),
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

        print(
            "[DC_V3_1] 构建完成：HL_base + Frozen DINO + LRD-Adapter + H/16 Road-DCA。"
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
                model = models.resnet34(
                    weights="IMAGENET1K_V1" if pretrained else None
                )

            return model

    def train(self, mode=True):
        super().train(mode)
        self.dino.train(False)
        return self

    def forward_features(self, x):
        input_size = x.shape[-2:]

        x_in = self.input_adapter(x)

        # CNN encoder
        x0 = self.stem(x_in)                    # B, 64,  H/2
        x1 = self.layer1(self.maxpool(x0))      # B, 64,  H/4
        x2 = self.layer2(x1)                    # B, 128, H/8
        x3 = self.layer3(x2)                    # B, 256, H/16
        x4 = self.layer4(x3)                    # B, 512, H/32

        # Frozen DINO
        dino_feats = self.dino(x_in)

        # LRD-Adapter
        g16, adapted_feats = self.lrd_adapter(
            dino_feats=dino_feats,
            target_size=x3.shape[-2:],
        )

        # H/16 Road-DCA
        a16 = self.dca16(
            query_map=x3,
            kv_map=g16,
        )

        x3_enhanced = x3 + self.gamma16 * a16

        # HL_base decoder
        d3 = self.dec3(x4, x3_enhanced)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)
        d0 = self.dec0(d1, x0)

        logits_half = self.out_head(d0)

        logits = F.interpolate(
            logits_half,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        aux = {
            "logits_half": logits_half,
            "x0": x0,
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "x3_enhanced": x3_enhanced,
            "g16": g16,
            "d3": d3,
            "d2": d2,
            "d1": d1,
            "d0": d0,
            "gamma16": self.gamma16.detach(),
        }

        for k, v in adapted_feats.items():
            aux[f"adapted_F{k}"] = v

        return logits, aux

    def forward_train(self, x):
        """
        训练专用 forward。

        V1 只使用 final_logits 和 logits_half：
            L_total = L_final + lambda_half * L_half

        这里同时返回 d3/d2/d1/g16，方便后续 V2/V3 继续扩展，
        但 V1 的 train_DC 不会使用它们。
        """
        logits, aux = self.forward_features(x)

        return {
            "final_logits": logits,
            "logits_half": aux.get("logits_half", None),
            "g16": aux.get("g16", None),
            "d3": aux.get("d3", None),
            "d2": aux.get("d2", None),
            "d1": aux.get("d1", None),
            "d0": aux.get("d0", None),
        }

    def forward(self, x):
        logits, aux = self.forward_features(x)

        if self.return_aux:
            aux["final_logits"] = logits
            return aux

        return logits


if __name__ == "__main__":
    model = DC_V3_1(
        n_channels=3,
        n_classes=1,
        pretrained=False,
        return_aux=False,
        dino_ckpt_path="",
    )

    x = torch.randn(1, 3, 256, 256)
    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)