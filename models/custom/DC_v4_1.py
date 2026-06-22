import os
import sys
import math
import importlib
from typing import Dict, Tuple, Sequence, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["DC_v4_1"]


def _auto_padding(kernel_size, dilation=1):
    if isinstance(kernel_size, tuple):
        if isinstance(dilation, tuple):
            return tuple(((k - 1) // 2) * d for k, d in zip(kernel_size, dilation))
        return tuple(((k - 1) // 2) * dilation for k in kernel_size)

    return ((kernel_size - 1) // 2) * dilation


def _make_gn(channels: int, max_groups: int = 8):
    """GroupNorm helper. 对小 batch / 小特征图更稳。"""
    channels = int(channels)
    groups = min(int(max_groups), channels)

    while groups > 1 and channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(groups, channels)


class ConvBNAct(nn.Module):
    """
    保持 HL_base / DC_v3 风格的 Conv-BN-ReLU。
    主要用于 CNN decoder，保证和旧实验公平。
    """
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

        if act_layer is nn.ReLU:
            act = act_layer(inplace=inplace)
        else:
            act = act_layer()

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
            act,
        )

    def forward(self, x):
        return self.block(x)


class ConvGNAct(nn.Module):
    """
    DINO branch 专用 Conv-GN-GELU。
    比 BatchNorm 更适合 128/256 小输入和小 batch。
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        dilation=1,
        groups=1,
        act_layer=nn.GELU,
        gn_groups=8,
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
            _make_gn(out_channels, gn_groups),
            act_layer(),
        )

    def forward(self, x):
        return self.block(x)


class ECALayer(nn.Module):
    """
    Efficient Channel Attention.
    和 HL_base / DC_v3 保持一致。
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
    HL_base 风格残差解码块：
        upsample decoder feature
        -> concat skip
        -> 2×ConvBNAct
        -> ECA
        -> shortcut
        -> residual add
        -> ReLU
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


class FrozenDINOv3TokenExtractor(nn.Module):
    """
    Frozen DINOv3 token extractor.

    严格按文档思路：
        Input image
        -> Frozen DINOv3
        -> T2/T5/T8/T11 token sequence

    输出:
        {
            layer_id: B × N × C
        }

    其中:
        N = H/patch_size × W/patch_size

    注意:
        1. DINO 原始参数全部冻结。
        2. forward 使用 no_grad。
        3. 优先取 reshape=False 的 token sequence。
        4. 如果某些 DINOv3 版本返回 4D feature map，则自动 flatten 回 token。
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
            "[DC_v4_1] Frozen DINOv3 token extractor ready. "
            f"out_layers={self.out_layers}, patch_size={self.patch_size}"
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
                f"[DC_v4_1] 加载 DINOv3 权重完成: {self.dino_ckpt_path}\n"
                f"          missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
            )

            if len(missing) > 0:
                print(f"          missing 示例: {missing[:10]}")
            if len(unexpected) > 0:
                print(f"          unexpected 示例: {unexpected[:10]}")

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
    def _feat_to_tokens(feat, h, w, patch_size):
        if isinstance(feat, (tuple, list)):
            feat = feat[0]

        patch_h = h // patch_size
        patch_w = w // patch_size
        patch_n = patch_h * patch_w

        if feat.dim() == 4:
            # B × C × Hp × Wp -> B × N × C
            b, c, fh, fw = feat.shape

            if (fh, fw) != (patch_h, patch_w):
                feat = F.interpolate(
                    feat,
                    size=(patch_h, patch_w),
                    mode="bilinear",
                    align_corners=False,
                )

            tokens = feat.flatten(2).transpose(1, 2).contiguous()
            return tokens

        if feat.dim() != 3:
            raise RuntimeError(f"DINO 特征维度异常: {feat.shape}")

        b, n, c = feat.shape

        # 如果含 cls/register token，只保留最后 patch_n 个 patch tokens。
        if n > patch_n:
            special_n = n - patch_n
            feat = feat[:, special_n:, :]

        if feat.shape[1] != patch_n:
            raise RuntimeError(
                f"DINO token 无法对齐 patch grid。"
                f"当前 token 数={feat.shape[1]}, 预期 patch 数={patch_n}, "
                f"输入尺寸={h}x{w}, patch_size={patch_size}"
            )

        return feat.contiguous()

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
                reshape=False,
                return_class_token=False,
                norm=self.dino_intermediate_norm,
            )
        except TypeError:
            try:
                feats = self.backbone.get_intermediate_layers(
                    x,
                    n=self.out_layers,
                    reshape=False,
                    return_class_token=False,
                )
            except TypeError:
                # 兜底：部分版本不支持 reshape=False，就先取 2D feature 再 flatten。
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
            tokens = self._feat_to_tokens(
                feat,
                h=h,
                w=w,
                patch_size=self.patch_size,
            )
            outputs[int(layer)] = tokens

        return outputs


class TokenSemanticCalibration(nn.Module):
    """
    Token Semantic Calibration.

    T_l:
        B × N × C
    输出:
        T_l + alpha · MLP(LN(T_l))

    alpha 初始很小，避免一开始破坏 frozen DINO 表征。
    """
    def __init__(
        self,
        embed_dim=384,
        bottleneck=64,
        alpha_init=0.01,
        dropout=0.0,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, bottleneck, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, embed_dim, bias=False),
        )

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self._init_weights()

    def _init_weights(self):
        # 最后一层零初始化，使初始接近恒等映射。
        last = self.mlp[-1]
        nn.init.zeros_(last.weight)

    def forward(self, tokens):
        delta = self.mlp(self.norm(tokens))
        return tokens + self.alpha * delta


class StructureQueryBridge(nn.Module):
    """
    Structure Query Bridge.

    用一组 learnable queries 从 DINO token 中读取结构信息:
        Q_global / Q_connect / Q_boundary / Q_suppress

    输入:
        T_l_calib: B × N × C

    输出:
        Z_l: B × num_queries × C

    默认 num_queries=32，四组各 8 个 query。
    对 128×128 小输入，可以在 json 中设置 structure_queries=16。
    """
    def __init__(
        self,
        embed_dim=384,
        num_queries=32,
        num_heads=8,
        mlp_ratio=2.0,
        dropout=0.0,
    ):
        super().__init__()

        if num_queries % 4 != 0:
            raise ValueError(
                f"structure_queries 必须能被 4 整除，当前为 {num_queries}"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim 必须能被 num_heads 整除。"
                f"当前 embed_dim={embed_dim}, num_heads={num_heads}"
            )

        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)

        self.structure_queries = nn.Parameter(
            torch.zeros(1, num_queries, embed_dim)
        )

        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        hidden = int(embed_dim * mlp_ratio)

        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim, bias=False),
        )

        self.gamma_attn = nn.Parameter(torch.tensor(1.0))
        self.gamma_ffn = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.structure_queries, std=0.02)
        last = self.ffn[-1]
        nn.init.zeros_(last.weight)

    def forward(self, tokens):
        b = tokens.shape[0]

        q = self.structure_queries.expand(b, -1, -1)
        q_norm = self.q_norm(q)
        kv = self.kv_norm(tokens)

        attn_out, _ = self.attn(
            query=q_norm,
            key=kv,
            value=kv,
            need_weights=False,
        )

        z = q + self.gamma_attn * attn_out
        z = z + self.gamma_ffn * self.ffn(z)

        return z


class DirectionalGridMixer(nn.Module):
    """
    Grid Directional Mixing.

    在 DINO patch grid 上做道路方向结构混合：
        horizontal: 1×k
        vertical:   k×1
        diagonal:   3×3 dilation=2
        local:      3×3

    输入 / 输出:
        B × C × Hp × Wp
    """
    def __init__(
        self,
        channels=128,
        kernel_size=7,
        beta_init=0.1,
        gn_groups=8,
    ):
        super().__init__()

        padding = (kernel_size - 1) // 2

        self.branch_h = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=channels,
            bias=False,
        )

        self.branch_v = nn.Conv2d(
            channels,
            channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=channels,
            bias=False,
        )

        self.branch_d = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            groups=channels,
            bias=False,
        )

        self.branch_l = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False),
            _make_gn(channels, gn_groups),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            _make_gn(channels, gn_groups),
        )

        self.act = nn.GELU()
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, x):
        h = self.branch_h(x)
        v = self.branch_v(x)
        d = self.branch_d(x)
        l = self.branch_l(x)

        y = torch.cat([h, v, d, l], dim=1)
        y = self.fuse(y)

        return self.act(x + self.beta * y)


class TokenLayerAdapter(nn.Module):
    """
    单个 DINO 层的 Token-Structure Adapter。

    T_l
    -> Token Semantic Calibration
    -> Token-to-Grid
    -> 1×1 projection
    -> Grid Directional Mixing
    -> A_l
    """
    def __init__(
        self,
        embed_dim=384,
        token_bottleneck=64,
        adapter_channels=128,
        alpha_init=0.01,
        beta_init=0.1,
        dropout=0.0,
        gn_groups=8,
    ):
        super().__init__()

        self.embed_dim = int(embed_dim)
        self.adapter_channels = int(adapter_channels)

        self.calibration = TokenSemanticCalibration(
            embed_dim=embed_dim,
            bottleneck=token_bottleneck,
            alpha_init=alpha_init,
            dropout=dropout,
        )

        self.proj = ConvGNAct(
            embed_dim,
            adapter_channels,
            kernel_size=1,
            padding=0,
            gn_groups=gn_groups,
        )

        self.grid_mixer = DirectionalGridMixer(
            channels=adapter_channels,
            kernel_size=7,
            beta_init=beta_init,
            gn_groups=gn_groups,
        )

    @staticmethod
    def _tokens_to_grid(tokens, patch_hw):
        b, n, c = tokens.shape
        hp, wp = patch_hw

        if n != hp * wp:
            raise RuntimeError(
                f"Token 数和 patch grid 不匹配: tokens={n}, patch_hw={patch_hw}"
            )

        grid = tokens.transpose(1, 2).contiguous().view(b, c, hp, wp)

        return grid

    def forward(self, tokens, patch_hw):
        t_calib = self.calibration(tokens)

        grid = self._tokens_to_grid(t_calib, patch_hw)
        grid = self.proj(grid)
        a = self.grid_mixer(grid)

        return a, t_calib


class DinoStructurePyramid(nn.Module):
    """
    文档版 DINO structure pyramid。

    输入:
        T2/T5/T8/T11

    输出:
        A2/A5/A8/A11:
            B × adapter_channels × Hp × Wp

        Z2/Z5/Z8/Z11:
            B × structure_queries × embed_dim
    """
    def __init__(
        self,
        dino_layers=(2, 5, 8, 11),
        embed_dim=384,
        token_bottleneck=64,
        adapter_channels=128,
        structure_queries=32,
        structure_heads=8,
        structure_mlp_ratio=2.0,
        adapter_alpha_init=0.01,
        adapter_beta_init=0.1,
        dropout=0.0,
        gn_groups=8,
    ):
        super().__init__()

        self.dino_layers = [int(x) for x in dino_layers]

        self.layer_adapters = nn.ModuleDict()

        for layer in self.dino_layers:
            key = str(layer)

            self.layer_adapters[key] = TokenLayerAdapter(
                embed_dim=embed_dim,
                token_bottleneck=token_bottleneck,
                adapter_channels=adapter_channels,
                alpha_init=adapter_alpha_init,
                beta_init=adapter_beta_init,
                dropout=dropout,
                gn_groups=gn_groups,
            )

        # 共享结构 queries，保证 global/connect/boundary/suppress 语义在层间对齐。
        self.query_bridge = StructureQueryBridge(
            embed_dim=embed_dim,
            num_queries=structure_queries,
            num_heads=structure_heads,
            mlp_ratio=structure_mlp_ratio,
            dropout=dropout,
        )

    def forward(self, token_dict: Dict[int, torch.Tensor], patch_hw: Tuple[int, int]):
        a_dict = {}
        z_dict = {}

        for layer in self.dino_layers:
            if layer not in token_dict:
                raise KeyError(
                    f"DINO token 输出中缺少 layer={layer}。"
                    f"当前 keys={list(token_dict.keys())}"
                )

            key = str(layer)

            a, t_calib = self.layer_adapters[key](
                token_dict[layer],
                patch_hw=patch_hw,
            )

            z = self.query_bridge(t_calib)

            a_dict[layer] = a
            z_dict[layer] = z

        return a_dict, z_dict


class StageAwareRouter(nn.Module):
    """
    Stage-aware layer router.

    输入:
        A_stack: B × L × C × Hp × Wp
        Z_dict:  每层 B × Q × D

    输出:
        R_stage: B × C × Hp × Wp
        W_stage: B × L × C × Hp × Wp

    W_stage = softmax(P + C + S + Q, dim=1)
    """
    def __init__(
        self,
        stage: str,
        dino_layers=(2, 5, 8, 11),
        adapter_channels=128,
        embed_dim=384,
        structure_queries=32,
        router_hidden=512,
        logit_scale_init=0.1,
    ):
        super().__init__()

        stage = str(stage)

        if stage not in ("16", "8", "4"):
            raise ValueError(f"stage 必须是 '16' / '8' / '4'，当前为 {stage}")

        if structure_queries % 4 != 0:
            raise ValueError(
                f"structure_queries 必须能被 4 整除，当前为 {structure_queries}"
            )

        self.stage = stage
        self.dino_layers = [int(x) for x in dino_layers]
        self.num_layers = len(self.dino_layers)
        self.adapter_channels = int(adapter_channels)
        self.embed_dim = int(embed_dim)
        self.structure_queries = int(structure_queries)
        self.group_queries = self.structure_queries // 4

        if self.stage == "16":
            prior = [0.05, 0.15, 0.40, 0.40]
        elif self.stage == "8":
            prior = [0.10, 0.30, 0.40, 0.20]
        else:
            prior = [0.35, 0.35, 0.20, 0.10]

        if len(prior) != self.num_layers:
            raise ValueError(
                f"当前 prior 长度={len(prior)}，但 dino_layers 数量={self.num_layers}。"
                "如果改了 dino_layers，需要同步修改 StageAwareRouter 的 prior。"
            )

        prior_tensor = torch.tensor(prior, dtype=torch.float32).view(
            1,
            self.num_layers,
            1,
            1,
            1,
        )

        self.stage_prior = nn.Parameter(torch.log(prior_tensor))

        channel_in = self.num_layers * self.adapter_channels
        channel_hidden = max(int(router_hidden), self.adapter_channels)

        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(channel_in),
            nn.Linear(channel_in, channel_hidden, bias=False),
            nn.GELU(),
            nn.Linear(channel_hidden, self.num_layers * self.adapter_channels),
        )

        self.spatial_score = nn.ModuleList([
            nn.Conv2d(self.adapter_channels, 1, kernel_size=1, bias=True)
            for _ in range(self.num_layers)
        ])

        if self.stage == "4":
            query_in = self.num_layers * self.embed_dim * 2
        else:
            query_in = self.num_layers * self.embed_dim

        query_hidden = max(int(router_hidden), self.embed_dim)

        self.query_mlp = nn.Sequential(
            nn.LayerNorm(query_in),
            nn.Linear(query_in, query_hidden, bias=False),
            nn.GELU(),
            nn.Linear(query_hidden, self.num_layers * self.adapter_channels),
        )

        # 小尺度初始化，让 prior 先发挥作用，训练中逐渐学习动态路由。
        self.scale_channel = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.scale_spatial = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.scale_query = nn.Parameter(torch.tensor(float(logit_scale_init)))

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.channel_mlp[-1].weight)
        nn.init.zeros_(self.channel_mlp[-1].bias)
        nn.init.zeros_(self.query_mlp[-1].weight)
        nn.init.zeros_(self.query_mlp[-1].bias)

        for conv in self.spatial_score:
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)

    def _extract_query_vector(self, z_dict: Dict[int, torch.Tensor]):
        """
        Z_l 分组:
            0: group       global
            1: group       connect
            2: group       boundary
            3: group       suppress
        """
        vecs = []

        g = self.group_queries

        for layer in self.dino_layers:
            if layer not in z_dict:
                raise KeyError(
                    f"Z_dict 中缺少 layer={layer}。当前 keys={list(z_dict.keys())}"
                )

            z = z_dict[layer]

            if z.shape[1] != self.structure_queries:
                raise RuntimeError(
                    f"Structure query 数量不匹配。"
                    f"期望 {self.structure_queries}, 实际 {z.shape[1]}"
                )

            z_global = z[:, 0:g, :].mean(dim=1)
            z_connect = z[:, g:2 * g, :].mean(dim=1)
            z_boundary = z[:, 2 * g:3 * g, :].mean(dim=1)
            z_suppress = z[:, 3 * g:4 * g, :].mean(dim=1)

            if self.stage == "16":
                vecs.append(z_global)
            elif self.stage == "8":
                vecs.append(z_connect)
            else:
                vecs.append(torch.cat([z_boundary, z_suppress], dim=1))

        return torch.cat(vecs, dim=1)

    def forward(self, a_stack: torch.Tensor, z_dict: Dict[int, torch.Tensor]):
        b, l, c, hp, wp = a_stack.shape

        if l != self.num_layers:
            raise RuntimeError(
                f"A_stack layer 数不匹配。期望 {self.num_layers}, 实际 {l}"
            )

        if c != self.adapter_channels:
            raise RuntimeError(
                f"A_stack channel 不匹配。期望 {self.adapter_channels}, 实际 {c}"
            )

        # Channel routing: B × L × C × 1 × 1
        channel_desc = a_stack.mean(dim=(-1, -2)).reshape(b, l * c)
        channel_logits = self.channel_mlp(channel_desc)
        channel_logits = channel_logits.view(b, l, c, 1, 1)

        # Spatial routing: B × L × 1 × Hp × Wp
        spatial_logits = []

        for i in range(l):
            spatial_logits.append(self.spatial_score[i](a_stack[:, i]))

        spatial_logits = torch.stack(spatial_logits, dim=1)

        # Query routing: B × L × C × 1 × 1
        query_vec = self._extract_query_vector(z_dict)
        query_logits = self.query_mlp(query_vec)
        query_logits = query_logits.view(b, l, c, 1, 1)

        logits = (
            self.stage_prior
            + self.scale_channel * channel_logits
            + self.scale_spatial * spatial_logits
            + self.scale_query * query_logits
        )

        weights = torch.softmax(logits, dim=1)
        routed = (weights * a_stack).sum(dim=1)

        return routed, weights


class ResidualRefineGN(nn.Module):
    """
    DINO guidance projection 中的轻量 residual refinement。
    """
    def __init__(self, channels, gn_groups=8):
        super().__init__()

        self.conv1 = ConvGNAct(
            channels,
            channels,
            kernel_size=3,
            gn_groups=gn_groups,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _make_gn(channels, gn_groups),
        )

        self.act = nn.GELU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return self.act(x + y)


class StageProjectionHead(nn.Module):
    """
    R16/R8/R4 -> G16/G8/G4.

    G16 对齐 x3: H/16, 256 channels
    G8  对齐 x2: H/8,  128 channels
    G4  对齐 x1: H/4,  64 channels

    所有 spatial size 都用 target_size 动态对齐，不写死 64/128/256。
    """
    def __init__(
        self,
        in_channels=128,
        g16_channels=256,
        g8_channels=128,
        g4_channels=64,
        gn_groups=8,
    ):
        super().__init__()

        self.g16_proj = nn.Sequential(
            ConvGNAct(
                in_channels,
                g16_channels,
                kernel_size=1,
                padding=0,
                gn_groups=gn_groups,
            ),
            DirectionalGridMixer(
                channels=g16_channels,
                kernel_size=7,
                beta_init=0.1,
                gn_groups=gn_groups,
            ),
            ResidualRefineGN(g16_channels, gn_groups=gn_groups),
        )

        self.g8_proj = ConvGNAct(
            in_channels,
            g8_channels,
            kernel_size=1,
            padding=0,
            gn_groups=gn_groups,
        )

        self.g8_refine = nn.Sequential(
            ResidualRefineGN(g8_channels, gn_groups=gn_groups),
            DirectionalGridMixer(
                channels=g8_channels,
                kernel_size=7,
                beta_init=0.1,
                gn_groups=gn_groups,
            ),
        )

        self.g4_proj = ConvGNAct(
            in_channels,
            g4_channels,
            kernel_size=1,
            padding=0,
            gn_groups=gn_groups,
        )

        self.g4_refine8 = ResidualRefineGN(
            g4_channels,
            gn_groups=gn_groups,
        )

        self.g4_refine4 = nn.Sequential(
            ResidualRefineGN(g4_channels, gn_groups=gn_groups),
            DirectionalGridMixer(
                channels=g4_channels,
                kernel_size=7,
                beta_init=0.1,
                gn_groups=gn_groups,
            ),
        )

    @staticmethod
    def _resize(x, size):
        if x.shape[-2:] == size:
            return x

        return F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    def forward(
        self,
        r16,
        r8,
        r4,
        target16: Tuple[int, int],
        target8: Tuple[int, int],
        target4: Tuple[int, int],
    ):
        g16 = self.g16_proj(r16)
        g16 = self._resize(g16, target16)

        g8 = self.g8_proj(r8)
        g8 = self._resize(g8, target8)
        g8 = self.g8_refine(g8)

        g4 = self.g4_proj(r4)
        g4 = self._resize(g4, target8)
        g4 = self.g4_refine8(g4)
        g4 = self._resize(g4, target4)
        g4 = self.g4_refine4(g4)

        return g16, g8, g4


class SimpleDinoInjector(nn.Module):
    """
    极简 CNN-DINO interaction:
        x_hat = x + gamma · refine(G)

    不做 Cross-Attention / DCA / Mutual Guidance。
    让实验重点落在 DINO feature selection 是否有效。
    """
    def __init__(
        self,
        channels,
        gamma_init=0.1,
        gn_groups=8,
    ):
        super().__init__()

        self.refine = ResidualRefineGN(
            channels,
            gn_groups=gn_groups,
        )

        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(
                g,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return x + self.gamma * self.refine(g)


class DC_v4_1(nn.Module):
    """
    DC_v4_1: DINO-centric Structure Pyramid + Simple CNN Injection.

    核心流程:
        I
        -> CNN encoder: x0/x1/x2/x3/x4
        -> Frozen DINOv3 tokens: T2/T5/T8/T11
        -> Token-Structure Adapter: A2/A5/A8/A11 + Z2/Z5/Z8/Z11
        -> Stage-aware Router:
              W16 -> R16
              W8  -> R8
              W4  -> R4
        -> Stage Projection:
              R16 -> G16 对齐 x3
              R8  -> G8  对齐 x2
              R4  -> G4  对齐 x1
        -> Simple residual injection:
              x3_hat = x3 + gamma16 * G16
              x2_hat = x2 + gamma8  * G8
              x1_hat = x1 + gamma4  * G4
        -> HL_base decoder
        -> logits

    输入尺寸支持:
        1024 / 512 / 256 / 128
    只要求 H/W 能被 dino_patch_size=16 整除。
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

        adapter_channels=128,
        adapter_bottleneck=64,
        adapter_alpha_init=0.01,
        adapter_beta_init=0.1,

        structure_queries=32,
        structure_heads=8,
        structure_mlp_ratio=2.0,
        structure_dropout=0.0,

        router_hidden=512,
        router_logit_scale_init=0.1,

        injector_gamma_init=0.1,
        gn_groups=8,

        enable_dino_prior_head=True,

        **kwargs,
    ):
        super().__init__()

        if in_channels is not None:
            n_channels = in_channels

        if num_classes is not None:
            n_classes = num_classes

        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.return_aux = bool(return_aux)
        self.dino_layers = [int(x) for x in dino_layers]
        self.dino_patch_size = int(dino_patch_size)
        self.enable_dino_prior_head = bool(enable_dino_prior_head)

        encoder = self._get_resnet34(pretrained=pretrained)

        if self.n_channels != 3:
            self.input_adapter = nn.Conv2d(
                self.n_channels,
                3,
                kernel_size=1,
                bias=False,
            )
        else:
            self.input_adapter = nn.Identity()

        # CNN encoder: 和 HL_base / DC_v3 保持一致。
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

        # Frozen DINO token branch.
        self.dino = FrozenDINOv3TokenExtractor(
            dino_model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            out_layers=dino_layers,
            embed_dim=dino_embed_dim,
            patch_size=dino_patch_size,
            dino_normalize=dino_normalize,
            dino_intermediate_norm=dino_intermediate_norm,
        )

        self.dino_pyramid = DinoStructurePyramid(
            dino_layers=dino_layers,
            embed_dim=dino_embed_dim,
            token_bottleneck=adapter_bottleneck,
            adapter_channels=adapter_channels,
            structure_queries=structure_queries,
            structure_heads=structure_heads,
            structure_mlp_ratio=structure_mlp_ratio,
            adapter_alpha_init=adapter_alpha_init,
            adapter_beta_init=adapter_beta_init,
            dropout=structure_dropout,
            gn_groups=gn_groups,
        )

        self.router16 = StageAwareRouter(
            stage="16",
            dino_layers=dino_layers,
            adapter_channels=adapter_channels,
            embed_dim=dino_embed_dim,
            structure_queries=structure_queries,
            router_hidden=router_hidden,
            logit_scale_init=router_logit_scale_init,
        )

        self.router8 = StageAwareRouter(
            stage="8",
            dino_layers=dino_layers,
            adapter_channels=adapter_channels,
            embed_dim=dino_embed_dim,
            structure_queries=structure_queries,
            router_hidden=router_hidden,
            logit_scale_init=router_logit_scale_init,
        )

        self.router4 = StageAwareRouter(
            stage="4",
            dino_layers=dino_layers,
            adapter_channels=adapter_channels,
            embed_dim=dino_embed_dim,
            structure_queries=structure_queries,
            router_hidden=router_hidden,
            logit_scale_init=router_logit_scale_init,
        )

        self.stage_projection = StageProjectionHead(
            in_channels=adapter_channels,
            g16_channels=256,
            g8_channels=128,
            g4_channels=64,
            gn_groups=gn_groups,
        )

        self.inject16 = SimpleDinoInjector(
            channels=256,
            gamma_init=injector_gamma_init,
            gn_groups=gn_groups,
        )

        self.inject8 = SimpleDinoInjector(
            channels=128,
            gamma_init=injector_gamma_init,
            gn_groups=gn_groups,
        )

        self.inject4 = SimpleDinoInjector(
            channels=64,
            gamma_init=injector_gamma_init,
            gn_groups=gn_groups,
        )

        # HL_base decoder.
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
            nn.Conv2d(64, self.n_classes, kernel_size=1),
        )

        if self.enable_dino_prior_head:
            # 可选 DINO-side direct supervision。
            # 普通 train.py 不会用；train_DC.py / forward_train 可用它进一步约束 DINO guidance。
            self.dino_prior_head = nn.Sequential(
                ConvGNAct(64, 64, kernel_size=3, gn_groups=gn_groups),
                nn.Conv2d(64, self.n_classes, kernel_size=1),
            )
        else:
            self.dino_prior_head = None

        self._print_trainable_summary()

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

    def _print_trainable_summary(self):
        total = 0
        trainable = 0
        dino_trainable = 0

        for p in self.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

        for p in self.dino.parameters():
            if p.requires_grad:
                dino_trainable += p.numel()

        print("--------------------------------------------------")
        print("🔧 DC_v4_1 DINO-centric Structure Pyramid")
        print(f"    - 总参数量:       {total / 1e6:.2f} M")
        print(f"    - 可训练参数量:   {trainable / 1e6:.2f} M")
        print(f"    - DINO原始主干可训练: {dino_trainable / 1e6:.2f} M")
        print("    - 说明: DINOv3 原始主干冻结；DINO侧做 Token/A/W/G；CNN侧简单残差注入")
        print("--------------------------------------------------")

    def train(self, mode=True):
        super().train(mode)
        # 无论外部怎么切 train/eval，DINO 原始主干始终 eval。
        self.dino.train(False)
        return self

    def _build_a_stack(self, a_dict: Dict[int, torch.Tensor]):
        feats = []

        for layer in self.dino_layers:
            if layer not in a_dict:
                raise KeyError(
                    f"a_dict 缺少 layer={layer}。当前 keys={list(a_dict.keys())}"
                )
            feats.append(a_dict[layer])

        return torch.stack(feats, dim=1)

    def forward_features(self, x, need_aux=False):
        input_size = x.shape[-2:]

        if input_size[0] % self.dino_patch_size != 0 or input_size[1] % self.dino_patch_size != 0:
            raise RuntimeError(
                f"DC_v4_1 要求输入 H/W 能被 dino_patch_size 整除。"
                f"当前输入={input_size}, patch_size={self.dino_patch_size}"
            )

        x_in = self.input_adapter(x)

        # CNN encoder.
        x0 = self.stem(x_in)                    # B, 64,  H/2
        x1 = self.layer1(self.maxpool(x0))      # B, 64,  H/4
        x2 = self.layer2(x1)                    # B, 128, H/8
        x3 = self.layer3(x2)                    # B, 256, H/16
        x4 = self.layer4(x3)                    # B, 512, H/32

        patch_hw = x3.shape[-2:]

        # Frozen DINO tokens.
        token_dict = self.dino(x_in)

        # Token-Structure Adapter.
        a_dict, z_dict = self.dino_pyramid(
            token_dict=token_dict,
            patch_hw=patch_hw,
        )

        a_stack = self._build_a_stack(a_dict)

        # Stage-aware routing.
        r16, w16 = self.router16(a_stack, z_dict)
        r8, w8 = self.router8(a_stack, z_dict)
        r4, w4 = self.router4(a_stack, z_dict)

        # R -> G, dynamic target size.
        g16, g8, g4 = self.stage_projection(
            r16=r16,
            r8=r8,
            r4=r4,
            target16=x3.shape[-2:],
            target8=x2.shape[-2:],
            target4=x1.shape[-2:],
        )

        # Simple residual injection.
        x3_hat = self.inject16(x3, g16)
        x2_hat = self.inject8(x2, g8)
        x1_hat = self.inject4(x1, g4)

        # HL_base decoder.
        d3 = self.dec3(x4, x3_hat)
        d2 = self.dec2(d3, x2_hat)
        d1 = self.dec1(d2, x1_hat)
        d0 = self.dec0(d1, x0)

        logits_half = self.out_head(d0)

        logits = F.interpolate(
            logits_half,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        dino_prior_logits = None

        if self.dino_prior_head is not None:
            dino_prior_logits = self.dino_prior_head(g4)

        aux = None

        if need_aux:
            aux = {
                "final_logits": logits,
                "logits_half": logits_half,
                "dino_prior_logits": dino_prior_logits,

                "x0": x0,
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "x4": x4,

                "x1_hat": x1_hat,
                "x2_hat": x2_hat,
                "x3_hat": x3_hat,

                "r16": r16,
                "r8": r8,
                "r4": r4,

                "g16": g16,
                "g8": g8,
                "g4": g4,

                "w16": w16,
                "w8": w8,
                "w4": w4,

                "d3": d3,
                "d2": d2,
                "d1": d1,
                "d0": d0,

                "gamma16": self.inject16.gamma.detach(),
                "gamma8": self.inject8.gamma.detach(),
                "gamma4": self.inject4.gamma.detach(),
            }

            for k, v in a_dict.items():
                aux[f"A{k}"] = v

            for k, v in z_dict.items():
                aux[f"Z{k}"] = v

        return logits, dino_prior_logits, aux

    def forward(self, x):
        logits, dino_prior_logits, aux = self.forward_features(
            x,
            need_aux=self.return_aux,
        )

        if self.return_aux:
            return aux

        return logits

    def forward_train(self, x):
        logits, dino_prior_logits, _ = self.forward_features(
            x,
            need_aux=False,
        )

        return {
            "final_logits": logits,
            "base_logits": None,
            "dino_prior_logits": dino_prior_logits,
        }


if __name__ == "__main__":
    model = DC_v4_1(
        n_channels=3,
        n_classes=1,
        pretrained=False,
        return_aux=False,
        dino_ckpt_path="",
    )

    x = torch.randn(1, 3, 256, 256)
    y = model(x)

    print("Input :", x.shape)
    if isinstance(y, dict):
        print("Output keys:", y.keys())
    else:
        print("Output:", y.shape)
