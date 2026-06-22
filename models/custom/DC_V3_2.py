import os
import sys
import importlib
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DC_v3_2"]


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

        f = torch.cat([x, skip], dim=1)

        out = self.fuse(f)
        out = self.eca(out)
        out = out + self.shortcut(f)

        return self.act(out)


class RoadTokenAdapter(nn.Module):
    """
    插入 DINO block 后面的道路 Adapter。

    初始状态接近恒等映射：
        up.weight = 0
        up.bias = 0

    所以训练一开始不会破坏 DINO 原始特征。
    """

    def __init__(
        self,
        embed_dim=384,
        bottleneck=64,
        init_scale=0.1,
        dropout=0.0,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.down = nn.Linear(embed_dim, bottleneck)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, embed_dim)

        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.down(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.up(x)
        return residual + self.scale * x


class LoRAQKVLinear(nn.Module):
    """
    给 DINO Attention 的 qkv Linear 加 Q/V LoRA。

    假设原始 qkv:
        in_features = C
        out_features = 3C

    forward:
        base_qkv = frozen_qkv(x)
        q += LoRA_q(x)
        v += LoRA_v(x)
        k 不变

    这样只调整 Q 和 V，不改 K。
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank=4,
        alpha=8.0,
        dropout=0.0,
    ):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRAQKVLinear 只支持 nn.Linear")

        self.base = base_linear

        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        if self.out_features % 3 != 0:
            raise ValueError(
                f"qkv out_features 应该能被 3 整除，但得到 {self.out_features}"
            )

        self.embed_dim = self.out_features // 3
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)

        self.dropout = nn.Dropout(dropout)

        self.q_down = nn.Linear(self.in_features, self.rank, bias=False)
        self.q_up = nn.Linear(self.rank, self.embed_dim, bias=False)

        self.v_down = nn.Linear(self.in_features, self.rank, bias=False)
        self.v_up = nn.Linear(self.rank, self.embed_dim, bias=False)

        nn.init.kaiming_uniform_(self.q_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.q_up.weight)

        nn.init.kaiming_uniform_(self.v_down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.v_up.weight)

    def forward(self, x):
        base_out = self.base(x)

        q_delta = self.q_up(self.q_down(self.dropout(x))) * self.scaling
        v_delta = self.v_up(self.v_down(self.dropout(x))) * self.scaling

        q, k, v = base_out.split(self.embed_dim, dim=-1)

        q = q + q_delta
        v = v + v_delta

        return torch.cat([q, k, v], dim=-1)


class LoRALinear(nn.Module):
    """
    兼容 separate q_proj / v_proj 的 LoRA 包装。
    如果你的 DINO attention 不是 qkv，而是 q_proj/v_proj，就会用到它。
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank=4,
        alpha=8.0,
        dropout=0.0,
    ):
        super().__init__()

        self.base = base_linear

        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)

        self.dropout = nn.Dropout(dropout)

        self.down = nn.Linear(self.in_features, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.out_features, bias=False)

        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.base(x) + self.up(self.down(self.dropout(x))) * self.scaling


class DINOv3RoadLoRAAdapterFeatureExtractor(nn.Module):
    """
    DINOv3 + Road Adapter + Q/V LoRA。

    训练：
        blocks 6-11 后面的 Road Adapter
        blocks 8-11 的 Q/V LoRA

    冻结：
        DINO 原始权重全部冻结
    """

    def __init__(
        self,
        dino_model_name="dinov3_vits16",
        dino_repo_path="/home/u2508183004/zyn/SEG/dinounet/dinov3",
        dino_ckpt_path="/home/u2508183004/zyn/SEG/weight/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        out_layers=(2, 5, 8, 11),
        adapt_layers=(6, 7, 8, 9, 10, 11),
        lora_layers=(8, 9, 10, 11),
        embed_dim=384,
        patch_size=16,
        dino_normalize=False,
        road_adapter_bottleneck=64,
        road_adapter_init_scale=0.1,
        road_adapter_dropout=0.0,
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,
    ):
        super().__init__()

        self.dino_model_name = dino_model_name
        self.dino_repo_path = dino_repo_path
        self.dino_ckpt_path = dino_ckpt_path

        self.out_layers = list(out_layers)
        self.adapt_layers = list(adapt_layers)
        self.lora_layers = list(lora_layers)

        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.dino_normalize = bool(dino_normalize)

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
        self.blocks = self._get_blocks()

        if len(self.blocks) == 0:
            raise RuntimeError(
                "没有在 DINOv3 backbone 中找到 blocks，请检查 dinounet/dinov3 的模型结构。"
            )

        if max(self.out_layers) >= len(self.blocks):
            raise RuntimeError(
                f"dino_layers={self.out_layers} 超出 blocks 数量。"
                f"当前 blocks 数量={len(self.blocks)}"
            )

        if len(self.adapt_layers) > 0 and max(self.adapt_layers) >= len(self.blocks):
            raise RuntimeError(
                f"adapt_layers={self.adapt_layers} 超出 blocks 数量。"
                f"当前 blocks 数量={len(self.blocks)}"
            )

        if len(self.lora_layers) > 0 and max(self.lora_layers) >= len(self.blocks):
            raise RuntimeError(
                f"lora_layers={self.lora_layers} 超出 blocks 数量。"
                f"当前 blocks 数量={len(self.blocks)}"
            )

        self._freeze_dino_original_params()

        self.road_adapters = nn.ModuleDict()

        for layer_idx in self.adapt_layers:
            self.road_adapters[str(layer_idx)] = RoadTokenAdapter(
                embed_dim=embed_dim,
                bottleneck=road_adapter_bottleneck,
                init_scale=road_adapter_init_scale,
                dropout=road_adapter_dropout,
            )

        installed = self._install_lora_to_blocks(
            lora_layers=self.lora_layers,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )

        print(
            "[DC_v3_2] DINO 原始参数冻结，Road Adapter + Q/V LoRA 可训练。 "
            f"adapt_layers={self.adapt_layers}, lora_layers={self.lora_layers}, "
            f"lora_installed={installed}"
        )

        if installed == 0 and len(self.lora_layers) > 0:
            print(
                "[DC_v3_2][警告] 没有成功安装 LoRA。"
                "请检查 DINO attention 里是否存在 qkv 或 q_proj/v_proj。"
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
                f"[DC_v3_2] 加载 DINOv3 权重完成: {self.dino_ckpt_path}\n"
                f"        missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
            )

            if len(missing) > 0:
                print(f"        missing 示例: {missing[:10]}")
            if len(unexpected) > 0:
                print(f"        unexpected 示例: {unexpected[:10]}")

        return model

    def _get_blocks(self):
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks

        if hasattr(self.backbone, "block_chunks"):
            blocks = []

            for chunk in self.backbone.block_chunks:
                for blk in chunk:
                    if not isinstance(blk, nn.Identity):
                        blocks.append(blk)

            return nn.ModuleList(blocks)

        return nn.ModuleList([])

    def _freeze_dino_original_params(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()

    def _install_lora_to_blocks(self, lora_layers, rank, alpha, dropout):
        installed = 0

        for layer_idx in lora_layers:
            block = self.blocks[layer_idx]

            attn = None

            if hasattr(block, "attn"):
                attn = block.attn
            elif hasattr(block, "attention"):
                attn = block.attention

            if attn is None:
                print(f"[DC_v3_2][警告] block {layer_idx} 找不到 attn/attention，跳过 LoRA")
                continue

            # 情况 1：DINO 常见 qkv Linear。
            if hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear):
                attn.qkv = LoRAQKVLinear(
                    attn.qkv,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                installed += 1
                continue

            # 情况 2：兼容 q_proj / v_proj。
            q_names = ["q_proj", "q", "query"]
            v_names = ["v_proj", "v", "value"]

            q_wrapped = False
            v_wrapped = False

            for q_name in q_names:
                if hasattr(attn, q_name) and isinstance(getattr(attn, q_name), nn.Linear):
                    setattr(
                        attn,
                        q_name,
                        LoRALinear(
                            getattr(attn, q_name),
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout,
                        ),
                    )
                    q_wrapped = True
                    break

            for v_name in v_names:
                if hasattr(attn, v_name) and isinstance(getattr(attn, v_name), nn.Linear):
                    setattr(
                        attn,
                        v_name,
                        LoRALinear(
                            getattr(attn, v_name),
                            rank=rank,
                            alpha=alpha,
                            dropout=dropout,
                        ),
                    )
                    v_wrapped = True
                    break

            if q_wrapped or v_wrapped:
                installed += 1
            else:
                print(
                    f"[DC_v3_2][警告] block {layer_idx} 没有找到 qkv 或 q_proj/v_proj，跳过 LoRA"
                )

        return installed

    def train(self, mode=True):
        super().train(mode)

        # DINO 原始 backbone 保持 eval。
        # 注意：LoRA 在 backbone 内部，若 dropout=0，不受影响。
        # Road Adapter 在 self.road_adapters 里，仍会保持 train。
        self.backbone.eval()

        return self

    def _unwrap_tokens(self, obj):
        if torch.is_tensor(obj):
            return obj

        if isinstance(obj, dict):
            for key in ("x", "tokens", "x_prenorm"):
                if key in obj and torch.is_tensor(obj[key]):
                    return obj[key]

            for v in obj.values():
                if torch.is_tensor(v) and v.ndim == 3:
                    return v

            raise RuntimeError(
                f"DINO 返回 dict，但找不到 token Tensor，keys={list(obj.keys())}"
            )

        if isinstance(obj, (tuple, list)):
            for v in obj:
                if torch.is_tensor(v) and v.ndim == 3:
                    return v

            if len(obj) > 0:
                return self._unwrap_tokens(obj[0])

            raise RuntimeError("DINO 返回空 tuple/list，无法取得 token。")

        raise RuntimeError(f"无法识别的 DINO token 类型: {type(obj)}")

    def _prepare_tokens(self, x):
        if hasattr(self.backbone, "prepare_tokens_with_masks"):
            try:
                tokens = self.backbone.prepare_tokens_with_masks(x, None)
            except TypeError:
                tokens = self.backbone.prepare_tokens_with_masks(x)

            return self._unwrap_tokens(tokens)

        if hasattr(self.backbone, "prepare_tokens"):
            tokens = self.backbone.prepare_tokens(x)
            return self._unwrap_tokens(tokens)

        if not hasattr(self.backbone, "patch_embed"):
            raise RuntimeError(
                "当前 DINOv3 backbone 没有 patch_embed / prepare_tokens_with_masks，"
                "无法手写 forward。"
            )

        tokens = self.backbone.patch_embed(x)

        if tokens.ndim == 4:
            tokens = tokens.flatten(2).transpose(1, 2)

        b = tokens.shape[0]

        if hasattr(self.backbone, "cls_token"):
            cls = self.backbone.cls_token.expand(b, -1, -1)

            storage = None

            if hasattr(self.backbone, "register_tokens") and self.backbone.register_tokens is not None:
                storage = self.backbone.register_tokens
            elif hasattr(self.backbone, "storage_tokens") and self.backbone.storage_tokens is not None:
                storage = self.backbone.storage_tokens

            if storage is not None:
                storage = storage.expand(b, -1, -1)
                tokens = torch.cat([cls, storage, tokens], dim=1)
            else:
                tokens = torch.cat([cls, tokens], dim=1)

        if hasattr(self.backbone, "interpolate_pos_encoding"):
            try:
                pos = self.backbone.interpolate_pos_encoding(tokens, x.shape[-2], x.shape[-1])
            except TypeError:
                pos = self.backbone.interpolate_pos_encoding(tokens, x)

            tokens = tokens + pos

        elif hasattr(self.backbone, "pos_embed"):
            pos = self.backbone.pos_embed

            if pos.shape[1] == tokens.shape[1]:
                tokens = tokens + pos

        return self._unwrap_tokens(tokens)

    def _run_block(self, block, tokens):
        tokens = self._unwrap_tokens(tokens)

        try:
            out = block(tokens)
        except TypeError:
            out = block(tokens, None)

        return self._unwrap_tokens(out)

    @staticmethod
    def _tokens_to_map(tokens, patch_h, patch_w):
        patch_n = patch_h * patch_w
        special_n = tokens.shape[1] - patch_n

        if special_n < 0:
            raise RuntimeError(
                f"token 数 {tokens.shape[1]} 小于 patch 数 {patch_n}，无法 reshape。"
            )

        patch_tokens = tokens[:, special_n:, :]

        fmap = patch_tokens.transpose(1, 2).contiguous()
        fmap = fmap.view(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)

        return fmap

    def forward(self, x):
        """
        注意：
            这里不能使用 @torch.no_grad()
            因为 Road Adapter 和 LoRA 需要梯度。
        """

        if self.dino_normalize:
            x = (x - self.mean) / self.std

        _, _, h, w = x.shape

        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise RuntimeError(
                f"DINO 输入尺寸必须能被 patch_size 整除。"
                f"当前输入: {h}x{w}, patch_size={self.patch_size}"
            )

        patch_h = h // self.patch_size
        patch_w = w // self.patch_size

        tokens = self._prepare_tokens(x)

        outputs = {}

        for i, block in enumerate(self.blocks):
            tokens = self._run_block(block, tokens)

            if i in self.adapt_layers:
                tokens = self.road_adapters[str(i)](tokens)

            if i in self.out_layers:
                outputs[i] = self._tokens_to_map(tokens, patch_h, patch_w)

        if len(outputs) != len(self.out_layers):
            raise RuntimeError(
                f"DINO 输出层不完整。期望 {self.out_layers}，实际得到 {list(outputs.keys())}。"
            )

        return outputs


class DinoOutAdapter(nn.Module):
    def __init__(self, in_ch=384, out_ch=128, bottleneck=64):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.local = nn.Sequential(
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                groups=out_ch,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(out_ch, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.proj_in(x)
        y = self.local(x)
        z = self.mlp(x + y)
        return self.act(x + y + z)


class StageProjectBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.proj = nn.Sequential(
            ConvBNAct(in_ch, out_ch, kernel_size=3),
            ConvBNAct(out_ch, out_ch, kernel_size=3),
        )

    def forward(self, x, size):
        if x.shape[-2:] != size:
            x = F.interpolate(
                x,
                size=size,
                mode="bilinear",
                align_corners=False,
            )

        return self.proj(x)


class ProgressivePyramid(nn.Module):
    """
    f16 -> P3
    P3 下采样 -> P4
    P3 上采样 -> P2
    P2 上采样 -> P1
    P1 上采样 -> P0
    """

    def __init__(self, in_ch=256):
        super().__init__()

        self.p3_proj = StageProjectBlock(in_ch, 256)
        self.p4_down = StageProjectBlock(256, 512)
        self.p2_up = StageProjectBlock(256, 128)
        self.p1_up = StageProjectBlock(128, 64)
        self.p0_up = StageProjectBlock(64, 64)

    def forward(self, f16, h, w):
        size_p3 = (max(1, h // 16), max(1, w // 16))
        size_p4 = (max(1, h // 32), max(1, w // 32))
        size_p2 = (max(1, h // 8), max(1, w // 8))
        size_p1 = (max(1, h // 4), max(1, w // 4))
        size_p0 = (max(1, h // 2), max(1, w // 2))

        p3 = self.p3_proj(f16, size=size_p3)
        p4 = self.p4_down(p3, size=size_p4)

        p2 = self.p2_up(p3, size=size_p2)
        p1 = self.p1_up(p2, size=size_p1)
        p0 = self.p0_up(p1, size=size_p0)

        return p4, p3, p2, p1, p0


class DC_v3_2(nn.Module):
    """
    DC_v3_2:
        Road Adapter + Q/V LoRA + Progressive Pyramid + HL-style Decoder

    目的：
        比 DC_v3_1 多训练 blocks 8-11 的 Q/V LoRA。
        让 DINO 中高层更适应道路分割的上下文和连通性。
    """

    def __init__(
        self,
        num_classes=1,
        n_channels=3,
        pretrained=True,
        return_aux=False,

        dino_model_name="dinov3_vits16",
        dino_repo_path="/home/u2508183004/zyn/SEG/dinounet/dinov3",
        dino_ckpt_path="/home/u2508183004/zyn/SEG/weight/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        dino_layers=(2, 5, 8, 11),
        dino_embed_dim=384,
        dino_patch_size=16,
        dino_normalize=False,

        adapt_layers=(6, 7, 8, 9, 10, 11),
        road_adapter_bottleneck=64,
        road_adapter_init_scale=0.1,
        road_adapter_dropout=0.0,

        lora_layers=(8, 9, 10, 11),
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.0,

        adapter_channels=128,
        adapter_bottleneck=64,
        fuse_channels=256,

        train_dino_branch_adapters=True,

        **kwargs,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.n_channels = int(n_channels)
        self.return_aux = bool(return_aux)

        self.dino_layers = list(dino_layers)
        self.dino_embed_dim = int(dino_embed_dim)
        self.dino_patch_size = int(dino_patch_size)
        self.num_dino_levels = len(self.dino_layers)

        self.dino = DINOv3RoadLoRAAdapterFeatureExtractor(
            dino_model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            out_layers=dino_layers,
            adapt_layers=adapt_layers,
            lora_layers=lora_layers,
            embed_dim=dino_embed_dim,
            patch_size=dino_patch_size,
            dino_normalize=dino_normalize,
            road_adapter_bottleneck=road_adapter_bottleneck,
            road_adapter_init_scale=road_adapter_init_scale,
            road_adapter_dropout=road_adapter_dropout,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        self.dino_adapters = nn.ModuleList([
            DinoOutAdapter(
                in_ch=dino_embed_dim,
                out_ch=adapter_channels,
                bottleneck=adapter_bottleneck,
            )
            for _ in range(self.num_dino_levels)
        ])

        self.fuse = nn.Sequential(
            ConvBNAct(
                adapter_channels * self.num_dino_levels,
                fuse_channels,
                kernel_size=1,
                padding=0,
            ),
            ConvBNAct(fuse_channels, fuse_channels, kernel_size=3),
        )

        self.pyramid = ProgressivePyramid(in_ch=fuse_channels)

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
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

        self._print_trainable_summary()

    def _print_trainable_summary(self):
        total = 0
        trainable = 0
        dino_trainable = 0
        road_adapter_trainable = 0
        lora_trainable = 0

        for p in self.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()

        for p in self.dino.parameters():
            if p.requires_grad:
                dino_trainable += p.numel()

        for name, p in self.dino.named_parameters():
            if "road_adapters" in name and p.requires_grad:
                road_adapter_trainable += p.numel()
            if (
                "q_down" in name
                or "q_up" in name
                or "v_down" in name
                or "v_up" in name
                or ".down." in name
                or ".up." in name
            ) and "road_adapters" not in name and p.requires_grad:
                lora_trainable += p.numel()

        print("--------------------------------------------------")
        print("🔧 DC_v3_2 DINO Road Adapter + Q/V LoRA + HL-style Decoder")
        print(f"    - 总参数量:             {total / 1e6:.2f} M")
        print(f"    - 可训练参数量:         {trainable / 1e6:.2f} M")
        print(f"    - DINO分支可训练:       {dino_trainable / 1e6:.2f} M")
        print(f"    - Road Adapter可训练:   {road_adapter_trainable / 1e6:.2f} M")
        print(f"    - Q/V LoRA可训练:       {lora_trainable / 1e6:.2f} M")
        print("    - 说明: DINO 原始参数冻结，只训练 Road Adapter、Q/V LoRA、外置 Adapter 和 Decoder")
        print("--------------------------------------------------")

    def set_return_aux(self, flag: bool):
        self.return_aux = bool(flag)

    def forward(self, x):
        input_hw = x.shape[-2:]
        _, _, h, w = x.shape

        dino_out = self.dino(x)

        dino_feats = []

        for layer in self.dino_layers:
            if layer not in dino_out:
                raise RuntimeError(
                    f"DINO 输出中找不到 layer={layer}。"
                    f"当前 keys={list(dino_out.keys())}"
                )
            dino_feats.append(dino_out[layer])

        base_hw = (
            max(1, h // self.dino_patch_size),
            max(1, w // self.dino_patch_size),
        )

        adapted = []

        for feat, adapter in zip(dino_feats, self.dino_adapters):
            if feat.shape[-2:] != base_hw:
                feat = F.interpolate(
                    feat,
                    size=base_hw,
                    mode="bilinear",
                    align_corners=False,
                )

            feat = adapter(feat)
            adapted.append(feat)

        fused = torch.cat(adapted, dim=1)
        f16 = self.fuse(fused)

        p4, p3, p2, p1, p0 = self.pyramid(f16, h=h, w=w)

        d3 = self.dec3(p4, p3)
        d2 = self.dec2(d3, p2)
        d1 = self.dec1(d2, p1)
        d0 = self.dec0(d1, p0)

        logits_half = self.out_head(d0)
        logits = F.interpolate(
            logits_half,
            size=input_hw,
            mode="bilinear",
            align_corners=False,
        )

        if self.return_aux:
            return {
                "final_logits": logits,
                "logits": logits,
                "logits_half": logits_half,
                "dino_feats": dino_feats,
                "adapted_feats": adapted,
                "f16": f16,
                "p4": p4,
                "p3": p3,
                "p2": p2,
                "p1": p1,
                "p0": p0,
                "d3": d3,
                "d2": d2,
                "d1": d1,
                "d0": d0,
            }

        return logits

    def forward_train(self, x):
        logits = self.forward(x)

        if isinstance(logits, dict):
            logits = logits["final_logits"]

        return {
            "final_logits": logits,
            "base_logits": None,
            "dino_prior_logits": None,
        }