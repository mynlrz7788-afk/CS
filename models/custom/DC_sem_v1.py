import os
import sys
import importlib
from typing import Dict, Tuple, Sequence, List

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["DC_sem_v1"]


def _auto_padding(kernel_size, dilation=1):
    if isinstance(kernel_size, tuple):
        if isinstance(dilation, tuple):
            return tuple(((k - 1) // 2) * d for k, d in zip(kernel_size, dilation))
        return tuple(((k - 1) // 2) * dilation for k in kernel_size)

    return ((kernel_size - 1) // 2) * dilation


def _count_params(module: nn.Module, trainable_only: bool = False) -> int:
    if module is None:
        return 0

    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    return sum(p.numel() for p in module.parameters())


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
    保持 HL_base / DC_v2 风格。
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
    HL_base 同款解码块。

    x 上采样到 skip 尺度
    concat
    2 个 ConvBNAct
    ECA
    shortcut
    residual add
    ReLU
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


class LoRAQKVLinear(nn.Module):
    """
    给 DINO attention.qkv 加 LoRA。

    只改 q 和 v，不改 k。
    原始 qkv 参数冻结。
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 8.0,
        dropout: float = 0.05,
    ):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRAQKVLinear 只能包装 nn.Linear 类型的 qkv。")

        self.base = base_linear

        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        if self.out_features % 3 != 0:
            raise ValueError(
                f"qkv out_features 应该能被 3 整除，当前是 {self.out_features}"
            )

        self.embed_dim = self.out_features // 3

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_q_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_q_B = nn.Linear(self.rank, self.embed_dim, bias=False)

        self.lora_v_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_v_B = nn.Linear(self.rank, self.embed_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_q_B.weight)

        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_v_B.weight)

    def forward(self, x):
        base_out = self.base(x)

        q, k, v = base_out.chunk(3, dim=-1)

        z = self.dropout(x)

        q_delta = self.lora_q_B(self.lora_q_A(z)) * self.scaling
        v_delta = self.lora_v_B(self.lora_v_A(z)) * self.scaling

        q = q + q_delta
        v = v + v_delta

        return torch.cat([q, k, v], dim=-1)


class TokenResidualAdapter(nn.Module):
    """
    DINO block 后的 token adapter。

    形式：
        y = x + scale * Up(GELU(Down(LN(x))))

    注意：
        up projection 零初始化，所以初始状态等价于原始 DINO。
        scale 默认 1.0，不会让梯度死亡。
    """

    def __init__(
        self,
        dim=384,
        bottleneck=48,
        dropout=0.0,
        scale_init=1.0,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.down = nn.Linear(dim, bottleneck, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.up = nn.Linear(bottleneck, dim, bias=True)

        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = self.norm(x)
        z = self.down(z)
        z = self.act(z)
        z = self.dropout(z)
        z = self.up(z)

        return x + self.scale * z


class DinoBlockWithOutputAdapter(nn.Module):
    """
    包装 DINO 原始 block。

    不重写 DINO block 内部 attention / FFN。
    只在整个 block 输出后接一个 token residual adapter。

    这样兼容性更高，也能继续使用 get_intermediate_layers。
    """

    def __init__(
        self,
        block: nn.Module,
        dim=384,
        bottleneck=48,
        dropout=0.0,
    ):
        super().__init__()

        self.block = block

        for p in self.block.parameters():
            p.requires_grad = False

        self.adapter = TokenResidualAdapter(
            dim=dim,
            bottleneck=bottleneck,
            dropout=dropout,
            scale_init=1.0,
        )

    def _adapt_obj(self, obj):
        if torch.is_tensor(obj):
            if obj.dim() == 3:
                return self.adapter(obj)
            return obj

        if isinstance(obj, tuple):
            out = list(obj)

            for i, v in enumerate(out):
                if torch.is_tensor(v) and v.dim() == 3:
                    out[i] = self.adapter(v)
                    return tuple(out)

            return obj

        if isinstance(obj, list):
            out = list(obj)

            for i, v in enumerate(out):
                if torch.is_tensor(v) and v.dim() == 3:
                    out[i] = self.adapter(v)
                    return out

            return obj

        if isinstance(obj, dict):
            out = dict(obj)

            for key in ("x", "tokens", "x_prenorm"):
                if key in out and torch.is_tensor(out[key]) and out[key].dim() == 3:
                    out[key] = self.adapter(out[key])
                    return out

            for key, value in out.items():
                if torch.is_tensor(value) and value.dim() == 3:
                    out[key] = self.adapter(value)
                    return out

            return obj

        return obj

    def forward(self, *args, **kwargs):
        out = self.block(*args, **kwargs)
        out = self._adapt_obj(out)
        return out


def _get_dino_block_refs(backbone: nn.Module):
    """
    返回 DINO block 的引用，方便原地替换。

    返回：
        [(container, index, block), ...]
    """

    refs = []

    if hasattr(backbone, "blocks"):
        blocks = backbone.blocks

        for idx, blk in enumerate(blocks):
            refs.append((blocks, idx, blk))

        return refs

    if hasattr(backbone, "block_chunks"):
        flat_idx = 0

        for chunk in backbone.block_chunks:
            for local_idx, blk in enumerate(chunk):
                if isinstance(blk, nn.Identity):
                    continue

                refs.append((chunk, local_idx, blk))
                flat_idx += 1

        return refs

    return refs


def _install_block_output_adapters(
    backbone: nn.Module,
    block_indices: Sequence[int],
    dim=384,
    bottleneck=48,
    dropout=0.0,
) -> int:
    """
    给指定 DINO blocks 安装 block-output token adapter。
    """

    refs = _get_dino_block_refs(backbone)

    if len(refs) == 0:
        print("[DC_sem_v1] 警告：未找到 DINO blocks，无法安装 block adapter。")
        return 0

    installed = 0

    for idx in block_indices:
        idx = int(idx)

        if idx < 0 or idx >= len(refs):
            print(
                f"[DC_sem_v1] 警告：block adapter index={idx} 超出范围，"
                f"当前 DINO blocks 数量={len(refs)}"
            )
            continue

        container, local_idx, blk = refs[idx]

        if isinstance(blk, DinoBlockWithOutputAdapter):
            continue

        wrapped = DinoBlockWithOutputAdapter(
            block=blk,
            dim=dim,
            bottleneck=bottleneck,
            dropout=dropout,
        )

        container[local_idx] = wrapped
        installed += 1

    return installed


def _flatten_dino_blocks(backbone: nn.Module) -> nn.ModuleList:
    blocks = []

    refs = _get_dino_block_refs(backbone)

    for _, _, blk in refs:
        if isinstance(blk, DinoBlockWithOutputAdapter):
            blocks.append(blk.block)
        else:
            blocks.append(blk)

    return nn.ModuleList(blocks)


def _install_lora_to_dino_qkv(
    backbone: nn.Module,
    block_indices: Sequence[int] = (8, 9, 10, 11),
    rank: int = 4,
    alpha: float = 8.0,
    dropout: float = 0.05,
) -> int:
    """
    给 DINO 后几层 attention.qkv 安装 LoRA。

    注意：
        如果同时使用 block adapter，本函数应先于 block adapter 安装。
    """

    refs = _get_dino_block_refs(backbone)

    if len(refs) == 0:
        print("[DC_sem_v1] 警告：未找到 DINO blocks，无法安装 LoRA。")
        return 0

    installed = 0

    for idx in block_indices:
        idx = int(idx)

        if idx < 0 or idx >= len(refs):
            print(
                f"[DC_sem_v1] 警告：LoRA block index={idx} 超出范围，"
                f"当前 DINO blocks 数量={len(refs)}"
            )
            continue

        _, _, blk = refs[idx]

        if isinstance(blk, DinoBlockWithOutputAdapter):
            blk = blk.block

        if not hasattr(blk, "attn"):
            print(f"[DC_sem_v1] 警告：block {idx} 没有 attn，跳过 LoRA。")
            continue

        attn = blk.attn

        if not hasattr(attn, "qkv"):
            print(f"[DC_sem_v1] 警告：block {idx}.attn 没有 qkv，跳过 LoRA。")
            continue

        if isinstance(attn.qkv, LoRAQKVLinear):
            continue

        if not isinstance(attn.qkv, nn.Linear):
            print(
                f"[DC_sem_v1] 警告：block {idx}.attn.qkv 不是 nn.Linear，"
                f"实际类型={type(attn.qkv)}，跳过 LoRA。"
            )
            continue

        attn.qkv = LoRAQKVLinear(
            base_linear=attn.qkv,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        installed += 1

    return installed


class PEFTDINOv3FeatureExtractor(nn.Module):
    """
    DINOv3 特征提取器。

    默认：
        DINO 原始参数冻结。
        使用官方 get_intermediate_layers。
        输出 F2/F5/F8/F11 的二维 feature map。

    可选：
        use_block_adapter=True：
            后若干 DINO blocks 输出后接 token residual adapter。

        use_lora=True：
            后若干 DINO blocks 的 q/v 加 LoRA。
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

        use_lora=False,
        lora_layers=(8, 9, 10, 11),
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.05,

        use_block_adapter=True,
        block_adapter_layers=(6, 7, 8, 9, 10, 11),
        block_adapter_bottleneck=48,
        block_adapter_dropout=0.0,
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

        self.use_lora = bool(use_lora)
        self.lora_layers = tuple(int(x) for x in lora_layers)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)

        self.use_block_adapter = bool(use_block_adapter)
        self.block_adapter_layers = tuple(int(x) for x in block_adapter_layers)
        self.block_adapter_bottleneck = int(block_adapter_bottleneck)
        self.block_adapter_dropout = float(block_adapter_dropout)

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

        self.lora_installed = 0
        self.block_adapter_installed = 0

        if self.use_lora:
            self.lora_installed = _install_lora_to_dino_qkv(
                self.backbone,
                block_indices=self.lora_layers,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )

            print(
                "[DC_sem_v1] LoRA 安装完成："
                f"layers={self.lora_layers}, rank={self.lora_rank}, "
                f"installed={self.lora_installed}"
            )

        if self.use_block_adapter:
            self.block_adapter_installed = _install_block_output_adapters(
                self.backbone,
                block_indices=self.block_adapter_layers,
                dim=self.embed_dim,
                bottleneck=self.block_adapter_bottleneck,
                dropout=self.block_adapter_dropout,
            )

            print(
                "[DC_sem_v1] Block Token Adapter 安装完成："
                f"layers={self.block_adapter_layers}, "
                f"bottleneck={self.block_adapter_bottleneck}, "
                f"installed={self.block_adapter_installed}"
            )

        print(
            "[DC_sem_v1] DINOv3 feature extractor ready. "
            f"out_layers={self.out_layers}, "
            f"use_lora={self.use_lora}, "
            f"use_block_adapter={self.use_block_adapter}"
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
                f"[DC_sem_v1] 加载 DINOv3 权重完成: {self.dino_ckpt_path}\n"
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

        # DINO 主干保持 eval，PEFT 参数仍然有梯度。
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

    def _forward_impl(self, x):
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

    def forward(self, x):
        has_trainable_peft = (
            (self.use_lora and self.lora_installed > 0)
            or (self.use_block_adapter and self.block_adapter_installed > 0)
        )

        if has_trainable_peft:
            return self._forward_impl(x)

        with torch.no_grad():
            return self._forward_impl(x)


class DinoMapProjector(nn.Module):
    """
    每个 DINO 输出层的二维投影。

    输入：
        B, 384, H/16, W/16

    输出：
        B, adapter_dim, H/16, W/16
    """

    def __init__(self, in_channels=384, out_channels=96):
        super().__init__()

        self.proj = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

        self.dw = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size=3,
            groups=out_channels,
        )

        self.pw = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.eca = ECALayer(out_channels)
        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        last_bn = self.pw[-1]
        nn.init.zeros_(last_bn.weight)
        nn.init.zeros_(last_bn.bias)

    def forward(self, x):
        x = self.proj(x)

        y = self.dw(x)
        y = self.pw(y)
        y = self.eca(y)

        return self.act(x + y)


class ScaleProjector(nn.Module):
    """
    从低维 DINO 融合特征投影到目标尺度通道。

    用 depthwise conv + strip depthwise conv，控制参数量。
    """

    def __init__(self, in_channels=96, out_channels=128):
        super().__init__()

        self.base = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0,
        )

        self.dw3 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.dwh = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.dwv = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.local_fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.eca = ECALayer(out_channels)
        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        last_bn = self.local_fuse[-1]
        nn.init.zeros_(last_bn.weight)
        nn.init.zeros_(last_bn.bias)

    def forward(self, x):
        base = self.base(x)

        local = self.dw3(base) + self.dwh(base) + self.dwv(base)
        local = self.local_fuse(local)
        local = self.eca(local)

        out = base + local

        return self.act(out)


class WeightedLayerFusion(nn.Module):
    """
    每个目标尺度都从多层 DINO 特征中自适应选择。
    """

    def __init__(
        self,
        valid_layers: Sequence[int],
        adapter_dim=96,
        out_channels=128,
    ):
        super().__init__()

        self.valid_layers = [int(x) for x in valid_layers]
        self.raw_weights = nn.Parameter(torch.zeros(len(self.valid_layers)))

        self.projector = ScaleProjector(
            in_channels=adapter_dim,
            out_channels=out_channels,
        )

    def forward(
        self,
        low_maps: Dict[int, torch.Tensor],
        target_size: Tuple[int, int],
    ):
        feats = []

        for layer in self.valid_layers:
            if layer not in low_maps:
                raise KeyError(
                    f"WeightedLayerFusion 缺少 layer={layer}，"
                    f"当前 keys={list(low_maps.keys())}"
                )

            f = low_maps[layer]

            if f.shape[-2:] != target_size:
                f = F.interpolate(
                    f,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                )

            feats.append(f)

        feat_stack = torch.stack(feats, dim=1)

        weights = torch.softmax(self.raw_weights, dim=0)
        weights_view = weights.view(1, -1, 1, 1, 1)

        fused = (feat_stack * weights_view).sum(dim=1)
        out = self.projector(fused)

        return out, weights.detach()


class RGBMultiScaleDetailStem(nn.Module):
    """
    极轻量 RGB 多尺度细节补偿。

    输出：
        r0: H/2, 64
        r1: H/4, 64
        r2: H/8, 128

    它不是完整 CNN encoder，只是给 DINO pyramid 的浅层补空间细节。
    """

    def __init__(self, in_channels=3):
        super().__init__()

        self.stem0 = nn.Sequential(
            ConvBNAct(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
            ),
            ConvBNAct(
                32,
                32,
                kernel_size=3,
                groups=32,
            ),
            ConvBNAct(
                32,
                64,
                kernel_size=1,
                padding=0,
            ),
        )

        self.down1 = nn.Sequential(
            ConvBNAct(
                64,
                64,
                kernel_size=3,
                stride=2,
                groups=1,
            ),
            ConvBNAct(
                64,
                64,
                kernel_size=3,
                groups=64,
            ),
        )

        self.down2 = nn.Sequential(
            ConvBNAct(
                64,
                128,
                kernel_size=3,
                stride=2,
                groups=1,
            ),
            ConvBNAct(
                128,
                128,
                kernel_size=3,
                groups=128,
            ),
        )

    def forward(self, x):
        r0 = self.stem0(x)
        r1 = self.down1(r0)
        r2 = self.down2(r1)

        return r0, r1, r2


class SemanticSuppressionGate(nn.Module):
    """
    可选语义门控。

    默认不建议打开。
    当前阶段更重视恢复 Recall。
    """

    def __init__(
        self,
        in_channels=96,
        hidden_channels=64,
        min_factor=0.5,
    ):
        super().__init__()

        self.min_factor = float(min_factor)

        self.net = nn.Sequential(
            ConvBNAct(
                in_channels * 2,
                hidden_channels,
                kernel_size=1,
                padding=0,
            ),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=True),
        )

        self._init_weights()

    def _init_weights(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.constant_(last.bias, 2.0)

    def forward(self, high_a, high_b):
        if high_a.shape[-2:] != high_b.shape[-2:]:
            high_b = F.interpolate(
                high_b,
                size=high_a.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        g = torch.sigmoid(self.net(torch.cat([high_a, high_b], dim=1)))

        return g

    def apply_gate(self, x, gate):
        if gate.shape[-2:] != x.shape[-2:]:
            gate = F.interpolate(
                gate,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        factor = self.min_factor + (1.0 - self.min_factor) * gate

        return x * factor


class DinoRoadPyramidAdapter(nn.Module):
    """
    DINO token-to-pyramid 道路适配器。

    输入：
        F2/F5/F8/F11，均为 H/16 的 DINO map

    输出：
        P4: H/32, 512
        P3: H/16, 256
        P2: H/8,  128
        P1: H/4,  64
        P0: H/2,  64
    """

    def __init__(
        self,
        dino_layers=(2, 5, 8, 11),
        dino_embed_dim=384,
        adapter_dim=96,
        rgb_in_channels=3,
        use_rgb_detail=True,
        rgb_detail_multi_scale=True,
        use_semantic_gate=False,
    ):
        super().__init__()

        self.dino_layers = [int(x) for x in dino_layers]
        self.dino_embed_dim = int(dino_embed_dim)
        self.adapter_dim = int(adapter_dim)

        self.use_rgb_detail = bool(use_rgb_detail)
        self.rgb_detail_multi_scale = bool(rgb_detail_multi_scale)
        self.use_semantic_gate = bool(use_semantic_gate)

        if len(self.dino_layers) < 4:
            raise ValueError(
                "当前适配器默认需要 4 个 DINO 层，例如 (2, 5, 8, 11)。"
            )

        self.layer_projectors = nn.ModuleDict()

        for layer in self.dino_layers:
            self.layer_projectors[str(layer)] = DinoMapProjector(
                in_channels=dino_embed_dim,
                out_channels=adapter_dim,
            )

        l0, l1, l2, l3 = self.dino_layers

        self.p4_fusion = WeightedLayerFusion(
            valid_layers=(l1, l2, l3),
            adapter_dim=adapter_dim,
            out_channels=512,
        )

        self.p3_fusion = WeightedLayerFusion(
            valid_layers=(l0, l1, l2, l3),
            adapter_dim=adapter_dim,
            out_channels=256,
        )

        self.p2_fusion = WeightedLayerFusion(
            valid_layers=(l0, l1, l2, l3),
            adapter_dim=adapter_dim,
            out_channels=128,
        )

        self.p1_fusion = WeightedLayerFusion(
            valid_layers=(l0, l1, l2),
            adapter_dim=adapter_dim,
            out_channels=64,
        )

        self.p0_fusion = WeightedLayerFusion(
            valid_layers=(l0, l1),
            adapter_dim=adapter_dim,
            out_channels=64,
        )

        if self.use_rgb_detail:
            self.rgb_detail = RGBMultiScaleDetailStem(
                in_channels=rgb_in_channels,
            )

            self.rgb_detail_scale0 = nn.Parameter(torch.tensor(0.10))
            self.rgb_detail_scale1 = nn.Parameter(torch.tensor(0.10))
            self.rgb_detail_scale2 = nn.Parameter(torch.tensor(0.10))
        else:
            self.rgb_detail = None
            self.rgb_detail_scale0 = None
            self.rgb_detail_scale1 = None
            self.rgb_detail_scale2 = None

        if self.use_semantic_gate:
            self.semantic_gate = SemanticSuppressionGate(
                in_channels=adapter_dim,
                hidden_channels=max(adapter_dim // 2, 32),
                min_factor=0.5,
            )
        else:
            self.semantic_gate = None

    @staticmethod
    def _size_div(h, w, div):
        return (max(1, h // div), max(1, w // div))

    def forward(
        self,
        dino_feats: Dict[int, torch.Tensor],
        rgb: torch.Tensor,
    ):
        _, _, h, w = rgb.shape

        base_hw = self._size_div(h, w, 16)

        low_maps = {}

        for layer in self.dino_layers:
            if layer not in dino_feats:
                raise KeyError(
                    f"DINO 输出中找不到 layer={layer}。"
                    f"当前 keys={list(dino_feats.keys())}"
                )

            feat = dino_feats[layer]

            if feat.shape[-2:] != base_hw:
                feat = F.interpolate(
                    feat,
                    size=base_hw,
                    mode="bilinear",
                    align_corners=False,
                )

            low_maps[layer] = self.layer_projectors[str(layer)](feat)

        p4, w4 = self.p4_fusion(
            low_maps,
            target_size=self._size_div(h, w, 32),
        )

        p3, w3 = self.p3_fusion(
            low_maps,
            target_size=self._size_div(h, w, 16),
        )

        p2, w2 = self.p2_fusion(
            low_maps,
            target_size=self._size_div(h, w, 8),
        )

        p1, w1 = self.p1_fusion(
            low_maps,
            target_size=self._size_div(h, w, 4),
        )

        p0, w0 = self.p0_fusion(
            low_maps,
            target_size=self._size_div(h, w, 2),
        )

        r0 = None
        r1 = None
        r2 = None

        if self.rgb_detail is not None:
            r0, r1, r2 = self.rgb_detail(rgb)

            if r0.shape[-2:] != p0.shape[-2:]:
                r0 = F.interpolate(
                    r0,
                    size=p0.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            p0 = p0 + self.rgb_detail_scale0 * r0

            if self.rgb_detail_multi_scale:
                if r1.shape[-2:] != p1.shape[-2:]:
                    r1 = F.interpolate(
                        r1,
                        size=p1.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                if r2.shape[-2:] != p2.shape[-2:]:
                    r2 = F.interpolate(
                        r2,
                        size=p2.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                p1 = p1 + self.rgb_detail_scale1 * r1
                p2 = p2 + self.rgb_detail_scale2 * r2

        gate = None

        if self.semantic_gate is not None:
            high_a = low_maps[self.dino_layers[-2]]
            high_b = low_maps[self.dino_layers[-1]]

            gate = self.semantic_gate(high_a, high_b)

            p2 = self.semantic_gate.apply_gate(p2, gate)
            p1 = self.semantic_gate.apply_gate(p1, gate)
            p0 = self.semantic_gate.apply_gate(p0, gate)

        aux = {
            "pyramid_weight_p4": w4,
            "pyramid_weight_p3": w3,
            "pyramid_weight_p2": w2,
            "pyramid_weight_p1": w1,
            "pyramid_weight_p0": w0,
            "semantic_gate": gate,
            "rgb_detail_r0": r0,
            "rgb_detail_r1": r1,
            "rgb_detail_r2": r2,
        }

        for layer, feat in low_maps.items():
            aux[f"low_F{layer}"] = feat

        return p0, p1, p2, p3, p4, aux


class DC_sem_v1(nn.Module):
    """
    DC_sem_v1

    当前版本定位：
        DINOv3-S
        + block output token adapter
        + token-to-pyramid road adapter
        + multi-scale RGB detail stem
        + HL_base decoder

    不包含：
        CNN encoder
        CNN-DINO 相互指导
        cross attention
        新 decoder
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

        adapter_dim=96,
        use_rgb_detail=True,
        rgb_detail_multi_scale=True,
        use_semantic_gate=False,

        use_lora=False,
        lora_layers=(8, 9, 10, 11),
        lora_rank=4,
        lora_alpha=8.0,
        lora_dropout=0.05,

        use_block_adapter=True,
        block_adapter_layers=(6, 7, 8, 9, 10, 11),
        block_adapter_bottleneck=48,
        block_adapter_dropout=0.0,

        hl_decoder_ckpt_path="",
        max_total_params_m=30.0,

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
        self.dino_embed_dim = int(dino_embed_dim)
        self.dino_patch_size = int(dino_patch_size)
        self.adapter_dim = int(adapter_dim)
        self.max_total_params_m = float(max_total_params_m)

        if self.n_channels != 3:
            self.input_adapter = nn.Conv2d(
                self.n_channels,
                3,
                kernel_size=1,
                bias=False,
            )
        else:
            self.input_adapter = nn.Identity()

        self.dino = PEFTDINOv3FeatureExtractor(
            dino_model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            out_layers=dino_layers,
            embed_dim=dino_embed_dim,
            patch_size=dino_patch_size,
            dino_normalize=dino_normalize,
            dino_intermediate_norm=dino_intermediate_norm,

            use_lora=use_lora,
            lora_layers=lora_layers,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,

            use_block_adapter=use_block_adapter,
            block_adapter_layers=block_adapter_layers,
            block_adapter_bottleneck=block_adapter_bottleneck,
            block_adapter_dropout=block_adapter_dropout,
        )

        self.pyramid_adapter = DinoRoadPyramidAdapter(
            dino_layers=dino_layers,
            dino_embed_dim=dino_embed_dim,
            adapter_dim=adapter_dim,
            rgb_in_channels=3,
            use_rgb_detail=use_rgb_detail,
            rgb_detail_multi_scale=rgb_detail_multi_scale,
            use_semantic_gate=use_semantic_gate,
        )

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

        if hl_decoder_ckpt_path is not None and hl_decoder_ckpt_path != "":
            self._load_hl_decoder(hl_decoder_ckpt_path)

        self._print_trainable_summary()

    def _load_hl_decoder(self, ckpt_path: str):
        if not os.path.isfile(ckpt_path):
            print(f"[DC_sem_v1] 警告：找不到 hl_decoder_ckpt_path: {ckpt_path}")
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")

        if isinstance(ckpt, dict):
            if "model" in ckpt:
                state_dict = ckpt["model"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        clean_state = {}

        for k, v in state_dict.items():
            nk = k

            for prefix in (
                "module.",
                "model.",
            ):
                if nk.startswith(prefix):
                    nk = nk[len(prefix):]

            clean_state[nk] = v

        allowed_prefixes = (
            "dec3.",
            "dec2.",
            "dec1.",
            "dec0.",
            "out_head.",
        )

        current_state = self.state_dict()
        load_state = {}

        for k, v in clean_state.items():
            if not any(k.startswith(prefix) for prefix in allowed_prefixes):
                continue

            if k not in current_state:
                continue

            if current_state[k].shape != v.shape:
                continue

            load_state[k] = v

        if len(load_state) == 0:
            print(
                "[DC_sem_v1] 警告：没有从 HL_base 权重中匹配到 decoder / out_head 参数。"
            )
            return

        self.load_state_dict(load_state, strict=False)

        print(
            f"[DC_sem_v1] 已加载 HL_base decoder 权重: {ckpt_path}\n"
            f"            matched_keys={len(load_state)}"
        )

    def _print_trainable_summary(self):
        total = _count_params(self, trainable_only=False)
        trainable = _count_params(self, trainable_only=True)

        dino_total = _count_params(self.dino, trainable_only=False)
        dino_trainable = _count_params(self.dino, trainable_only=True)

        adapter_total = _count_params(self.pyramid_adapter, trainable_only=False)
        adapter_trainable = _count_params(self.pyramid_adapter, trainable_only=True)

        decoder_modules = nn.ModuleList([
            self.dec3,
            self.dec2,
            self.dec1,
            self.dec0,
            self.out_head,
        ])

        decoder_total = _count_params(decoder_modules, trainable_only=False)
        decoder_trainable = _count_params(decoder_modules, trainable_only=True)

        print("--------------------------------------------------")
        print("🔧 DC_sem_v1: DINO Block Adapter + Pyramid Adapter + HL_base Decoder")
        print(f"    - 总参数量:           {total / 1e6:.2f} M")
        print(f"    - 可训练参数量:       {trainable / 1e6:.2f} M")
        print(f"    - DINO 总参数量:      {dino_total / 1e6:.2f} M")
        print(f"    - DINO 可训练参数:    {dino_trainable / 1e6:.4f} M")
        print(f"    - Adapter 参数量:     {adapter_total / 1e6:.2f} M")
        print(f"    - Adapter 可训练:     {adapter_trainable / 1e6:.2f} M")
        print(f"    - Decoder 参数量:     {decoder_total / 1e6:.2f} M")
        print(f"    - Decoder 可训练:     {decoder_trainable / 1e6:.2f} M")

        if total / 1e6 > self.max_total_params_m:
            print(
                f"⚠️  警告：当前总参数量 {total / 1e6:.2f} M "
                f"超过限制 {self.max_total_params_m:.2f} M。"
            )
            print(
                "    建议优先调小 adapter_dim，例如 96 -> 80，"
                "或关闭 LoRA / 减少 block adapter 层数。"
            )

        print("    - 说明：无 CNN encoder，无相互指导，decoder 使用 HL_base 风格")
        print("--------------------------------------------------")

    def set_return_aux(self, flag: bool):
        self.return_aux = bool(flag)

    def train(self, mode=True):
        super().train(mode)

        # 保持 DINO backbone eval。
        # PEFT adapter、LoRA、pyramid adapter、decoder 仍然有梯度。
        self.dino.train(False)

        return self

    def forward_features(self, x):
        input_size = x.shape[-2:]

        x_in = self.input_adapter(x)

        dino_feats = self.dino(x_in)

        p0, p1, p2, p3, p4, adapter_aux = self.pyramid_adapter(
            dino_feats=dino_feats,
            rgb=x_in,
        )

        d3 = self.dec3(p4, p3)
        d2 = self.dec2(d3, p2)
        d1 = self.dec1(d2, p1)
        d0 = self.dec0(d1, p0)

        logits_half = self.out_head(d0)

        logits = F.interpolate(
            logits_half,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        aux = {
            "logits_half": logits_half,
            "p0": p0,
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "p4": p4,
            "d3": d3,
            "d2": d2,
            "d1": d1,
            "d0": d0,
        }

        for k, v in adapter_aux.items():
            aux[k] = v

        return logits, aux

    def forward(self, x):
        logits, aux = self.forward_features(x)

        if self.return_aux:
            aux["final_logits"] = logits
            return aux

        return logits


if __name__ == "__main__":
    model = DC_sem_v1(
        n_channels=3,
        n_classes=1,
        pretrained=False,
        return_aux=False,
        dino_ckpt_path="",
        use_block_adapter=True,
        block_adapter_layers=(6, 7, 8, 9, 10, 11),
        block_adapter_bottleneck=48,
        use_lora=False,
        adapter_dim=96,
        use_semantic_gate=False,
        use_rgb_detail=True,
        rgb_detail_multi_scale=True,
    )

    x = torch.randn(1, 3, 512, 512)

    with torch.no_grad():
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