"""
DC_v2_step1.py

第一步目标：
1. 保留 HL_base 结构分支，输出 C1/C2/C3/C4。
2. 按当前本地 DINOv3 目录结构加载 DINOv3-S。
3. 在 DINOv3 的每个 block 后加入可训练 TokenAdapter。
4. 不再使用 get_intermediate_layers，而是手写串联 DINO forward，取出 D2/D5/D8/D11。
5. 做 DINO 四层注意力融合和 DINO prior 辅助头。
6. 暂时不接 SB-RTGFI，主输出先用 HL_base logits，用来验证：
   - HL_base 分支尺寸是否正确；
   - DINO + Adapter 手写 forward 是否正确；
   - dino_prior_logits 是否能参与 train_DC.py 的辅助监督。

建议放置路径：SEG/models/custom/DC_v2_step1.py
主模块名：DC_v2_step1
"""

import os
import sys
import importlib
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["DC_v2_step1"]


def _auto_padding(kernel_size, dilation=1):
    if isinstance(kernel_size, tuple):
        if isinstance(dilation, tuple):
            return tuple(((k - 1) // 2) * d for k, d in zip(kernel_size, dilation))
        return tuple(((k - 1) // 2) * dilation for k in kernel_size)
    return ((kernel_size - 1) // 2) * dilation


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: int = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = None,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        act_layer=nn.ReLU,
        inplace: bool = True,
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


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, dilation=1):
        super().__init__()
        padding = _auto_padding(kernel_size, dilation)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TokenAdapter(nn.Module):
    """DINOv3-PEFT 风格 token adapter。

    输入输出都是 B × N × dim。
    最后一层初始化为 0，使模型初始时近似等于原始 DINO。
    """
    def __init__(self, dim: int = 384, bottleneck: int = 64, init_scale: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.act(self.down(self.norm(x))))
        return x + self.scale * residual


class DINOFourLayerFusion(nn.Module):
    """四层 DINO 特征注意力融合。"""
    def __init__(self, in_dim: int = 384, out_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.proj = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(in_dim, out_dim, kernel_size=1),
                DepthwiseSeparableConv(out_dim, out_dim, kernel_size=3),
            )
            for _ in range(num_layers)
        ])
        self.weight_gen = nn.Sequential(
            ConvBNAct(out_dim * num_layers, out_dim, kernel_size=3),
            nn.Conv2d(out_dim, num_layers, kernel_size=1),
        )
        # 初始偏向深层，减少浅层纹理噪声。
        bias = torch.tensor([-1.0, -0.5, 0.3, 0.5], dtype=torch.float32)
        if num_layers != 4:
            bias = torch.zeros(num_layers, dtype=torch.float32)
        self.layer_bias = nn.Parameter(bias.view(1, num_layers, 1, 1))

    def forward(self, feats: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        assert len(feats) == self.num_layers, f"需要 {self.num_layers} 层 DINO 特征，实际 {len(feats)}"
        ps = [proj(f) for proj, f in zip(self.proj, feats)]
        cat = torch.cat(ps, dim=1)
        weights = self.weight_gen(cat) + self.layer_bias
        weights = torch.softmax(weights, dim=1)
        fused = 0.0
        for i, p in enumerate(ps):
            fused = fused + weights[:, i:i + 1] * p
        return fused, weights


class DinoPriorHead(nn.Module):
    """低分辨率 DINO 道路先验头，输出 64×64 logits。"""
    def __init__(self, in_channels: int = 64, num_classes: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            DepthwiseSeparableConv(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class DINOv3AdapterBranch(nn.Module):
    """带 Adapter 的 DINOv3 串联前向分支。

    本步骤只取 D2/D5/D8/D11，不做 RTGFI token 回写。
    下一步会把 SB-RTGFI 插入到 i==2/5/8/11 的位置。
    """
    def __init__(
        self,
        dino_model_name: str = "dinov3_vits16",
        dino_repo_path: Optional[str] = None,
        dino_ckpt_path: Optional[str] = None,
        out_layers: Sequence[int] = (2, 5, 8, 11),
        embed_dim: int = 384,
        patch_size: int = 16,
        adapter_bottleneck: int = 64,
        adapter_init_scale: float = 0.1,
        dino_normalize: bool = False,
        apply_norm_to_intermediate: bool = True,
    ):
        super().__init__()
        self.dino_model_name = dino_model_name
        self.dino_repo_path = dino_repo_path
        self.dino_ckpt_path = dino_ckpt_path
        self.out_layers = list(out_layers)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.dino_normalize = dino_normalize
        self.apply_norm_to_intermediate = apply_norm_to_intermediate

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.backbone = self._build_dino_backbone()
        self._freeze_dino_backbone()
        self.blocks = self._get_blocks()
        if len(self.blocks) == 0:
            raise RuntimeError("没有在 DINOv3 backbone 中找到 blocks，请检查本地 DINOv3 结构。")
        self.adapters = nn.ModuleList([
            TokenAdapter(dim=embed_dim, bottleneck=adapter_bottleneck, init_scale=adapter_init_scale)
            for _ in range(len(self.blocks))
        ])

    def _build_dino_backbone(self):
        if self.dino_repo_path is None:
            raise ValueError("需要提供 dino_repo_path，例如 /home/u2508183004/zyn/SEG/dinounet/dinov3")
        if not os.path.isdir(self.dino_repo_path):
            raise FileNotFoundError(f"找不到 dino_repo_path: {self.dino_repo_path}")

        # 适配当前目录：SEG/dinounet/dinov3/hub/backbones.py
        dinov3_parent = os.path.dirname(self.dino_repo_path)
        if dinov3_parent not in sys.path:
            sys.path.insert(0, dinov3_parent)
        try:
            backbones = importlib.import_module("dinov3.hub.backbones")
        except Exception as e:
            raise ImportError(
                "导入 dinov3.hub.backbones 失败。请确认 dino_repo_path 指向 .../dinounet/dinov3。\n"
                f"原始错误：{repr(e)}"
            )
        if not hasattr(backbones, self.dino_model_name):
            available = [n for n in dir(backbones) if n.startswith("dinov3_")]
            raise AttributeError(f"找不到 {self.dino_model_name}，可用模型示例：{available[:30]}")
        model = getattr(backbones, self.dino_model_name)(pretrained=False)

        if self.dino_ckpt_path:
            if not os.path.isfile(self.dino_ckpt_path):
                raise FileNotFoundError(f"找不到 DINO 权重文件: {self.dino_ckpt_path}")
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
                for prefix in ("module.", "backbone.", "student.", "teacher."):
                    if nk.startswith(prefix):
                        nk = nk[len(prefix):]
                clean_state[nk] = v
            missing, unexpected = model.load_state_dict(clean_state, strict=False)
            print(
                f"[DC_v2_step1] 加载 DINOv3 权重完成: {self.dino_ckpt_path}\n"
                f"        missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
            )
        return model

    def _freeze_dino_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True):
        # DINO 主体始终 eval，Adapter 仍按外部 mode 训练。
        super().train(mode)
        self.backbone.eval()
        return self

    def _get_blocks(self) -> nn.ModuleList:
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

    def _unwrap_tokens(self, obj):
        """把 DINOv3 prepare / block 的返回值统一整理成 Tensor token。

        DINOv3 不同版本里，prepare_tokens_with_masks 可能返回 Tensor、tuple 或 dict。
        transformer block 只能接收 Tensor 或 list[Tensor]。
        我们这里手写串联 forward，所以必须拿到 B×N×C 的 Tensor。
        """
        if torch.is_tensor(obj):
            return obj

        if isinstance(obj, dict):
            # 优先找完整 token 序列
            for key in ("x", "tokens", "x_prenorm"):
                if key in obj and torch.is_tensor(obj[key]):
                    return obj[key]
            # 兜底：找第一个 3D Tensor
            for v in obj.values():
                if torch.is_tensor(v) and v.ndim == 3:
                    return v
            raise RuntimeError(f"DINO prepare 返回 dict，但找不到 token Tensor，keys={list(obj.keys())}")

        if isinstance(obj, (tuple, list)):
            # tuple/list 里优先找 3D Tensor
            for v in obj:
                if torch.is_tensor(v) and v.ndim == 3:
                    return v
            # 如果第一个元素本身是 dict，再递归处理
            if len(obj) > 0:
                return self._unwrap_tokens(obj[0])
            raise RuntimeError("DINO prepare 返回空 tuple/list，无法取得 token。")

        raise RuntimeError(f"无法识别的 DINO token 类型: {type(obj)}")

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # 优先使用 DINOv3 自带 prepare_tokens_with_masks。
        if hasattr(self.backbone, "prepare_tokens_with_masks"):
            try:
                tokens = self.backbone.prepare_tokens_with_masks(x, None)
            except TypeError:
                tokens = self.backbone.prepare_tokens_with_masks(x)
            return self._unwrap_tokens(tokens)

        if hasattr(self.backbone, "prepare_tokens"):
            tokens = self.backbone.prepare_tokens(x)
            return self._unwrap_tokens(tokens)

        # 通用 fallback。
        if not hasattr(self.backbone, "patch_embed"):
            raise RuntimeError("当前 DINOv3 backbone 没有 patch_embed / prepare_tokens_with_masks，无法手写 forward。")

        tokens = self.backbone.patch_embed(x)
        if tokens.ndim == 4:
            tokens = tokens.flatten(2).transpose(1, 2)

        b = tokens.shape[0]

        if hasattr(self.backbone, "cls_token"):
            cls = self.backbone.cls_token.expand(b, -1, -1)
            if hasattr(self.backbone, "register_tokens") and self.backbone.register_tokens is not None:
                reg = self.backbone.register_tokens.expand(b, -1, -1)
                tokens = torch.cat([cls, reg, tokens], dim=1)
            else:
                tokens = torch.cat([cls, tokens], dim=1)

        if hasattr(self.backbone, "interpolate_pos_encoding"):
            try:
                tokens = tokens + self.backbone.interpolate_pos_encoding(tokens, x.shape[-2], x.shape[-1])
            except TypeError:
                tokens = tokens + self.backbone.interpolate_pos_encoding(tokens, x)
        elif hasattr(self.backbone, "pos_embed"):
            pos = self.backbone.pos_embed
            if pos.shape[1] == tokens.shape[1]:
                tokens = tokens + pos

        return self._unwrap_tokens(tokens)


    def _run_block(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # DINO block 只应该接收 Tensor。这里先兜底整理一次。
        x = self._unwrap_tokens(x)

        try:
            y = block(x)
        except TypeError:
            y = block(x, None)

        y = self._unwrap_tokens(y)
        return y


    def _tokens_to_map(self, tokens: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        patch_n = patch_h * patch_w
        special_n = tokens.shape[1] - patch_n
        if special_n < 0:
            raise RuntimeError(f"token 数 {tokens.shape[1]} 小于 patch 数 {patch_n}，无法 reshape。")
        patch_tokens = tokens[:, special_n:, :]
        return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w).contiguous()

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        if self.dino_normalize:
            x = (x - self.mean) / self.std
        b, _, h, w = x.shape
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        tokens = self._prepare_tokens(x)
        outs: Dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.blocks):
            tokens = self._run_block(block, tokens)
            tokens = self.adapters[i](tokens)
            if i in self.out_layers:
                out_tokens = tokens
                if self.apply_norm_to_intermediate and hasattr(self.backbone, "norm"):
                    out_tokens = self.backbone.norm(out_tokens)
                outs[i] = self._tokens_to_map(out_tokens, patch_h, patch_w)
        missing = [i for i in self.out_layers if i not in outs]
        if missing:
            raise RuntimeError(f"没有取到 DINO 层 {missing}，当前 blocks 数量为 {len(self.blocks)}")
        return outs


class DC_v2_step1(nn.Module):
    """DC_v2 第一步验证模型。

    forward 返回：
    - return_aux=False：只返回 HL_base logits，方便 test/complexity。
    - return_aux=True：返回 dict，包含 final_logits、base_logits、dino_prior_logits、DINO 层权重等。
    """
    def __init__(
        self,
        num_classes: int = 1,
        n_channels: int = 3,
        pretrained: bool = True,
        return_aux: bool = False,
        dino_model_name: str = "dinov3_vits16",
        dino_repo_path: Optional[str] = None,
        dino_ckpt_path: Optional[str] = None,
        dino_layers: Sequence[int] = (2, 5, 8, 11),
        dino_embed_dim: int = 384,
        dino_patch_size: int = 16,
        adapter_bottleneck: int = 64,
        adapter_init_scale: float = 0.1,
        dino_fusion_channels: int = 64,
        dino_normalize: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.return_aux = return_aux
        self.dino_layers = list(dino_layers)

        # 当前项目里的 HL_base 位置：
        # /home/u2508183004/zyn/SEG/models/baselines/HL_base.py
        from models.baselines.HL_base import HL_base

        self.hl_base = HL_base(
            n_channels=n_channels,
            n_classes=num_classes,
            pretrained=pretrained,
            return_aux=False,
        )
        self.dino = DINOv3AdapterBranch(
            dino_model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            out_layers=dino_layers,
            embed_dim=dino_embed_dim,
            patch_size=dino_patch_size,
            adapter_bottleneck=adapter_bottleneck,
            adapter_init_scale=adapter_init_scale,
            dino_normalize=dino_normalize,
        )
        self.dino_fusion = DINOFourLayerFusion(
            in_dim=dino_embed_dim,
            out_dim=dino_fusion_channels,
            num_layers=len(dino_layers),
        )
        self.dino_prior_head = DinoPriorHead(in_channels=dino_fusion_channels, num_classes=num_classes)

    def set_return_aux(self, flag: bool):
        self.return_aux = bool(flag)

    def forward(self, x: torch.Tensor):
        base_logits, hl_aux = self.hl_base.forward_features(x)
        c1, c2, c3, c4 = hl_aux["x1"], hl_aux["x2"], hl_aux["x3"], hl_aux["x4"]

        dino_outs = self.dino(x)
        dino_feats = [dino_outs[i] for i in self.dino_layers]
        dino_sem, layer_weights = self.dino_fusion(dino_feats)
        dino_prior_logits = self.dino_prior_head(dino_sem)

        if not self.return_aux:
            return base_logits

        return {
            "final_logits": base_logits,
            "base_logits": base_logits,
            "coarse_logits": base_logits,
            "dino_prior_logits": dino_prior_logits,
            "dino_sem": dino_sem,
            "dino_layer_weights": layer_weights,
            "C1": c1,
            "C2": c2,
            "C3": c3,
            "C4": c4,
        }


if __name__ == "__main__":
    # 只用于快速检查非 DINO 部分。正式检查请在项目里传 dino_repo_path。
    print("DC_v2_step1 module loaded.")
