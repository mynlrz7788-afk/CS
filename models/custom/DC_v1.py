import os
import sys
import math
import importlib
from typing import Optional, Sequence, Dict, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DC_v1"]


class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=None):
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class FrozenDINOv3FeatureExtractor(nn.Module):
    """
    直接从 dinounet/dinov3 引入 DINOv3，不再依赖 DC_v2_step2.py。

    功能：
    1. 从 dinounet/dinov3/hub/backbones.py 构建 DINOv3 backbone
    2. 加载本地 DINOv3 权重
    3. 冻结全部 DINOv3 原始参数
    4. 手写 forward，提取指定 block 后的 patch feature map
    5. 返回多层 2D feature map

    注意：
    这个类没有内部 Adapter。
    它只负责提取 frozen DINO dense features。
    """

    def __init__(
        self,
        dino_model_name: str = "dinov3_vits16",
        dino_repo_path: str = "/home/u2508183004/zyn/SEG/dinounet/dinov3",
        dino_ckpt_path: str = "/home/u2508183004/zyn/SEG/weight/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        out_layers: Sequence[int] = (2, 5, 8, 11),
        embed_dim: int = 384,
        patch_size: int = 16,
        dino_normalize: bool = False,
    ):
        super().__init__()

        self.dino_model_name = dino_model_name
        self.dino_repo_path = dino_repo_path
        self.dino_ckpt_path = dino_ckpt_path
        self.out_layers = list(out_layers)
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

        max_layer = max(self.out_layers)
        if max_layer >= len(self.blocks):
            raise RuntimeError(
                f"dino_layers={self.out_layers} 超出 DINO blocks 数量。"
                f"当前 blocks 数量={len(self.blocks)}"
            )

        self._freeze_backbone()

    def _build_dino_backbone(self):
        if self.dino_repo_path is None:
            raise ValueError(
                "需要提供 dino_repo_path，例如 "
                "/home/u2508183004/zyn/SEG/dinounet/dinov3"
            )

        if not os.path.isdir(self.dino_repo_path):
            raise FileNotFoundError(f"找不到 dino_repo_path: {self.dino_repo_path}")

        # dino_repo_path = .../dinounet/dinov3
        # 需要把 .../dinounet 加进 sys.path，之后才能 import dinov3.xxx
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
                f"[DC_v1] 加载 DINOv3 权重完成: {self.dino_ckpt_path}\n"
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

    def train(self, mode: bool = True):
        super().train(mode)

        # 无论外部 model.train() 怎么调用，DINO backbone 都保持 eval。
        self.backbone.eval()

        return self

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

    def _prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        兼容不同 DINOv3 版本的 token 准备方式。
        优先使用 prepare_tokens_with_masks，其次 prepare_tokens，最后手写 patch_embed。
        """

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

            if hasattr(self.backbone, "register_tokens") and self.backbone.register_tokens is not None:
                reg = self.backbone.register_tokens.expand(b, -1, -1)
                tokens = torch.cat([cls, reg, tokens], dim=1)
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

    def _run_block(self, block: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self._unwrap_tokens(tokens)

        try:
            out = block(tokens)
        except TypeError:
            out = block(tokens, None)

        return self._unwrap_tokens(out)

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
        """
        tokens: B × N × C
        输出:   B × C × patch_h × patch_w

        自动去掉 cls token / register token。
        """

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

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        使用 DINOv3 官方 get_intermediate_layers 提取中间层 dense features。

        返回：
            {
                layer_idx: B × C × H/patch × W/patch
            }

        说明：
            这里不再手写跑 blocks。
            cls token / storage token / reshape 交给 DINOv3 官方接口处理。
        """

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
                "请继续使用手写 blocks 版本，或检查 vision_transformer.py。"
            )

        # 为了尽量复现之前 DC_v1 的结果，这里先用 norm=False。
        # 因为之前手写 blocks 取的是 block 输出，没有额外经过 final norm。
        # 后面可以单独做一次 norm=True 对照。
        try:
            feats = self.backbone.get_intermediate_layers(
                x,
                n=self.out_layers,
                reshape=True,
                return_class_token=False,
                norm=False,
            )
        except TypeError:
            # 兼容某些版本不支持 norm 参数
            try:
                feats = self.backbone.get_intermediate_layers(
                    x,
                    n=self.out_layers,
                    reshape=True,
                    return_class_token=False,
                )
            except TypeError:
                # 兼容某些版本不支持 list 形式的 n
                # 这种情况下只能取最后 max_layer+1 层，再筛选。
                max_layer = max(self.out_layers)
                raw_feats = self.backbone.get_intermediate_layers(
                    x,
                    n=max_layer + 1,
                    reshape=True,
                    return_class_token=False,
                )

                # raw_feats 如果返回的是最后 n 层，这个 fallback 不一定严格等价。
                # 所以正常情况下应该优先走上面的 list 形式。
                if len(raw_feats) < len(self.out_layers):
                    raise RuntimeError(
                        "get_intermediate_layers 不支持 layer list，且 fallback 返回层数不足。"
                        f"需要层: {self.out_layers}, 实际返回: {len(raw_feats)}"
                    )

                feats = [raw_feats[i] for i in self.out_layers]

        if isinstance(feats, torch.Tensor):
            feats = [feats]

        feats = list(feats)

        # 如果 return_class_token=True 或某些版本返回 (patch, cls)，这里做兼容。
        clean_feats = []

        for feat in feats:
            if isinstance(feat, (tuple, list)):
                feat = feat[0]

            if feat.dim() == 3:
                # 如果官方接口没有 reshape 成 2D，这里兜底 reshape。
                b, n, c = feat.shape
                patch_h = h // self.patch_size
                patch_w = w // self.patch_size
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

            if feat.dim() != 4:
                raise RuntimeError(
                    f"get_intermediate_layers 返回特征维度异常: {feat.shape}"
                )

            clean_feats.append(feat)

        if len(clean_feats) != len(self.out_layers):
            raise RuntimeError(
                f"DINO 输出层数不匹配。"
                f"期望 {len(self.out_layers)} 层: {self.out_layers}, "
                f"实际得到 {len(clean_feats)} 层。"
            )

        outputs = {
            layer: feat
            for layer, feat in zip(self.out_layers, clean_feats)
        }

        return outputs


class DinoOutAdapter(nn.Module):
    """
    DINO 输出后的外置 Adapter。

    作用：
    1. 通道降维
    2. 加一点局部卷积归纳偏置
    3. 把 DINO feature map 转成道路分割 decoder 能用的特征

    注意：
    这不是微调 DINO 原始参数。
    """

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


class UpBlock(nn.Module):
    """
    简单上采样块。

    这一版是 DINO-only Probe，不加复杂模块。
    目的不是追最高指标，而是验证 DINO dense features 里有没有道路信息。
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Sequential(
            ConvBNAct(in_ch, out_ch, kernel_size=3),
            ConvBNAct(out_ch, out_ch, kernel_size=3),
        )

    def forward(self, x, size):
        x = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
        )
        x = self.conv(x)
        return x


class DC_v1(nn.Module):
    """
    DC_v1: DINO-only Road Probe

    目的：
        验证冻结 DINOv3 dense features 对道路分割有没有用。

    当前版本：
        不再依赖 DC_v2_step2.py。
        直接从 dinounet/dinov3 引入 DINOv3。
        DINO 原始参数全部冻结。
        只训练 DINO 输出后的外置 Adapter 和轻量 Decoder。

    结构：
        输入图像
        -> Frozen DINOv3
        -> 提取多层 DINO feature map
        -> 外置 DinoOutAdapter
        -> 多层特征融合
        -> 轻量上采样 decoder
        -> 道路 mask
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

        adapter_channels=128,
        adapter_bottleneck=64,
        adapter_init_scale=0.1,

        decoder_channels=(256, 128, 64, 32, 32),

        # 保留这个参数，是为了兼容旧 json。
        # 当前版本没有 DINO 内部 Adapter，所以这个参数不会生效。
        train_dino_branch_adapters=False,

        **kwargs,
    ):
        super().__init__()

        self.num_classes = int(num_classes)
        self.n_channels = int(n_channels)
        self.return_aux = bool(return_aux)

        self.dino_layers = list(dino_layers)
        self.dino_embed_dim = int(dino_embed_dim)
        self.dino_patch_size = int(dino_patch_size)

        self.adapter_channels = int(adapter_channels)
        self.num_dino_levels = len(self.dino_layers)

        self.dino = FrozenDINOv3FeatureExtractor(
            dino_model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            dino_ckpt_path=dino_ckpt_path,
            out_layers=dino_layers,
            embed_dim=dino_embed_dim,
            patch_size=dino_patch_size,
            dino_normalize=dino_normalize,
        )

        self.dino_adapters = nn.ModuleList([
            DinoOutAdapter(
                in_ch=dino_embed_dim,
                out_ch=adapter_channels,
                bottleneck=adapter_bottleneck,
            )
            for _ in range(self.num_dino_levels)
        ])

        fuse_in_ch = adapter_channels * self.num_dino_levels
        dec0, dec1, dec2, dec3, dec4 = decoder_channels

        self.fuse = nn.Sequential(
            ConvBNAct(fuse_in_ch, dec0, kernel_size=1, padding=0),
            ConvBNAct(dec0, dec0, kernel_size=3),
        )

        self.up8 = UpBlock(dec0, dec1)
        self.up4 = UpBlock(dec1, dec2)
        self.up2 = UpBlock(dec2, dec3)
        self.up1 = UpBlock(dec3, dec4)

        self.out_head = nn.Conv2d(dec4, num_classes, kernel_size=1)

        self._print_trainable_summary()

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
        print("🔧 DC_v1 DINO-only Road Probe")
        print(f"    - 总参数量:       {total / 1e6:.2f} M")
        print(f"    - 可训练参数量:   {trainable / 1e6:.2f} M")
        print(f"    - DINO分支可训练: {dino_trainable / 1e6:.2f} M")
        print("    - 说明: DINOv3 直接从 dinounet/dinov3 引入；DINO 原始参数冻结")
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
            max(1, h // 16),
            max(1, w // 16),
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

        f8 = self.up8(
            f16,
            size=(max(1, h // 8), max(1, w // 8)),
        )
        f4 = self.up4(
            f8,
            size=(max(1, h // 4), max(1, w // 4)),
        )
        f2 = self.up2(
            f4,
            size=(max(1, h // 2), max(1, w // 2)),
        )
        f1 = self.up1(
            f2,
            size=input_hw,
        )

        logits = self.out_head(f1)

        if self.return_aux:
            return {
                "final_logits": logits,
                "logits": logits,
                "dino_feats": dino_feats,
                "adapted_feats": adapted,
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