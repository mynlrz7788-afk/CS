# -*- coding: utf-8 -*-
"""
MTNet.py

把 MTNet / MT-RoadNet 的道路面分割分支接入 SEG 框架。

放置位置：
    SEG/models/baselines/MTNet.py

要求 MTNet 官方 networks 目录下的网络文件已经复制到：
    SEG/models/baselines/mtnet/

至少应包含：
    LGC_Encoder.py
    LGFF_module.py
    curfnet.py
    center_bridge.py
    vit_encoder.py
    Decoder.py
    DeformableDecoder.py
    basic_blocks.py
    resnet.py
    sam/
    以及 VSSM / selective_scan 相关文件

同时建议执行：
    touch SEG/models/baselines/mtnet/__init__.py
"""

import os
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mtnet.LGC_Encoder import LGC_Encoder


class _MTNetConfig(SimpleNamespace):
    """
    给官方 MTNet 网络文件提供 config.xxx 形式的参数。
    """

    def __getattr__(self, name):
        defaults = {
            # 数据相关
            "DATASET": "road",
            "DATA_FUSION_TYPE": "I",
            "PATCH_SIZE": 512,
            "LITTLE_DATA_TEST_CODE": False,

            # SAM 相关
            "SAM_VERSION": "vit_b",
            "SAM_CKPT_PATH": "",

            # 主体结构开关
            "USE_LGFF": True,
            "USE_SCATT": False,
            "USE_SCMA": True,
            "USE_VSSM_SS2D": True,
            "USE_RESENCODER_ADD_VITENCODER": True,

            # curfnet / decoder 相关
            # 你本地 curfnet.py 里用到了这个字段。
            # True 表示使用 DeformableRoadDecoder。
            # False 表示回到 DecoderBlock1DConv4。
            "USE_NEW_DLINK_DECODER": True,

            # curfnet.forward 里可能会用到。
            # 当前 LGC_Encoder.forward 不直接调用 curfnet.forward，
            # 但补上可以避免后续测试时报错。
            "DLINKNET_ENCODER_AND_VIT": False,

            # 兼容官方代码里的其他字段
            "ORIGINAL_DLINK_SAM_ROAD": False,
            "FOCAL_LOSS": False,
            "USE_BCE_DICE": True,
            "TOPONET_VERSION": "normal",

            # 训练相关
            "BASE_LR": 0.0001,
            "TRAIN_EPOCHS": 120,
            "LOAD_TOTAL_WEIGHT": False,
            "LOAD_TOTAL_WEIGHT_PATH": "",
            "LOAD_OPT_SCH": False,

            # 推理相关
            "INFER_BATCH_SIZE": 64,
            "SAMPLE_MARGIN": 0,
            "INFER_PATCHES_PER_EDGE": 16,
            "INFER_BATCH_SIZE_HEATMAP": 6,

            # 路径相关
            "ROAD_RESULT_PATH": "",
            "TRAIN_WEIGHT": "",
        }

        if name in defaults:
            return defaults[name]

        raise AttributeError(f"MTNet config has no attribute: {name}")


class MTNet(nn.Module):
    """
    SEG 框架可直接调用的 MTNet。

    输入：
        x: [B, C, H, W]

    默认假设：
        x 已经被你的 RoadDataset 做过 ImageNet Normalize：
            mean = [0.485, 0.456, 0.406]
            std  = [0.229, 0.224, 0.225]

    本文件会先把它还原到 0-255，
    再按官方 MTNet / SAM 的 pixel_mean / pixel_std 重新归一化。

    输出：
        logits: [B, 1, H, W]

    注意：
        输出不做 sigmoid。
        继续配合你 SEG 里的 BCEWithLogitsLoss + DiceLoss 使用。
    """

    def __init__(
        self,
        num_classes=1,
        n_channels=3,
        img_size=512,
        sam_version="vit_b",
        sam_ckpt_path="",
        data_fusion_type="I",
        use_lgff=True,
        use_scma=True,
        use_vssm=True,
        use_resencoder_add_vitencoder=True,
        input_is_imagenet_normalized=True,
    ):
        super().__init__()

        assert sam_version in {"vit_b", "vit_l", "vit_h"}, \
            f"sam_version must be vit_b, vit_l or vit_h, got {sam_version}"

        self.num_classes = num_classes
        self.n_channels = n_channels
        self.img_size = img_size
        self.sam_version = sam_version
        self.sam_ckpt_path = sam_ckpt_path
        self.data_fusion_type = data_fusion_type
        self.input_is_imagenet_normalized = input_is_imagenet_normalized

        if data_fusion_type == "ITO":
            in_chans = 5
        else:
            in_chans = n_channels

        self.in_chans = in_chans

        if sam_version == "vit_b":
            encoder_embed_dim = 768
            encoder_depth = 12
            encoder_num_heads = 12
            encoder_global_attn_indexes = [2, 5, 8, 11]
        elif sam_version == "vit_l":
            encoder_embed_dim = 1024
            encoder_depth = 24
            encoder_num_heads = 16
            encoder_global_attn_indexes = [5, 11, 17, 23]
        else:
            encoder_embed_dim = 1280
            encoder_depth = 32
            encoder_num_heads = 16
            encoder_global_attn_indexes = [7, 15, 23, 31]

        prompt_embed_dim = 256
        vit_patch_size = 16

        self.config = _MTNetConfig(
            DATASET="road",
            DATA_FUSION_TYPE=data_fusion_type,
            PATCH_SIZE=img_size,

            SAM_VERSION=sam_version,
            SAM_CKPT_PATH=sam_ckpt_path,

            USE_LGFF=use_lgff,
            USE_SCATT=False,
            USE_SCMA=use_scma,
            USE_VSSM_SS2D=use_vssm,
            USE_RESENCODER_ADD_VITENCODER=use_resencoder_add_vitencoder,

            USE_NEW_DLINK_DECODER=True,
            DLINKNET_ENCODER_AND_VIT=False,

            ORIGINAL_DLINK_SAM_ROAD=False,

            FOCAL_LOSS=False,
            USE_BCE_DICE=True,
            TOPONET_VERSION="normal",

            BASE_LR=0.0001,
            TRAIN_EPOCHS=120,
        )

        # 你的 RoadDataset 常用的 ImageNet Normalize 参数
        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        # 官方 MTNet / SAM 的 pixel_mean / pixel_std
        if data_fusion_type == "ITO":
            self.register_buffer(
                "pixel_mean",
                torch.tensor([123.675, 116.28, 103.53, 114.49, 114.49]).view(1, 5, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "pixel_std",
                torch.tensor([58.395, 57.12, 57.375, 57.63, 57.63]).view(1, 5, 1, 1),
                persistent=False,
            )
        else:
            self.register_buffer(
                "pixel_mean",
                torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "pixel_std",
                torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
                persistent=False,
            )

        # 和官方 model.py 里构建 LGC_Encoder 的方式保持一致
        self.lgc_encoder = LGC_Encoder(
            self.config,
            in_chans=in_chans,
            encoder_1dconv=0,
            decoder_1dconv=4,
            num_classes=num_classes,
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=img_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        )

        # 官方代码里会记录成功匹配到的 SAM 参数名，分组学习率会用到
        self.matched_param_names = set()

        if sam_ckpt_path is not None and sam_ckpt_path != "":
            self.load_sam_checkpoint(
                sam_ckpt_path=sam_ckpt_path,
                image_size=img_size,
                vit_patch_size=vit_patch_size,
                encoder_global_attn_indexes=encoder_global_attn_indexes,
            )
        else:
            print("========== MTNet 提示 ==========")
            print("sam_ckpt_path 为空，当前 ViT 分支将随机初始化。")
            print("这只能用于跑通流程，不能作为复现结果。")
            print("================================")

    def _imagenet_to_pixel_255(self, x):
        """
        把 SEG RoadDataset 输出的 ImageNet Normalize 图像还原成 0-255。

        输入：
            x = (img_0_1 - imagenet_mean) / imagenet_std

        输出：
            x_255，大致范围 0-255
        """
        if x.shape[1] < 3:
            raise ValueError(f"MTNet expects at least 3 channels, got {x.shape[1]}")

        if x.shape[1] == 3:
            x = x * self.imagenet_std + self.imagenet_mean
            x = torch.clamp(x, 0.0, 1.0)
            x = x * 255.0
            return x

        # 预留给 ITO 多源输入。
        # 前 3 个通道按 ImageNet 还原，后续通道保持原样。
        rgb = x[:, :3, :, :]
        extra = x[:, 3:, :, :]

        rgb = rgb * self.imagenet_std + self.imagenet_mean
        rgb = torch.clamp(rgb, 0.0, 1.0)
        rgb = rgb * 255.0

        return torch.cat([rgb, extra], dim=1)

    def _sam_normalize(self, x):
        """
        官方 MTNet / SAM 的归一化：
            x = (x - pixel_mean) / pixel_std
        """
        return (x - self.pixel_mean) / self.pixel_std

    def forward(self, x):
        """
        输入：
            x: [B, C, H, W]

        输出：
            logits: [B, 1, H, W]
        """
        if x.dim() != 4:
            raise ValueError(f"MTNet expects input [B, C, H, W], got shape {tuple(x.shape)}")

        if x.shape[1] != self.in_chans:
            raise ValueError(
                f"MTNet expects {self.in_chans} input channels, got {x.shape[1]}"
            )

        if self.input_is_imagenet_normalized:
            x = self._imagenet_to_pixel_255(x)
        else:
            # 如果以后单独写不做 ImageNet Normalize 的 dataset，
            # 并且输入范围是 0-1，这里转成 0-255。
            with torch.no_grad():
                max_val = float(x.detach().max().cpu())
            if max_val <= 2.0:
                x = x * 255.0

        x = self._sam_normalize(x)

        logits = self.lgc_encoder(x)

        return logits

    def load_sam_checkpoint(
        self,
        sam_ckpt_path,
        image_size,
        vit_patch_size,
        encoder_global_attn_indexes,
    ):
        """
        按官方 model.py 的方式加载 SAM image_encoder 权重。

        官方逻辑：
        1. 读取 SAM ckpt。
        2. 如果输入尺寸不是 1024，resize pos_embed 和 rel_pos。
        3. 把 image_encoder.xxx 映射成 lgc_encoder.cen_encoder.xxx。
        4. 只加载名称和 shape 都匹配的参数。
        """
        if not os.path.isfile(sam_ckpt_path):
            raise FileNotFoundError(f"SAM checkpoint not found: {sam_ckpt_path}")

        ckpt_state_dict = torch.load(sam_ckpt_path, map_location="cpu")

        if isinstance(ckpt_state_dict, dict) and "state_dict" in ckpt_state_dict:
            ckpt_state_dict = ckpt_state_dict["state_dict"]

        # 官方在 ITO 5 通道输入时会重新初始化 patch_embed.proj.weight
        if self.config.DATA_FUSION_TYPE == "ITO":
            nn.init.kaiming_normal_(
                self.lgc_encoder.cen_encoder.patch_embed.proj.weight,
                mode="fan_out",
            )
            ckpt_state_dict["image_encoder.patch_embed.proj.weight"] = \
                self.lgc_encoder.cen_encoder.patch_embed.proj.weight.detach().cpu()

        if image_size != 1024:
            ckpt_state_dict = self.resize_sam_pos_embed(
                state_dict=ckpt_state_dict,
                image_size=image_size,
                vit_patch_size=vit_patch_size,
                encoder_global_attn_indexes=encoder_global_attn_indexes,
            )

        new_ckpt_state_dict = {}

        if not self.config.ORIGINAL_DLINK_SAM_ROAD:
            for key, value in ckpt_state_dict.items():
                if key.split(".")[0] == "image_encoder":
                    new_key = key.replace("image_encoder", "lgc_encoder.cen_encoder")
                    new_ckpt_state_dict[new_key] = value
                else:
                    new_ckpt_state_dict[key] = value
            ckpt_state_dict = new_ckpt_state_dict

        matched_names = []
        mismatch_names = []
        state_dict_to_load = {}

        for k, v in self.named_parameters():
            if k in ckpt_state_dict and v.shape == ckpt_state_dict[k].shape:
                matched_names.append(k)
                state_dict_to_load[k] = ckpt_state_dict[k]
            else:
                mismatch_names.append(k)

        self.matched_param_names = set(matched_names)

        self.load_state_dict(state_dict_to_load, strict=False)

        print("========== MTNet SAM 权重加载信息 ==========")
        print(f"SAM ckpt: {sam_ckpt_path}")
        print(f"Matched params: {len(matched_names)}")
        print(f"Mismatched or not loaded params: {len(mismatch_names)}")
        print("==========================================")

    def resize_sam_pos_embed(
        self,
        state_dict,
        image_size,
        vit_patch_size,
        encoder_global_attn_indexes,
    ):
        """
        和官方 model.py 里的 resize_sam_pos_embed 逻辑保持一致。
        512 输入时，SAM 原始 64x64 位置编码会插值成 32x32。
        1024 输入时不会走这里。
        """
        new_state_dict = {k: v for k, v in state_dict.items()}

        if "image_encoder.pos_embed" not in new_state_dict:
            return new_state_dict

        pos_embed = new_state_dict["image_encoder.pos_embed"]
        token_size = int(image_size // vit_patch_size)

        if pos_embed.shape[1] != token_size:
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed,
                (token_size, token_size),
                mode="bilinear",
                align_corners=False,
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1)
            new_state_dict["image_encoder.pos_embed"] = pos_embed

        rel_pos_keys = [k for k in state_dict.keys() if "rel_pos" in k]
        global_rel_pos_keys = [
            k for k in rel_pos_keys
            if any([str(i) in k for i in encoder_global_attn_indexes])
        ]

        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape

            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(
                rel_pos_params,
                (token_size * 2 - 1, w),
                mode="bilinear",
                align_corners=False,
            )
            new_state_dict[k] = rel_pos_params[0, 0, ...]

        return new_state_dict

    def get_param_groups(self):
        """
        可选。
        如果你后面在 train.py 里支持 model.get_param_groups()，
        就可以复现官方分组学习率。

        官方大致设置：
            SAM / ViT 已加载参数：0.00005
            curfnet：0.0002
            LGFF：0.0002
            SCMA / SCATT：0.0002
            VSSM：0.0002
            map_decoder：0.001
        """
        param_groups = []

        cen_encoder_params = [
            p for k, p in self.lgc_encoder.cen_encoder.named_parameters()
            if ("lgc_encoder.cen_encoder." + k) in self.matched_param_names
            and p.requires_grad
        ]

        if len(cen_encoder_params) > 0:
            param_groups.append({
                "params": cen_encoder_params,
                "lr": 0.00005,
            })

        param_groups.append({
            "params": [p for p in self.lgc_encoder.curfnet.parameters() if p.requires_grad],
            "lr": 0.0002,
        })

        if self.config.USE_LGFF:
            param_groups.append({
                "params": [p for p in self.lgc_encoder.LGFF.parameters() if p.requires_grad],
                "lr": 0.0002,
            })

        if self.config.USE_SCATT or self.config.USE_SCMA:
            param_groups.append({
                "params": [p for p in self.lgc_encoder.curf_center_bridge.parameters() if p.requires_grad],
                "lr": 0.0002,
            })

        if self.config.USE_VSSM_SS2D:
            param_groups.append({
                "params": [p for p in self.lgc_encoder.vssm_module.parameters() if p.requires_grad],
                "lr": 0.0002,
            })

        param_groups.append({
            "params": [p for p in self.lgc_encoder.map_decoder.parameters() if p.requires_grad],
            "lr": 0.001,
        })

        return param_groups