import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class RoadDataset(Dataset):
    def __init__(self, root_path, dataset_name, mode='train', img_size=1024):
        self.mode = mode
        self.dataset_name = dataset_name
        self.img_size = img_size

        # 只保留这一种目录结构：
        # root_path = /home/u2508183004/zyn/datasets
        # dataset_name = Mass / DeepGlobe / Massachusetts ...
        #
        # 实际读取：
        # /home/u2508183004/zyn/datasets/Mass/train/images
        # /home/u2508183004/zyn/datasets/Mass/train/masks
        self.img_dir = os.path.join(root_path, dataset_name, mode, 'images')
        self.mask_dir = os.path.join(root_path, dataset_name, mode, 'masks')

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"图片目录不存在: {self.img_dir}")
        if not os.path.isdir(self.mask_dir):
            raise FileNotFoundError(f"标签目录不存在: {self.mask_dir}")

        self.img_list = sorted(os.listdir(self.img_dir))

        # =====================================================
        # 原来的预处理全部保留
        # train: RandomCrop + 颜色增强 + 几何增强 + Normalize
        # val/test: CenterCrop + Normalize
        # =====================================================
        if self.mode == 'train':
            self.transform = A.Compose([
                A.RandomCrop(height=img_size, width=img_size),

                A.HueSaturationValue(
                    hue_shift_limit=30,
                    sat_shift_limit=5,
                    val_shift_limit=15,
                    p=0.5
                ),

                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.3,
                    rotate_limit=0,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    mask_value=0,
                    p=0.5
                ),

                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                A.CenterCrop(height=img_size, width=img_size),

                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)

        dataset_name = self.dataset_name.lower()

        # =====================================================
        # 根据不同数据集匹配 mask 文件名
        # =====================================================
        if dataset_name == 'deepglobe':

            # image: 12345_sat.jpg
            # mask : 12345_mask.png
            mask_name = img_name.replace('_sat.jpg', '_mask.png')

        elif dataset_name in ['mass', 'massachusetts', 'lmassachusetts']:

            # image: 10378780_15.tiff
            # mask : 10378780_15.tif
            mask_name = img_name.replace('.tiff', '.tif')

        elif self.dataset_name.lower() == 'drive_512':
            mask_name = os.path.splitext(img_name)[0] + '.png'

        elif dataset_name in ['chn', 'chn6', 'chn6_cug', 'chn6-cug']:

            # CHN 数据集可能有两种命名：
            # image: am100001_sat.jpg
            # mask : am100001.png
            #
            # 或者：
            # image: am100001_sat.jpg
            # mask : am100001_sat.png
            #
            # 这里做兼容匹配。

            base, ext = os.path.splitext(img_name)

            possible_bases = [base]

            if base.endswith('_sat'):
                possible_bases.append(base[:-4])

            possible_exts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

            candidates = []

            for b in possible_bases:
                for e in possible_exts:
                    candidates.append(b + e)
                    candidates.append(b + '_gt' + e)
                    candidates.append(b + '_mask' + e)
                    candidates.append(b + '_label' + e)

            mask_name = None

            for cand in candidates:
                cand_path = os.path.join(self.mask_dir, cand)
                if os.path.isfile(cand_path):
                    mask_name = cand
                    break

            if mask_name is None:
                raise ValueError(
                    f"CHN 数据集中找不到对应标签。\n"
                    f"图像名: {img_name}\n"
                    f"标签目录: {self.mask_dir}\n"
                    f"尝试过的前几个名字: {candidates[:10]}"
                )

        else:

            raise ValueError(
                f"未知的 Dataset 名字: {self.dataset_name}，请检查 JSON 配置！\n"
                f"目前 RoadDataset 支持: DeepGlobe, Mass, Massachusetts, LMassachusetts, CHN, DRIVE_512"
            )
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"图片损坏或路径错误，无法读取: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"标签损坏或路径错误，无法读取: {mask_path}")

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']

        mask = (mask > 0).float()
        mask = mask.unsqueeze(0)

        return image, mask, img_name