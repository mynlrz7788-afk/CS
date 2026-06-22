import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DRIVEDataset(Dataset):
    # 🌟 1. 这里的参数必须加上 dataset_name，和 train.py 里传过来的一致！
    def __init__(self, root_path, dataset_name, mode='train', img_size=512):
        self.mode = mode
        self.img_size = img_size
        
        # 🌟 2. 拼接出完整的 DRIVE 数据集路径 
        # (/home/u2508183004/zyn/datasets + DRIVE)
        dataset_root = os.path.join(root_path, dataset_name)

        # 🌟 3. 下面的路径全都要改成从 dataset_root 开始找
        if mode == 'train':
            self.img_dir = os.path.join(dataset_root, 'train', 'images')
            self.mask_dir = os.path.join(dataset_root, 'train', '1st_manual')
            self.img_suffix = '_training.tif'
            self.mask_suffix = '_manual1.gif'
        elif mode == 'val':
            self.img_dir = os.path.join(dataset_root, 'val', 'images')
            self.mask_dir = os.path.join(dataset_root, 'val', '1st_manual')
            self.img_suffix = '_training.tif'
            self.mask_suffix = '_manual1.gif'
        elif mode == 'test':
            self.img_dir = os.path.join(dataset_root, 'test', 'images')
            self.mask_dir = os.path.join(dataset_root, 'test', '1st_manual')
            self.img_suffix = '_test.tif'
            self.mask_suffix = '_manual1.gif'
        else:
            raise ValueError(f"不支持的模式: {mode}")

        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.tif')]

        # 🌟 核心策略：训练裁剪 512，验证/测试边缘纯黑填充到 32 的倍数
        if mode == 'train':
            self.transform = A.Compose([
                # 训练时：在 565x584 的原图上随机切出 512x512 的块
                A.RandomCrop(height=img_size, width=img_size),  
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                # 验证/测试时：保持 565x584 原始比例，边缘用黑色 (0) 填充到 576x608
                A.PadIfNeeded(
                    min_height=None,  # 🌟 新增：显式关闭固定高度
                    min_width=None,   # 🌟 新增：显式关闭固定宽度
                    pad_height_divisor=32, 
                    pad_width_divisor=32, 
                    border_mode=cv2.BORDER_CONSTANT, 
                    value=0,        
                    mask_value=0    
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        # 名字转换逻辑：例如 21_training.tif -> 21_manual1.gif
        base_id = img_name.split('_')[0] 
        mask_name = base_id + self.mask_suffix

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # ⚠️ 注意：读取 .gif 格式必须用 PIL.Image，不能用 cv2
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = np.array(image)
        mask = np.array(mask)

        # 二值化：将 255 的血管变为 1.0，背景变为 0.0
        mask = (mask > 127).astype(np.float32)

        # 执行图像增强/裁剪/填充
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask'].unsqueeze(0) # 加上通道维度变为 [1, H, W]

        return image, mask