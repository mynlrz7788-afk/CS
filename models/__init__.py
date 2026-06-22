from .baselines.UTANet import UTANet
from .baselines.UTANetU import UTANetU
# 1. 导入搬运过来的模型
from .baselines.deeplabv3plus import DeepLabV3Plus
from .baselines.dinknet import DLinkNet34
from .baselines.Unet import UNet
from .baselines.BMDCNet import BMDCNet
from .baselines.SAM2UNet import SAM2UNet
from .baselines.SAM2MSnet import SAM2MSNet
from .baselines.CGCNet import CGCNet
from .baselines.CGCNet import CGCNet
from .baselines.CRNet import CRNet
from .baselines.AFDANet import AFDANet

from .baselines.baseline import Baseline
from .baselines.baselineU import baselineU
from .baselines.baseline_g1 import baseline_g1
from .baselines.baseline_g2 import baseline_g2
from .baselines.baseline_L import baseline_L

from .baselines.HL_base import HL_base
from .custom.HL_v1 import HL_v1
from .custom.HL_v2 import HL_v2
from .custom.HL50_v2 import HL50_v2
from .custom.HL_v3 import HL_v3
from .custom.HL_v4 import HL_v4
from .custom.DC_v1 import DC_v1

from .custom.DU_v4 import DU_v4
from .custom.DU_v3 import DU_v3

from .custom.DC_v2 import DC_v2
from .custom.DINOv3OnlySeg import DINOv3OnlySeg

from .custom.DinoRoadUNet import DinoRoadUNet

from .custom.HL_v2U import HL_v2U
# 如果在 baselines 里加了 unet.py，就在这里 from .baselines.unet import UNet
# 导入魔改模型 (假设叫 AFDANet)
# from .custom.afdanet import AFDANet
from .baselines.OARENet.oarenet_baseline import OARENet


def get_model(config, img_size=1024):
    """模型工厂：根据配置文件的名字，自动返回对应的模型"""
    model_name = config['name']
    num_classes = config.get('num_classes', 1)

    if model_name == 'DeepLabV3Plus':
        return DeepLabV3Plus(n_classes=num_classes)
    elif model_name == 'DLinkNet':
        return DLinkNet34(num_classes=num_classes)
    elif model_name == 'UNet':
        return UNet()
   
    
    # elif model_name == 'UNet':
    #     return UNet()
    # elif model_name == 'AFDANet':
    #     return AFDANet()
    # --- 如果以后魔改了 UNet 加了注意力机制 ---
    # elif model_name == 'UNet_Attention':
    #     return UNet_Attention()
    elif model_name == 'AFDANet':
        # 注意：AFDANet 初始化有 img_size 参数，默认 1024。
        # 如果你后续实验换了其他尺寸，可以在 config 里传过来，这里先用默认的
        return AFDANet(num_classes=num_classes)
    

    elif model_name == 'Baseline_v':
        return Baseline_v(num_classes=num_classes)
    elif model_name == 'Baseline_g1':
        return Baseline_g1(num_classes=num_classes)
    
    elif model_name == 'BMDCNet':
        # BMDCNet 初始化需要 img_size，用来在内部做插值缩放
        return BMDCNet(img_size=img_size, num_classes=num_classes)
    
    elif model_name == 'SAM2UNet':
        # 把预训练权重的路径传进去（写死或者从 config 里读都行）
        return SAM2UNet(checkpoint_path=config.get('hiera_path'))
    elif model_name == 'SAM2MSnet':
        # 同样使用 checkpoint_path 接收预训练权重
        return SAM2MSNet(checkpoint_path=config.get('hiera_path'))
    elif model_name == 'CGCNet':
        return CGCNet(out_channels=num_classes, img_size=img_size)
    elif model_name == 'CRNet':
        return CRNet(num_classes=num_classes)
    elif model_name == 'OARENet':
        return OARENet(num_classes=num_classes)
    
    elif model_name == 'Baseline':
        return Baseline(num_classes=num_classes)
    
    elif model_name == 'baselineU':
        return baselineU(num_classes=num_classes)
    elif model_name == 'baseline_g1':
        return baseline_g1(num_classes=num_classes)
    elif model_name == 'baseline_g2':
        return baseline_g2(num_classes=num_classes)
        
    elif model_name == 'baseline_L':
        return baseline_L(num_classes=num_classes)
    elif model_name == 'HL_base':
        return HL_base(num_classes=num_classes)
    

    elif model_name == 'HL_base_BS4':
        from .baselines.HL_base_BS4 import HL_base_BS4
        return HL_base_BS4(
            num_classes=config.get('num_classes', config.get('n_classes', 1)),
            n_channels=config.get('n_channels', 3),
            pretrained=config.get('pretrained', True),
            return_aux=config.get('return_aux', False),
        )
    

    elif model_name == "DC_v4_3":
        from .custom.DC_v4_3 import DC_v4_3
        return DC_v4_3(**config)
    
    elif model_name == "RD_v1":
        from .custom.RD_v1 import RD_v1
        return RD_v1(**config)
    
    elif model_name =="RD_v1_A":
        from .custom.RD_v1_A import RD_v1_A
        return RD_v1_A(**config)
    
    elif model_name =="RD_v1_AB":
        from .custom.RD_v1_AB import RD_v1_AB
        return RD_v1_AB(**config)
    elif model_name =="RD_v1_C":
        from .custom.RD_v1_C import RD_v1_C
        return RD_v1_C(**config)

    elif model_name == "RD_v2":
        from .custom.RD_v2 import RD_v2
        return RD_v2(**config)
    
    elif model_name =="RD_v3":
        from .custom.RD_v3 import RD_v3
        return RD_v3(**config)
    
    elif model_name =="RD_v3A":
        from .custom.RD_v3A import RD_v3A
        return RD_v3A(**config)
    
    elif model_name =="RD_v4":
        from .custom.RD_v4 import RD_v4
        return RD_v4(**config)

    elif model_name =="RD_v5":
        from .custom.RD_v5 import RD_v5
        return RD_v5(**config)

    elif model_name == 'DinoRoadUNet':
        return DinoRoadUNet(
            num_classes=config.get('num_classes', 1),
            dinov3_model=config.get('dinov3_model', 'dinounet_s'),
            pretrained_path=config.get('pretrained_path'),
            out_channels=config.get('out_channels', [64, 128, 256, 512]),
            rank=config.get('rank', 256),
            img_size=img_size,
            freeze_backbone=config.get('freeze_backbone', True),
            imagenet_norm=config.get('imagenet_norm', True),
            input_already_normalized=config.get('input_already_normalized', False),
            conv_inplane=config.get('conv_inplane', 64),
            deform_num_heads=config.get('deform_num_heads', 12),
            n_points=config.get('n_points', 4),
            with_cp=config.get('with_cp', False)
        )
    
    elif model_name == 'DINOv3OnlySeg':
        return DINOv3OnlySeg(
            num_classes=config.get('num_classes', 1),
            dinov3_model=config.get('dinov3_model', 'dinounet_s'),
            pretrained_path=config.get('pretrained_path'),
            img_size=img_size,
            freeze_backbone=config.get('freeze_backbone', True),
            imagenet_norm=config.get('imagenet_norm', True),
            input_already_normalized=config.get('input_already_normalized', False),
            layer_idx=config.get('layer_idx', None)
        )
    

    

    

    
    elif model_name == 'DC_v3':
        from models.custom.DC_v3 import DC_v3
        return DC_v3(num_classes=num_classes)
    
    
    elif model_name == "DC_V3_1":
        from models.custom.DC_V3_1 import DC_V3_1
        return DC_V3_1(**config)
    
    elif model_name == "DC_V3_2":
        from models.custom.DC_V3_2 import DC_V3_2
        return DC_V3_2(**config)
    
    




    else:
        raise ValueError(f"没找到模型: {model_name}，请检查名字！")
