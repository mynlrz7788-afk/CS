import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .ta_mosc import MoE

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UTANetU(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, img_size=1024, pretrained=True):
        super(UTANetU, self).__init__()
        self.img_size = img_size
        self.pretrained = pretrained
        
        # Encoder (ResNet34)
        self.resnet = models.resnet34(weights='DEFAULT' if pretrained else None)
        self.filters_resnet = [64, 64, 128, 256, 512]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(2, 2)
        
        # TA-MoSC (MoE)
        self.moe = MoE(in_channels=512, out_channels=64)
        self.fuse = nn.Sequential(
            nn.Conv2d(512, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Docker 模块用于尺度对齐
        self.docker1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.docker2 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.docker3 = nn.Sequential(nn.Conv2d(64, 128, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.docker4 = nn.Sequential(nn.Conv2d(64, 256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        # Decoder
        self.up5 = UpBlock(512, 256)
        self.up4 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up2 = UpBlock(64, 32)
        
        # 主输出预测头
        self.pred = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1)
        )
        #self.sigmoid = nn.Sigmoid()

        # 🌟 深监督预测头 

        self.aux_head5 = nn.Conv2d(256, n_classes, 1)
        self.aux_head4 = nn.Conv2d(128, n_classes, 1)
        self.aux_head3 = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.conv1(x)        # (B, 64, 1024, 1024)
        e2 = self.resnet.layer1(self.maxpool(e1)) # (B, 64, 512, 512)
        e3 = self.resnet.layer2(e2) # (B, 128, 256, 256)
        e4 = self.resnet.layer3(e3) # (B, 256, 128, 128)
        e5 = self.resnet.layer4(e4) # (B, 512, 64, 64)

        aux_loss = 0.0
        if self.pretrained:
            m, aux_loss = self.moe(e5)
            o1, o2, o3, o4 = self.docker1(m), self.docker2(m), self.docker3(m), self.docker4(m)
        else:
            o1, o2, o3, o4 = e1, e2, e3, e4

        # Decoder with Deep Supervision
        d5 = self.up5(e5, o4) # 256, 128x128
        d4 = self.up4(d5, o3) # 128, 256x256
        d3 = self.up3(d4, o2) # 64, 512x512
        d1 = self.up2(d3, o1) # 32, 1024x1024

        out = self.pred(d1)
        #out = self.sigmoid(out)

        if self.training:
            # 🌟 提取辅助预测并插值到原图尺寸
            p5 = F.interpolate(self.aux_head5(d5), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            p4 = F.interpolate(self.aux_head4(d4), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            p3 = F.interpolate(self.aux_head3(d3), size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
            
            return out, aux_loss, p5, p4, p3
            
        return out