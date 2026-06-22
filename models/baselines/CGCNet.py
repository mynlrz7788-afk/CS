import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

class DecoderBlock(nn.Module):
    # （此部分保持原样，省略细节）
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class CompactGlobalContextawareBlock(nn.Module):
    # （此部分保持原样）
    def __init__(self, in_channels, size=(64, 64)):
        super().__init__()
        self.in_channels = in_channels
        self.inter_channel = self.in_channels // 2
        self.conv_g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pooling_size = 2
        self.token_len = self.pooling_size * self.pooling_size
        self.to_qk = nn.Linear(self.in_channels, 2 * self.inter_channel, bias=False)
        self.conv_a = nn.Conv2d(in_channels, self.token_len, kernel_size=1, padding=0, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels // 16, self.in_channels, bias=False),
            nn.Sigmoid()
        )
        self.with_pos = True
        if self.with_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, 4, in_channels))
        self.with_pos_2 = True
        if self.with_pos_2:
            self.pos_embedding_2 = nn.Parameter(torch.randn(1, self.inter_channel, size[0], size[1]))

    def compact_representation(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()
        channel_attention = self.avg_pool(x).view(b, c)
        channel_attention = self.fc(channel_attention).view(b, c, 1)
        x = x * channel_attention
        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)
        return tokens

    def forward(self, x):
        b, c, h, w = x.size()
        x_clone = x
        x = self.compact_representation(x)
        if self.with_pos:
            x = x + self.pos_embedding
        _, n, _ = x.size()
        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = qk[0].reshape(b, -1, n), qk[1]
        if self.with_pos_2:
            x_g = (self.conv_g(x_clone) + self.pos_embedding_2).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()
        else:
            x_g = self.conv_g(x_clone).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()
        mul_theta_phi = torch.matmul(q, k)
        mul_theta_phi = self.softmax(mul_theta_phi)
        mul_theta_phi_g = torch.matmul(x_g, mul_theta_phi)
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().reshape(b, self.inter_channel, h, w)
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x_clone
        return out

class CGCNet(nn.Module):
    # 🌟 修改 1：添加 img_size 参数
    def __init__(self, out_channels=1, img_size=1024):
        super(CGCNet, self).__init__()

        # 🌟 修改 2：使用 torchvision 自带的预训练权重，避免本地找不到 .pth 报错
        try:
            # 兼容高版本 torchvision
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            resnet = models.resnet34(weights=weights)
        except AttributeError:
            # 兼容低版本 torchvision
            resnet = models.resnet34(pretrained=True)
            
        self.resnet = resnet
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()  # 🌟 修改 3：注释掉，不要Sigmoid，由外层Loss控制

        # encoder
        self.first_conv = resnet.conv1
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.first_maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        # decoder
        dims_decoder = [512, 256, 128, 64]
        self.decoder3 = DecoderBlock(dims_decoder[1], dims_decoder[2])
        self.decoder2 = DecoderBlock(dims_decoder[2], dims_decoder[3])
        self.decoder1 = DecoderBlock(dims_decoder[3], dims_decoder[3])
        self.final_deconv1 = nn.ConvTranspose2d(dims_decoder[3], 32, 4, 2, 1)
        self.final_relu1 = nonlinearity
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nonlinearity
        self.final_conv3 = nn.Conv2d(32, out_channels, 3, padding=1)

        # Dimensionality Reduction
        self.reduction_conv = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        # Dimensionality Increase
        self.increase_conv = nn.Conv2d(32, 256, kernel_size=3, padding=1)

        # 🌟 修改 4：动态计算 CGCB 的 size 尺寸。
        # 原作者基于 ResNet 提取了 layer3，layer3 的下采样率是 16。
        # 所以如果输入是 1024，到达此处时尺寸为 1024/16 = 64。
        feat_size = img_size // 16 
        self.cgcb = CompactGlobalContextawareBlock(in_channels=32, size=(feat_size, feat_size))

    def compact_global_contextaware_block_reduction_increase(self, x1):
        x1 = self.reduction_conv(x1)
        x1 = self.cgcb(x1)
        x1 = self.increase_conv(x1)
        return x1

    def forward_features(self, x):
        skip_list = []
        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_maxpool(x)
        e1_l = self.encoder1(x)
        skip_list.append(e1_l)
        e2_l = self.encoder2(e1_l)
        skip_list.append(e2_l)
        e3_l = self.encoder3(e2_l)
        skip_list.append(e3_l)
        return e3_l, skip_list

    def up_features(self, x, skip_list):
        x = x + self.compact_global_contextaware_block_reduction_increase(x)
        d3 = self.decoder3(x) + skip_list[1]
        d2 = self.decoder2(d3) + skip_list[0]
        d1 = self.decoder1(d2)
        out = self.final_deconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)
        return out

    def forward(self, x1):
        x1, skip_list = self.forward_features(x1)
        x = self.up_features(x1, skip_list)
        return x  # 🌟 修改 5：直接返回 x，不再使用 return self.sigmoid(x)