##网络里没有进行了sigmoid的
# #在Loss里进行sigmoid()
#先检查有没有sigmoid ,models/ 文件夹下时，拉到那个代码的最底部（通常在 forward 函数的最后几行）或者def __init__(self):里，找有没有Sigmoid 
#网络的最后一层没有加 nn.Sigmoid() 函数，在Loss里再进行sigmoid()
    # def forward(self, x):
    #     # ... 前面的网络层 ...
    #     out = self.final_conv(x)
    #     out = torch.sigmoid(out)  # 👈 找到罪魁祸首！
    #     return out
        
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs) # 把输出变成 0~1 的概率
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # PyTorch 自带的自动结合了 Sigmoid 的 BCE 损失
        self.bce = nn.BCEWithLogitsLoss() 
        self.dice = DiceLoss()

    def forward(self, inputs, targets):
        # 核心在这里：把两者的结果相加，权重都是 1
        return self.bce(inputs, targets) + self.dice(inputs, targets)

# 在 train.py 里如何调用：
# if config['training']['loss_type'] == 'BCE_Dice':
#     criterion = BCEDiceLoss()




##网络里已经进行了sigmoid的
# import torch
# import torch.nn as nn

# class DiceLoss(nn.Module):
#     def forward(self, inputs, targets, smooth=1):
#         # ⚠️ 这里把 inputs = torch.sigmoid(inputs) 删掉了！
#         # 因为传进来的 inputs 已经是模型经过 Sigmoid 输出的 0~1 的概率了
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#         return 1 - dice

# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # ⚠️ 核心修改：将 BCEWithLogitsLoss 替换为普通的 BCELoss！
#         # BCELoss 要求输入必须是严格在 0 到 1 之间的浮点数（即自带 Sigmoid 模型的输出）
#         self.bce = nn.BCELoss() 
#         self.dice = DiceLoss()

#     def forward(self, inputs, targets):
#         # 把两者的结果相加，权重都是 1
#         return self.bce(inputs, targets) + self.dice(inputs, targets)