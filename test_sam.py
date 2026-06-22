import os
import json
import torch
import argparse
import datetime  
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.road_dataset import RoadDataset
from models import get_model

class Evaluator:
    """
    全局指标累加器 (Dataset-level Metrics)
    不算单张图的均值，而是把整个测试集的像素丢进一个大池子里算。
    分别累加整个测试集的 TP, FP, TN, FN，最后统一计算指标。
    避免“图像中道路太少导致单张图 IoU 为 0，拉低整体均值”
    防止因为某些图片道路像素极少（分母极小）而导致指标剧烈波动。
    """
    def __init__(self):
        self.TP = 0.0  # True Positive: 预测是路，实际也是路（找对了）
        self.FP = 0.0  # False Positive: 预测是路，实际是背景（报假警）
        self.FN = 0.0  # False Negative: 预测是背景，实际是路（漏掉了）

    def update(self, preds, targets):
        # 1. 严格二值化：确保预测概率图变成 0 和 1 的硬标签
        preds = (preds > 0.5).float()
        targets = targets.float()
        
        # 2. 利用张量点乘的特性高效计算混淆矩阵
        # (preds * targets) 只有 1*1=1 时才生效，即两者都为路
        self.TP += (preds * targets).sum().item()
        # preds 为 1 且 targets 为 0 (取反后为1) 的部分，即误报
        self.FP += (preds * (1 - targets)).sum().item()
        # preds 为 0 (取反后为1) 且 targets 为 1 的部分，即漏报
        self.FN += ((1 - preds) * targets).sum().item()

    def get_metrics(self):
        # 加 1e-6 是深度学习里的基操，防止分母为 0 导致抛出 ZeroDivisionError 甚至 NaN
        precision = self.TP / (self.TP + self.FP + 1e-6)
        recall = self.TP / (self.TP + self.FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = self.TP / (self.TP + self.FP + self.FN + 1e-6)
        return precision, recall, f1, iou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='配置文件的路径')
    parser.add_argument('-w', '--weight', type=str, required=True, help='best_model.pth 的路径')
    args = parser.parse_args()

    # 读取实验配置文件
    with open(args.config, 'r') as f:
        config = json.load(f)

    # 🌟 智能分流：如果是 DRIVE，就调用专门的 DRIVEDataset (自动处理边缘黑色填充)
    dataset_name = config['dataset']['name']
    img_size = config['dataset'].get('input_size', 512)
    
    if dataset_name == 'DRIVE':
        from dataloaders.drive_dataset import DRIVEDataset
        # 自动把 config 里的 root_path 和 DRIVE 名字拼起来
        drive_full_path = os.path.join(config['dataset']['root_path'], dataset_name)
        # 传 mode='val' 时，DRIVEDataset 内部会自动去读 test 文件夹
        val_dataset = DRIVEDataset(drive_full_path, mode='val', img_size=img_size)
    else:
        # 其他正常的遥感道路数据集
        val_dataset = RoadDataset(config['dataset']['root_path'], dataset_name, mode='val', img_size=img_size)

    # ⚠️ 测试时 batch_size 必须固定为 1，确保每张图被原汁原味地评估
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    
    
    # 实例化模型（与训练保持一致，传入 img_size）
    model = get_model(config['model'], img_size=img_size).cuda()
    checkpoint = torch.load(args.weight)
    
    # 兼容性加载：strict=False 是神器！自动忽略模型测试 FLOPs 时留下的多余变量（如 total_ops）
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False) 
        
    # 【关键】切换为测试模式，冻结 BatchNorm 和 Dropout 的状态
    model.eval()
    evaluator = Evaluator()

    print(f"开始评估模型，权重路径: {args.weight}")
    
    # 【关键】关闭梯度计算图，极大节省显存，让推理速度起飞
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Testing Metrics"):
            # 🌟 智能解包：兼容 Dataset 返回 3 个变量的情况
            if len(batch_data) == 3:
                imgs, masks, _ = batch_data  # 纯算指标不需要名字，用 _ 丢弃
            else:
                imgs, masks = batch_data
                
            imgs, masks = imgs.cuda(), masks.cuda()
            
            # 模型推理
            preds = model(imgs)

            # 🌟 核心修改 3：测试时过滤掉中间层结果
            if isinstance(preds, (tuple, list)):
                preds = preds[0]

            # 🌟 Bug 修复：将原始 Logits 转换为 0~1 的真实概率！
            preds = torch.sigmoid(preds)
            
            # 把这一个批次的预测结果喂给评价器去累加
            evaluator.update(preds, masks)

    # 循环结束，计算最终总成绩
    precision, recall, f1, iou = evaluator.get_metrics()

    save_dir = os.path.dirname(args.weight)
    result_file_path = os.path.join(save_dir, 'test_results.log')
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result_msg = (
        f"\n==================================================\n"
        f" ⏱测试时间: {now_str}\n"
        f" 权重路径: {args.weight}\n"
        f" 数据集:   {config['dataset']['name']}\n"
        f" 模型:     {config['model']['name']}\n"
        f"--------------------------------------------------\n"
        f" 精确率 Precision: {precision * 100:.2f}%\n"
        f" 召回率 Recall:    {recall * 100:.2f}%\n"
        f" F1-Score:       {f1 * 100:.2f}%\n"
        f" 交并比 IoU:       {iou * 100:.2f}%\n"
        f"==================================================\n\n"
    )

    # 1. 打印到终端
    print(result_msg)
    
    # 2. 追加写入模式 'a' 保存到文件。
    # (用 'a' 模式的好处是：如果用同一份权重在不同测试集上测了多次，它会一行行往下加，不会覆盖之前的记录)
    with open(result_file_path, 'a') as f:
        f.write(result_msg)
        
    print(f"已成功保存至: {result_file_path}")

if __name__ == '__main__':
    main()