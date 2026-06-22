import os
import json
import torch
import argparse
import datetime  
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入组件
from dataloaders.road_dataset import RoadDataset
# 🌟 直接导入 UTANet
from models.baselines.UTANet import UTANet 

class Evaluator:
    """
    全局指标累加器 (Dataset-level Metrics)
    不算单张图的均值，而是把整个测试集的像素丢进一个大池子里算。
    分别累加整个测试集的 TP, FP, TN, FN，最后统一计算指标。
    """
    def __init__(self):
        self.TP = 0.0  
        self.FP = 0.0  
        self.FN = 0.0  

    def update(self, preds, targets):
        # 1. 严格二值化：确保预测概率图变成 0 和 1 的硬标签
        preds = (preds > 0.5).float()
        targets = targets.float()
        
        # 2. 利用张量点乘的特性高效计算混淆矩阵
        self.TP += (preds * targets).sum().item()
        self.FP += (preds * (1 - targets)).sum().item()
        self.FN += ((1 - preds) * targets).sum().item()

    def get_metrics(self):
        # 计算全局 Precision, Recall, F1, IoU
        precision = self.TP / (self.TP + self.FP + 1e-6)
        recall = self.TP / (self.TP + self.FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = self.TP / (self.TP + self.FP + self.FN + 1e-6)
        return precision, recall, f1, iou

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-w', '--weight', type=str, required=True, help='Path to best_model.pth')
    # 🌟 新增：指定你要测试的是第几阶段的权重
    parser.add_argument('--phase', type=int, choices=[1, 2], default=2, help='Model phase to test (1: Base, 2: TA-MoSC)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    dataset_name = config['dataset']['name']
    img_size = config['dataset'].get('input_size', 512)
    n_classes = config['dataset'].get('num_classes', 1) 

    # --- [1] 数据集加载 ---
    if dataset_name == 'DRIVE':
        from dataloaders.drive_dataset import DRIVEDataset
        test_dataset = DRIVEDataset(config['dataset']['root_path'], dataset_name, mode='test', img_size=img_size)
    else:
        test_dataset = RoadDataset(config['dataset']['root_path'], dataset_name, mode='test', img_size=img_size)
        
    n_workers = config['dataset'].get('num_workers', 4)
    test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, num_workers=n_workers, pin_memory=True)

    # --- [2] 初始化 UTANet ---
    print(f"🔧 初始化 UTANet (Phase {args.phase})...")
    is_pretrained = (args.phase == 2)
    model = UTANet(pretrained=is_pretrained, n_classes=n_classes, img_size=img_size).cuda()

    # --- [3] 加载权重 ---
    print(f"📥 正在加载权重: {args.weight}")
    
    checkpoint = torch.load(args.weight, map_location='cuda', weights_only=False)
    # 兼容直接保存 state_dict 或保存 dict 的形式
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ 权重加载成功！")

    evaluator = Evaluator()

    # --- [4] 开始测试 ---
    with torch.no_grad():
        val_loader_tqdm = tqdm(test_loader, desc="Testing", leave=False)
        for batch_data in val_loader_tqdm:
            # 兼容你的 Dataset 输出
            if len(batch_data) == 3:
                imgs, masks, _ = batch_data
            else:
                imgs, masks = batch_data
                
            imgs, masks = imgs.cuda(), masks.cuda()
            
            # 模型推理
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
            
            # 🌟 核心修改 1：解析 UTANet 的 tuple 返回值，舍弃 aux_loss
            if isinstance(outputs, tuple):
                preds = outputs[0]
            else:
                preds = outputs

            # 🌟 核心修改 2：UTANet 内部已包含 Sigmoid，所以不需要像原版那样套 torch.sigmoid(preds) 了
            # 直接将预测概率传给 evaluator 进行二值化和累加即可
            evaluator.update(preds, masks)

    # --- [5] 计算指标并输出 ---
    precision, recall, f1, iou = evaluator.get_metrics()

    save_dir = os.path.dirname(args.weight)
    result_file_path = os.path.join(save_dir, f'test_phase{args.phase}_results.log')
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result_msg = (
        f"\n==================================================\n"
        f" ⏱ 测试时间: {now_str}\n"
        f" ⚙️ 测试阶段: UTANet Phase {args.phase}\n"
        f" 📂 权重路径: {args.weight}\n"
        f" 📊 数据集:   {dataset_name}\n"
        f"--------------------------------------------------\n"
        f" 精确率 Precision: {precision * 100:.2f}%\n"
        f" 召回率 Recall:    {recall * 100:.2f}%\n"
        f" F1-Score:       {f1 * 100:.2f}%\n"
        f" 交并比 IoU:       {iou * 100:.2f}%\n"
        f"==================================================\n"
    )

    print(result_msg)

    # 将结果追加写入到对应的日志文件中
    with open(result_file_path, 'a') as f:
        f.write(result_msg)
    
    print(f"📝 测试结果已保存至: {result_file_path}")

if __name__ == '__main__':
    main()