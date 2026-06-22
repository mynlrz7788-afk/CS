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
    统计整个测试集的 TP / FP / FN，最后统一计算 Precision / Recall / F1 / IoU。
    """
    def __init__(self):
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0

    def update(self, preds, targets):
        preds = (preds > 0.5).float()
        targets = targets.float()

        self.TP += (preds * targets).sum().item()
        self.FP += (preds * (1 - targets)).sum().item()
        self.FN += ((1 - preds) * targets).sum().item()

    def get_metrics(self):
        precision = self.TP / (self.TP + self.FP + 1e-6)
        recall = self.TP / (self.TP + self.FN + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = self.TP / (self.TP + self.FP + self.FN + 1e-6)
        return precision, recall, f1, iou


def extract_main_pred(outputs):
    """
    统一提取主输出：
    - dict: outputs['main']
    - tuple/list: outputs[0]
    - tensor: outputs
    """
    if isinstance(outputs, dict):
        return outputs['main']
    elif isinstance(outputs, (tuple, list)):
        return outputs[0]
    else:
        return outputs


def build_val_loader(config):
    """
    根据配置创建验证/测试集 DataLoader
    """
    dataset_name = config['dataset']['name']
    img_size = config['dataset'].get('input_size', 1024)
    batch_size = config['dataset'].get('test_batch_size', 4)
    num_workers = config['dataset'].get('num_workers', 8)

    if dataset_name == 'DRIVE':
        from dataloaders.drive_dataset import DRIVEDataset
        val_dataset = DRIVEDataset(
            config['dataset']['root_path'],
            dataset_name,
            mode='val',
            img_size=img_size
        )
    else:
        val_dataset = RoadDataset(
            config['dataset']['root_path'],
            dataset_name,
            mode='val',
            img_size=img_size
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return val_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('-w', '--weight', type=str, required=True, help='权重文件路径（best_model.pth）')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    img_size = config['dataset'].get('input_size', 1024)
    val_loader = build_val_loader(config)

    model = get_model(config['model'], img_size=img_size).cuda()

    checkpoint = torch.load(args.weight, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    model.eval()
    evaluator = Evaluator()

    print(f"开始评估模型，权重路径: {args.weight}")

    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Testing Metrics"):
            if len(batch_data) == 3:
                imgs, masks, _ = batch_data
            else:
                imgs, masks = batch_data

            imgs, masks = imgs.cuda(), masks.cuda()

            outputs = model(imgs)
            preds = extract_main_pred(outputs)
            preds = torch.sigmoid(preds)

            evaluator.update(preds, masks)

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
        f" F1-Score:         {f1 * 100:.2f}%\n"
        f" 交并比 IoU:       {iou * 100:.2f}%\n"
        f"==================================================\n\n"
    )

    print(result_msg)

    with open(result_file_path, 'a') as f:
        f.write(result_msg)

    print(f"已成功保存至: {result_file_path}")


if __name__ == '__main__':
    main()
