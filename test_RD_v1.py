# -*- coding: utf-8 -*-
"""Test script for RD_v1.

用法示例：
    python test_RD_v1.py --config saved_runs/CHN/RD_v1_xxx/config.json --weights saved_runs/CHN/RD_v1_xxx/best_model.pth --thresholds 0.35,0.4,0.45,0.5
"""

import os
import json
import argparse
import datetime
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.road_dataset import RoadDataset
from dataloaders.drive_dataset import DRIVEDataset
from models import get_model


def parse_thresholds(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    new_state = OrderedDict()
    changed = False
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
            changed = True
        else:
            new_state[k] = v
    return new_state if changed else state_dict


def load_checkpoint_state(weight_path):
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_type = "raw_state_dict"
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        state_type = "model_state_dict"
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        state_type = "state_dict"
    else:
        state = ckpt
    info = {}
    if isinstance(ckpt, dict):
        for key in ["epoch", "metrics"]:
            if key in ckpt:
                info[key] = ckpt[key]
    return strip_module_prefix(state), state_type, info


def build_dataset(config, mode="test"):
    name = config["dataset"]["name"]
    root = config["dataset"]["root_path"]
    img_size = int(config["dataset"].get("input_size", 1024))
    if name.lower() == "drive":
        return DRIVEDataset(root, name, mode=mode, img_size=img_size)
    return RoadDataset(root, name, mode=mode, img_size=img_size)


def extract_logits(outputs):
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if isinstance(outputs, dict):
        for k in ("final_logits", "logits", "out"):
            if k in outputs:
                return outputs[k]
        for v in outputs.values():
            if torch.is_tensor(v) and v.dim() == 4 and v.shape[1] == 1:
                return v
    raise RuntimeError(f"无法从模型输出中提取 logits: {type(outputs)}")


def apply_tta(model, imgs, amp=True):
    with torch.amp.autocast("cuda", enabled=amp):
        logits = extract_logits(model(imgs))
        logits_h = torch.flip(extract_logits(model(torch.flip(imgs, dims=[3]))), dims=[3])
        logits_v = torch.flip(extract_logits(model(torch.flip(imgs, dims=[2]))), dims=[2])
    return (logits + logits_h + logits_v) / 3.0


class DatasetLevelEvaluator:
    def __init__(self, thresholds):
        self.thresholds = [float(t) for t in thresholds]
        self.stats = {
            t: {"TP": 0.0, "FP": 0.0, "FN": 0.0, "pred_fg": 0.0, "gt_fg": 0.0, "pixels": 0.0, "images": 0}
            for t in self.thresholds
        }

    @torch.no_grad()
    def update(self, probs, targets):
        targets = targets.float()
        pixels = targets.numel()
        bsz = targets.shape[0]
        gt_fg = targets.sum().item()
        for t in self.thresholds:
            preds = (probs > t).float()
            s = self.stats[t]
            s["TP"] += (preds * targets).sum().item()
            s["FP"] += (preds * (1.0 - targets)).sum().item()
            s["FN"] += ((1.0 - preds) * targets).sum().item()
            s["pred_fg"] += preds.sum().item()
            s["gt_fg"] += gt_fg
            s["pixels"] += pixels
            s["images"] += bsz

    def get_metrics(self):
        out = {}
        for t, s in self.stats.items():
            tp, fp, fn = s["TP"], s["FP"], s["FN"]
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
            iou = tp / (tp + fp + fn + 1e-6)
            out[t] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "iou": iou,
                "pred_fg_ratio": s["pred_fg"] / (s["pixels"] + 1e-6),
                "gt_fg_ratio": s["gt_fg"] / (s["pixels"] + 1e-6),
                "images": s["images"],
            }
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--thresholds", type=str, default="0.35,0.4,0.45,0.5")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--tta", action="store_true", help="horizontal+vertical flip TTA")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    img_size = int(config["dataset"].get("input_size", 1024))
    model = get_model(config["model"], img_size=img_size).cuda()
    state, state_type, ckpt_info = load_checkpoint_state(args.weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    dataset = build_dataset(config, mode=args.split)
    num_workers = args.num_workers if args.num_workers is not None else int(config["dataset"].get("num_workers", 8))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    thresholds = parse_thresholds(args.thresholds)
    evaluator = DatasetLevelEvaluator(thresholds)

    with torch.no_grad():
        for batch_data in tqdm(loader, desc=f"Testing RD_v1 [{args.split}]", leave=False):
            if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
                imgs, masks = batch_data[0], batch_data[1]
            else:
                raise RuntimeError("无法解析 batch")
            imgs = imgs.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)
            if args.tta:
                logits = apply_tta(model, imgs, amp=args.amp)
            else:
                with torch.amp.autocast("cuda", enabled=args.amp):
                    logits = extract_logits(model(imgs))
            probs = torch.sigmoid(logits)
            evaluator.update(probs, masks)

    metrics = evaluator.get_metrics()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n==================================================")
    print(f" ⏱测试时间: {now}")
    print(f" 权重路径: {args.weights}")
    print(f" 加载字段: {state_type}")
    print(f" 数据集:   {config['dataset']['name']}")
    print(f" Split:    {args.split}")
    print(f" 模型:     {config['model']['name']}")
    print(f" Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    if ckpt_info:
        print(f" Checkpoint info: {ckpt_info}")
    print("--------------------------------------------------")
    best_t, best_m = None, None
    for t, m in metrics.items():
        print(
            f" threshold={t:.3f} | IoU={m['iou']*100:.2f}% | "
            f"Precision={m['precision']*100:.2f}% | Recall={m['recall']*100:.2f}% | F1={m['f1']*100:.2f}% | "
            f"pred_fg={m['pred_fg_ratio']*100:.2f}% gt_fg={m['gt_fg_ratio']*100:.2f}%"
        )
        if best_m is None or m["iou"] > best_m["iou"]:
            best_t, best_m = t, m
    print("--------------------------------------------------")
    print(
        f" 最优阈值: {best_t:.3f} | IoU: {best_m['iou']*100:.2f}% | "
        f"Precision: {best_m['precision']*100:.2f}% | Recall: {best_m['recall']*100:.2f}% | F1: {best_m['f1']*100:.2f}%"
    )
    print("==================================================")


if __name__ == "__main__":
    main()
