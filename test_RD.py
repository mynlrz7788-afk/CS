# -*- coding: utf-8 -*-
"""Generic test / evaluation script for RD-family segmentation models.

适配对象：RD_v1 / RD_v2 / 后续 RD_v3...，也兼容普通 logits 输出模型。
功能：
- 自动从 checkpoint 中提取 model_state_dict / state_dict；
- 自动跳过 size mismatch，方便 RD 变体之间做微调测试；
- 支持 val/test/train split；
- 支持 threshold sweep（阈值扫描）；
- 支持 flip TTA（测试时增强）；
- 保存 metrics json/csv，可选择保存预测图。
"""

import os
import json
import argparse
import datetime
from contextlib import nullcontext
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.road_dataset import RoadDataset
from dataloaders.drive_dataset import DRIVEDataset
from models import get_model


def parse_thresholds(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return nullcontext()


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state = OrderedDict()
    changed = False
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
            changed = True
        else:
            new_state[k] = v
    return new_state if changed else state_dict


def extract_checkpoint_state(ckpt_obj: Any) -> Tuple[Dict[str, torch.Tensor], str, Dict[str, Any]]:
    info = {}
    if isinstance(ckpt_obj, dict):
        for meta in ("epoch", "metrics", "config"):
            if meta in ckpt_obj:
                info[meta] = ckpt_obj[meta]
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return strip_module_prefix(ckpt_obj[key]), key, info
    if isinstance(ckpt_obj, dict):
        return strip_module_prefix(ckpt_obj), "raw_state_dict", info
    raise RuntimeError("checkpoint 格式无法解析")


def load_checkpoint_state(weight_path: str):
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    return extract_checkpoint_state(ckpt)


def load_model_weights(model, weight_path: str, strict: bool = False, skip_mismatch: bool = True):
    state, state_type, info = load_checkpoint_state(weight_path)
    model_state = model.state_dict()
    skipped = []
    if skip_mismatch:
        filtered = OrderedDict()
        for k, v in state.items():
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
                filtered[k] = v
            else:
                skipped.append(k)
        state = filtered
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return {
        "state_type": state_type,
        "info": info,
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped": skipped,
        "loaded_keys": len(state),
    }


def build_dataset(config: Dict[str, Any], mode: str):
    ds_cfg = config["dataset"]
    name = ds_cfg["name"]
    root = ds_cfg["root_path"]
    img_size = int(ds_cfg.get("input_size", 1024))
    if name.lower() == "drive":
        return DRIVEDataset(root, name, mode=mode, img_size=img_size)
    return RoadDataset(root, name, mode=mode, img_size=img_size)


def unwrap_batch(batch_data):
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
        imgs, masks = batch_data[0], batch_data[1]
        extra = batch_data[2:] if len(batch_data) > 2 else None
        return imgs, masks, extra
    if isinstance(batch_data, dict):
        img = batch_data.get("image") or batch_data.get("img") or batch_data.get("images")
        mask = batch_data.get("mask") or batch_data.get("label") or batch_data.get("masks")
        if img is not None and mask is not None:
            return img, mask, batch_data
    raise RuntimeError(f"无法解析 batch 数据结构: {type(batch_data)}")


def extract_logits(outputs):
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if isinstance(outputs, dict):
        for k in ("final_logits", "logits", "out", "pred", "prediction"):
            if k in outputs and torch.is_tensor(outputs[k]):
                return outputs[k]
        for v in outputs.values():
            if torch.is_tensor(v) and v.dim() == 4 and v.shape[1] == 1:
                return v
    raise RuntimeError(f"无法从模型输出中提取 logits: {type(outputs)}")


def set_model_return_aux(model, enabled: bool):
    if hasattr(model, "return_aux"):
        try:
            setattr(model, "return_aux", bool(enabled))
        except Exception:
            pass


def model_forward(model, imgs):
    return model(imgs)


@torch.no_grad()
def apply_tta(model, imgs, device, amp=True, mode="h,v"):
    modes = [m.strip().lower() for m in str(mode).split(",") if m.strip()]
    logits_sum = None
    count = 0

    def add_logits(x, reverse_dims=None):
        nonlocal logits_sum, count
        with amp_context(device, amp):
            y = extract_logits(model_forward(model, x))
        if reverse_dims:
            y = torch.flip(y, dims=reverse_dims)
        logits_sum = y if logits_sum is None else logits_sum + y
        count += 1

    add_logits(imgs, None)
    if "h" in modes or "horizontal" in modes:
        add_logits(torch.flip(imgs, dims=[3]), [3])
    if "v" in modes or "vertical" in modes:
        add_logits(torch.flip(imgs, dims=[2]), [2])
    if "hv" in modes or "both" in modes:
        add_logits(torch.flip(imgs, dims=[2, 3]), [2, 3])
    return logits_sum / max(count, 1)


class DatasetLevelEvaluator:
    def __init__(self, thresholds: List[float]):
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


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def get_model_name(config):
    m = config.get("model", {})
    return m.get("name") or m.get("model_name") or config.get("model_name") or "RD"


def save_prediction_maps(probs: torch.Tensor, out_dir: str, start_index: int, threshold: float):
    os.makedirs(out_dir, exist_ok=True)
    probs_np = probs.detach().float().cpu().numpy()
    for i in range(probs_np.shape[0]):
        p = probs_np[i, 0]
        prob_img = (np.clip(p, 0, 1) * 255).astype(np.uint8)
        mask_img = ((p > threshold).astype(np.uint8) * 255)
        Image.fromarray(prob_img).save(os.path.join(out_dir, f"{start_index+i:06d}_prob.png"))
        Image.fromarray(mask_img).save(os.path.join(out_dir, f"{start_index+i:06d}_mask.png"))


def write_metrics_csv(path: str, metrics: Dict[float, Dict[str, float]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["threshold", "iou", "precision", "recall", "f1", "pred_fg_ratio", "gt_fg_ratio", "images"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t, m in metrics.items():
            row = {"threshold": t}
            row.update(m)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--thresholds", type=str, default="0.35,0.4,0.45,0.5")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tta", action="store_true", help="enable flip TTA")
    parser.add_argument("--tta-mode", type=str, default="h,v", help="h,v,hv")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--no-skip-mismatch", action="store_true")
    parser.add_argument("--save-dir", type=str, default="", help="directory to save metrics/predictions")
    parser.add_argument("--save-pred", action="store_true", help="save probability and binary masks")
    parser.add_argument("--save-threshold", type=float, default=None, help="threshold for saved binary masks; default best threshold after eval is unavailable online, so uses 0.5")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = int(config["dataset"].get("input_size", 1024))
    model = get_model(config["model"], img_size=img_size).to(device)
    set_model_return_aux(model, False)
    load_info = load_model_weights(
        model,
        args.weights,
        strict=bool(args.strict_load),
        skip_mismatch=not bool(args.no_skip_mismatch),
    )
    model.eval()

    dataset = build_dataset(config, mode=args.split)
    batch_size = args.batch_size if args.batch_size is not None else int(config["dataset"].get("val_batch_size", config["dataset"].get("batch_size", 1)))
    num_workers = args.num_workers if args.num_workers is not None else int(config["dataset"].get("num_workers", 8))
    loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=False, num_workers=num_workers, pin_memory=True)

    thresholds = parse_thresholds(args.thresholds)
    evaluator = DatasetLevelEvaluator(thresholds)
    amp_enabled = bool(args.amp) and not bool(args.no_amp) and device.type == "cuda"

    save_dir = args.save_dir
    if not save_dir:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join("test_results", config["dataset"]["name"], f"{get_model_name(config)}_{args.split}_{ts}")
    os.makedirs(save_dir, exist_ok=True)

    save_index = 0
    save_thr = float(args.save_threshold) if args.save_threshold is not None else 0.5
    with torch.no_grad():
        for batch_data in tqdm(loader, desc=f"Testing {get_model_name(config)} [{args.split}]", leave=False):
            imgs, masks, _ = unwrap_batch(batch_data)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if args.tta:
                logits = apply_tta(model, imgs, device=device, amp=amp_enabled, mode=args.tta_mode)
            else:
                with amp_context(device, amp_enabled):
                    logits = extract_logits(model_forward(model, imgs))
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)
            evaluator.update(probs, masks)
            if args.save_pred:
                save_prediction_maps(probs, os.path.join(save_dir, "predictions"), save_index, threshold=save_thr)
            save_index += probs.shape[0]

    metrics = evaluator.get_metrics()
    best_t, best_m = None, None
    for t, m in metrics.items():
        if best_m is None or m["iou"] > best_m["iou"]:
            best_t, best_m = t, m

    total, trainable, frozen = count_params(model)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n==================================================")
    print(f" ⏱测试时间: {now}")
    print(f" 权重路径: {args.weights}")
    print(f" 加载字段: {load_info['state_type']} | loaded={load_info['loaded_keys']} missing={len(load_info['missing'])} unexpected={len(load_info['unexpected'])} skipped={len(load_info['skipped'])}")
    print(f" 数据集:   {config['dataset']['name']}")
    print(f" Split:    {args.split}")
    print(f" 模型:     {get_model_name(config)}")
    print(f" Params:   total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M frozen={frozen/1e6:.2f}M")
    print(f" TTA:      {args.tta} ({args.tta_mode}) | AMP: {amp_enabled}")
    if load_info.get("info"):
        print(f" Checkpoint info: {load_info['info']}")
    print("--------------------------------------------------")
    for t, m in metrics.items():
        print(
            f" threshold={t:.3f} | IoU={m['iou']*100:.2f}% | "
            f"Precision={m['precision']*100:.2f}% | Recall={m['recall']*100:.2f}% | F1={m['f1']*100:.2f}% | "
            f"pred_fg={m['pred_fg_ratio']*100:.2f}% gt_fg={m['gt_fg_ratio']*100:.2f}%"
        )
    print("--------------------------------------------------")
    print(
        f" 最优阈值: {best_t:.3f} | IoU: {best_m['iou']*100:.2f}% | "
        f"Precision: {best_m['precision']*100:.2f}% | Recall: {best_m['recall']*100:.2f}% | F1: {best_m['f1']*100:.2f}%"
    )
    print(f" 结果保存: {save_dir}")
    print("==================================================")

    result = {
        "time": now,
        "config": args.config,
        "weights": args.weights,
        "split": args.split,
        "model": get_model_name(config),
        "dataset": config["dataset"]["name"],
        "load_info": {k: v for k, v in load_info.items() if k not in ("missing", "unexpected", "skipped")},
        "missing_count": len(load_info["missing"]),
        "unexpected_count": len(load_info["unexpected"]),
        "skipped_count": len(load_info["skipped"]),
        "params": {"total": total, "trainable": trainable, "frozen": frozen},
        "tta": args.tta,
        "tta_mode": args.tta_mode,
        "amp": amp_enabled,
        "metrics": {str(k): v for k, v in metrics.items()},
        "best": {"threshold": best_t, **best_m},
    }
    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_metrics_csv(os.path.join(save_dir, "metrics.csv"), metrics)


if __name__ == "__main__":
    main()
