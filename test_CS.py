# -*- coding: utf-8 -*-
"""test_CS.py

Generic evaluation script for curvilinear-structure segmentation models in the CS project.

核心要求：
1) 与原 test.py 一样，读取 config + weight，构建 test/val/train split；
2) 结果保存到权重所在文件夹，不再另存到 test_results；
3) 兼容普通 Tensor 输出、tuple/list 输出、dict 输出模型；
4) 支持阈值扫描、AMP、flip TTA、可选保存预测图；
5) 指标描述统一为曲线结构分割，而不是道路分割。
"""

import os
try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass

import csv
import json
import torch
import argparse
import datetime
from contextlib import nullcontext
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloaders.road_dataset import RoadDataset
from models import get_model




def parse_input_size(input_size):
    """
    支持：
    - 512 -> (512, 512)
    - [320, 480] -> (320, 480)
    - "320,480" / "320x480" -> (320, 480)
    - {"height": 320, "width": 480} -> (320, 480)
    返回顺序统一为 (H, W)。
    """
    if isinstance(input_size, int):
        return int(input_size), int(input_size)

    if isinstance(input_size, str):
        s = input_size.lower().replace("x", ",").replace("*", ",")
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) == 1:
            v = int(parts[0])
            return v, v
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])

    if isinstance(input_size, (list, tuple)):
        if len(input_size) == 1:
            v = int(input_size[0])
            return v, v
        if len(input_size) == 2:
            return int(input_size[0]), int(input_size[1])

    if isinstance(input_size, dict):
        h = input_size.get("height", input_size.get("h", None))
        w = input_size.get("width", input_size.get("w", None))
        if h is not None and w is not None:
            return int(h), int(w)

    raise ValueError(f"Unsupported input_size format: {input_size}")


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
        for meta in ("epoch", "metrics", "config", "best_val_iou", "epochs_without_improvement"):
            if meta in ckpt_obj:
                info[meta] = ckpt_obj[meta]
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return strip_module_prefix(ckpt_obj[key]), key, info
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
    root = ds_cfg["root_path"]
    name = ds_cfg["name"]
    input_size = ds_cfg.get("input_size", 1024)
    return RoadDataset(root, name, mode=mode, img_size=input_size)


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
        for k in ("final_logits", "logits", "out", "pred", "prediction", "mask_logits"):
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
    """Dataset-level evaluator for curvilinear structure segmentation.

    不按单张图平均，而是累加整个 split 的 TP/FP/FN 后统一计算，
    对前景占比很低的血管、道路、裂缝等曲线结构更稳定。
    """
    def __init__(self, thresholds: List[float]):
        self.thresholds = [float(t) for t in thresholds]
        self.stats = {
            t: {"TP": 0.0, "FP": 0.0, "FN": 0.0, "pred_fg": 0.0, "gt_fg": 0.0, "pixels": 0.0, "images": 0}
            for t in self.thresholds
        }

    @torch.no_grad()
    def update(self, probs, targets):
        targets = targets.float()
        if probs.shape[-2:] != targets.shape[-2:]:
            probs = F.interpolate(probs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
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
        out = OrderedDict()
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
                "pred_structure_ratio": s["pred_fg"] / (s["pixels"] + 1e-6),
                "gt_structure_ratio": s["gt_fg"] / (s["pixels"] + 1e-6),
                "images": s["images"],
            }
        return out


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def get_model_name(config):
    m = config.get("model", {})
    return m.get("name") or m.get("model_name") or config.get("model_name") or "CSModel"


def save_prediction_maps(probs: torch.Tensor, out_dir: str, start_index: int, threshold: float):
    os.makedirs(out_dir, exist_ok=True)
    probs_np = probs.detach().float().cpu().numpy()
    for i in range(probs_np.shape[0]):
        p = probs_np[i, 0]
        prob_img = (np.clip(p, 0, 1) * 255).astype(np.uint8)
        mask_img = ((p > threshold).astype(np.uint8) * 255)
        Image.fromarray(prob_img).save(os.path.join(out_dir, f"{start_index + i:06d}_prob.png"))
        Image.fromarray(mask_img).save(os.path.join(out_dir, f"{start_index + i:06d}_mask.png"))


def write_metrics_csv(path: str, metrics: Dict[float, Dict[str, float]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["threshold", "iou", "precision", "recall", "f1", "pred_structure_ratio", "gt_structure_ratio", "images"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t, m in metrics.items():
            row = {"threshold": t}
            row.update(m)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="配置文件路径，通常是权重目录里的 config.json")
    parser.add_argument("-w", "--weight", "--weights", dest="weight", type=str, required=True, help="best_model.pth / latest_model.pth 路径")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--thresholds", type=str, default="0.35,0.4,0.45,0.5")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--tta", action="store_true", help="启用水平/垂直翻转 TTA")
    parser.add_argument("--tta-mode", type=str, default="h,v", help="h,v,hv")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--no-skip-mismatch", action="store_true")
    parser.add_argument("--save-pred", action="store_true", help="保存概率图和二值 mask，保存位置仍在权重目录下")
    parser.add_argument("--save-threshold", type=float, default=None, help="保存二值 mask 用的阈值；默认 0.5")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = config["dataset"].get("input_size", 1024)
    img_h, img_w = parse_input_size(input_size)
    if img_h % 16 != 0 or img_w % 16 != 0:
        raise ValueError(f"CS/RD/DINO 模型要求输入 H/W 能被 16 整除，当前 H={img_h}, W={img_w}")

    try:
        model = get_model(config["model"], img_size=input_size).to(device)
    except TypeError:
        # 兼容只接受 int img_size 的旧模型
        model = get_model(config["model"], img_size=max(img_h, img_w)).to(device)
    set_model_return_aux(model, False)
    load_info = load_model_weights(
        model,
        args.weight,
        strict=bool(args.strict_load),
        skip_mismatch=not bool(args.no_skip_mismatch),
    )
    model.eval()

    dataset = build_dataset(config, mode=args.split)
    batch_size = args.batch_size if args.batch_size is not None else int(config["dataset"].get("val_batch_size", config["dataset"].get("batch_size", 4)))
    num_workers = args.num_workers if args.num_workers is not None else int(config["dataset"].get("num_workers", 8))
    loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=False, num_workers=num_workers, pin_memory=True)

    thresholds = parse_thresholds(args.thresholds)
    evaluator = DatasetLevelEvaluator(thresholds)
    amp_enabled = bool(args.amp) and not bool(args.no_amp) and device.type == "cuda"

    # 关键：所有测试结果保存到权重所在文件夹。
    weight_dir = os.path.dirname(os.path.abspath(args.weight))
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_prefix = f"test_CS_{args.split}_{ts}"
    result_log_path = os.path.join(weight_dir, f"{result_prefix}.log")
    result_json_path = os.path.join(weight_dir, f"{result_prefix}_metrics.json")
    result_csv_path = os.path.join(weight_dir, f"{result_prefix}_metrics.csv")
    append_log_path = os.path.join(weight_dir, "test_results.log")

    pred_dir = os.path.join(weight_dir, f"{result_prefix}_predictions")
    save_thr = float(args.save_threshold) if args.save_threshold is not None else 0.5

    save_index = 0
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
                save_prediction_maps(probs, pred_dir, save_index, threshold=save_thr)
            save_index += probs.shape[0]

    metrics = evaluator.get_metrics()
    best_t, best_m = None, None
    for t, m in metrics.items():
        if best_m is None or m["iou"] > best_m["iou"]:
            best_t, best_m = t, m

    total, trainable, frozen = count_params(model)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_msg = "\n==================================================\n"
    result_msg += f" ⏱测试时间: {now}\n"
    result_msg += f" 权重路径: {args.weight}\n"
    result_msg += f" 加载字段: {load_info['state_type']} | loaded={load_info['loaded_keys']} missing={len(load_info['missing'])} unexpected={len(load_info['unexpected'])} skipped={len(load_info['skipped'])}\n"
    result_msg += f" 数据集:   {config['dataset']['name']}\n"
    result_msg += f" Split:    {args.split}\n"
    result_msg += f" 模型:     {get_model_name(config)}\n"
    result_msg += f" Params:   total={total / 1e6:.2f}M trainable={trainable / 1e6:.2f}M frozen={frozen / 1e6:.2f}M\n"
    result_msg += f" TTA:      {args.tta} ({args.tta_mode}) | AMP: {amp_enabled}\n"
    if load_info.get("info"):
        result_msg += f" Checkpoint info: {load_info['info']}\n"
    result_msg += "--------------------------------------------------\n"
    for t, m in metrics.items():
        result_msg += (
            f" threshold={t:.3f} | IoU={m['iou'] * 100:.2f}% | "
            f"Precision={m['precision'] * 100:.2f}% | Recall={m['recall'] * 100:.2f}% | F1={m['f1'] * 100:.2f}% | "
            f"pred_structure={m['pred_structure_ratio'] * 100:.2f}% gt_structure={m['gt_structure_ratio'] * 100:.2f}%\n"
        )
    result_msg += "--------------------------------------------------\n"
    result_msg += (
        f" 最优阈值: {best_t:.3f} | IoU: {best_m['iou'] * 100:.2f}% | "
        f"Precision: {best_m['precision'] * 100:.2f}% | Recall: {best_m['recall'] * 100:.2f}% | F1: {best_m['f1'] * 100:.2f}%\n"
    )
    result_msg += f" 结果保存目录: {weight_dir}\n"
    result_msg += f" 详细日志: {result_log_path}\n"
    if args.save_pred:
        result_msg += f" 预测图保存: {pred_dir}\n"
    result_msg += "==================================================\n\n"

    print(result_msg)
    with open(result_log_path, "w", encoding="utf-8") as f:
        f.write(result_msg)
    with open(append_log_path, "a", encoding="utf-8") as f:
        f.write(result_msg)

    result = {
        "time": now,
        "config": args.config,
        "weights": args.weight,
        "split": args.split,
        "model": get_model_name(config),
        "dataset": config["dataset"]["name"],
        "input_size": {"height": img_h, "width": img_w},
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
        "result_dir": weight_dir,
    }
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_metrics_csv(result_csv_path, metrics)

    print(f"已成功保存至权重同目录: {weight_dir}")


if __name__ == "__main__":
    main()
