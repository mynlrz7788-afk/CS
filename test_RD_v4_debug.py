# -*- coding: utf-8 -*-
"""RD_v4-friendly generic test / evaluation script for RD-family segmentation models.

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
import csv
import json
import math
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

SCRIPT_VERSION = "20260602_SAVE_TO_WEIGHT_DIR_V2"


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


def tensor_to_float(x, default: float = float("nan")) -> float:
    try:
        if torch.is_tensor(x):
            return float(x.detach().float().mean().cpu().item())
        if isinstance(x, (float, int)):
            return float(x)
    except Exception:
        return default
    return default


def update_sum_dict(sum_dict: Dict[str, float], count_dict: Dict[str, int], values: Dict[str, float]):
    for k, v in values.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if math.isfinite(fv):
            sum_dict[k] = sum_dict.get(k, 0.0) + fv
            count_dict[k] = count_dict.get(k, 0) + 1


def avg_dict(sum_dict: Dict[str, float], count_dict: Dict[str, int]) -> Dict[str, float]:
    return {k: sum_dict[k] / max(count_dict.get(k, 1), 1) for k in sum_dict.keys()}


@torch.no_grad()
def collect_output_stats(outputs, masks=None, threshold: float = 0.5) -> Dict[str, float]:
    """Collect RD_v4 diagnostic stats from one forward output.

    It reads scalar outputs such as dlaem4_alpha/gate/lambda and probability
    distribution statistics. This does not affect inference.
    """
    stats: Dict[str, float] = {}
    if isinstance(outputs, dict):
        for k, v in outputs.items():
            lk = k.lower()
            if torch.is_tensor(v) and v.numel() <= 16 and any(s in lk for s in ("alpha", "gate", "lambda", "weight")):
                if v.dim() <= 2:
                    stats[k] = tensor_to_float(v)
            if torch.is_tensor(v) and "prior" in lk and "logit" in lk and v.dim() == 4:
                p = torch.sigmoid(v.detach().float())
                stats[f"{k}_mean"] = float(p.mean().cpu().item())
                stats[f"{k}_std"] = float(p.std().cpu().item())
                stats[f"{k}_gt05"] = float((p > 0.5).float().mean().cpu().item())

    logits = extract_logits(outputs).detach().float()
    prob = torch.sigmoid(logits)
    stats["logit_mean"] = float(logits.mean().cpu().item())
    stats["logit_std"] = float(logits.std().cpu().item())
    stats["prob_mean"] = float(prob.mean().cpu().item())
    stats["prob_std"] = float(prob.std().cpu().item())
    stats["prob_min"] = float(prob.min().cpu().item())
    stats["prob_max"] = float(prob.max().cpu().item())
    stats["prob_gt_03"] = float((prob > 0.30).float().mean().cpu().item())
    stats["prob_gt_05"] = float((prob > 0.50).float().mean().cpu().item())
    stats["prob_gt_07"] = float((prob > 0.70).float().mean().cpu().item())
    stats["pred_ratio"] = float((prob > threshold).float().mean().cpu().item())

    if masks is not None:
        m = masks.detach().float()
        if prob.shape[-2:] != m.shape[-2:]:
            prob = F.interpolate(prob, size=m.shape[-2:], mode="bilinear", align_corners=False)
        pred = (prob > threshold).float()
        bg = 1.0 - m
        numel = float(m.numel())
        tp = (pred * m).sum()
        fp = (pred * bg).sum()
        fn = ((1.0 - pred) * m).sum()
        fg_pixels = m.sum().clamp_min(1.0)
        bg_pixels = bg.sum().clamp_min(1.0)
        stats["target_ratio"] = float(m.mean().cpu().item())
        stats["pred_target_gap"] = stats["pred_ratio"] - stats["target_ratio"]
        stats["tp_ratio"] = float((tp / max(numel, 1.0)).cpu().item())
        stats["fp_ratio"] = float((fp / max(numel, 1.0)).cpu().item())
        stats["fn_ratio"] = float((fn / max(numel, 1.0)).cpu().item())
        stats["batch_precision"] = float((tp / (tp + fp + 1e-6)).cpu().item())
        stats["batch_recall"] = float((tp / (tp + fn + 1e-6)).cpu().item())
        stats["fg_prob_mean"] = float(((prob * m).sum() / fg_pixels).cpu().item())
        stats["bg_prob_mean"] = float(((prob * bg).sum() / bg_pixels).cpu().item())
        stats["fg_bg_prob_gap"] = stats["fg_prob_mean"] - stats["bg_prob_mean"]
    return stats


def module_param_summary(model):
    rows = []
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        rows.append({"module": name, "total": total, "trainable": trainable, "frozen": total - trainable})
    return rows


def special_parameter_stats(model) -> Dict[str, float]:
    buckets = {
        "dino_gamma": [],
        "adapter_up_weight": [],
        "adapter_down_weight": [],
        "alpha_raw": [],
        "lambda_prior_raw": [],
    }
    for name, p in model.named_parameters():
        lname = name.lower()
        if "gamma_attn" in lname or "gamma_ffn" in lname:
            buckets["dino_gamma"].append(p)
        if "adapter" in lname and "up.weight" in lname:
            buckets["adapter_up_weight"].append(p)
        if "adapter" in lname and "down.weight" in lname:
            buckets["adapter_down_weight"].append(p)
        if "alpha" in lname and lname.endswith("raw"):
            buckets["alpha_raw"].append(p)
        if "lambda_prior" in lname and lname.endswith("raw"):
            buckets["lambda_prior_raw"].append(p)

    out: Dict[str, float] = {}
    for key, params in buckets.items():
        if not params:
            continue
        total_sq = 0.0
        abs_sum = 0.0
        elem_count = 0
        max_abs = 0.0
        for p in params:
            v = p.detach().float()
            n = float(v.norm(2).cpu().item())
            total_sq += n * n
            av = v.abs()
            abs_sum += float(av.sum().cpu().item())
            elem_count += int(v.numel())
            max_abs = max(max_abs, float(av.max().cpu().item()))
        out[f"special_{key}_norm"] = math.sqrt(total_sq)
        out[f"special_{key}_mean_abs"] = abs_sum / max(elem_count, 1)
        out[f"special_{key}_max_abs"] = max_abs
    return out


def write_flat_csv(path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_model_summary(path: str, model, config: Dict[str, Any]):
    total, trainable, frozen = count_params(model)
    with open(path, "w", encoding="utf-8") as f:
        f.write("========== RD_v4 Test Model Summary ==========" + "\n")
        f.write(f"Model: {get_model_name(config)}\n")
        f.write(f"Total params:     {total/1e6:.4f} M\n")
        f.write(f"Trainable params: {trainable/1e6:.4f} M\n")
        f.write(f"Frozen params:    {frozen/1e6:.4f} M\n\n")
        f.write("[Top-level modules]\n")
        for r in module_param_summary(model):
            f.write(f"{r['module']:<22} total={r['total']/1e6:8.4f}M trainable={r['trainable']/1e6:8.4f}M frozen={r['frozen']/1e6:8.4f}M\n")
        f.write("\n[Model config]\n")
        f.write(json.dumps(config.get("model", {}), indent=2, ensure_ascii=False))
        f.write("\n")


def format_diag(stats: Dict[str, float], max_items: int = 32) -> str:
    priority = [
        "dlaem4_alpha", "dlaem8_alpha", "dlaem16_alpha",
        "dlaem4_gate_mean", "dlaem8_gate_mean", "dlaem16_gate_mean",
        "dlaem4_lambda_prior", "dlaem8_lambda_prior", "dlaem16_lambda_prior",
        "prob_mean", "prob_std", "pred_ratio", "target_ratio", "pred_target_gap",
        "fg_prob_mean", "bg_prob_mean", "fg_bg_prob_gap",
        "fp_ratio", "fn_ratio", "prior4_logits_mean", "prior8_logits_mean", "prior16_logits_mean",
    ]
    keys = [k for k in priority if k in stats] + [k for k in sorted(stats.keys()) if k not in priority]
    return " | ".join([f"{k}={stats[k]:.4f}" for k in keys[:max_items]])


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
    parser.add_argument("--diagnostics", action="store_true", help="enable RD_v4 branch/output diagnostics; disables return_aux=False and collects DLAEM/prior stats")
    parser.add_argument("--diag-threshold", type=float, default=0.5, help="threshold used for diagnostic pred/fp/fn ratios")
    parser.add_argument("--diag-max-batches", type=int, default=0, help="0 means all batches; otherwise collect diagnostics for first N batches")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    print(f"[test_RD_v4_debug] version={SCRIPT_VERSION} | default save_dir=weight directory unless --save-dir is explicitly set")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = int(config["dataset"].get("input_size", 1024))
    model = get_model(config["model"], img_size=img_size).to(device)
    set_model_return_aux(model, bool(args.diagnostics))
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

    # 默认保存位置参考原 test.py：直接保存到权重所在目录，方便和训练日志/权重放在一起。
    # 如果用户显式传入 --save-dir，则使用用户指定目录。
    save_dir = args.save_dir.strip() if isinstance(args.save_dir, str) else ""
    if not save_dir:
        save_dir = os.path.dirname(os.path.abspath(args.weights)) or "."
        save_mode = "weight_dir(default)"
    else:
        save_dir = os.path.abspath(save_dir)
        save_mode = "custom(--save-dir)"
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    save_index = 0
    save_thr = float(args.save_threshold) if args.save_threshold is not None else 0.5
    diag_sum: Dict[str, float] = {}
    diag_count: Dict[str, int] = {}
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(loader, desc=f"Testing {get_model_name(config)} [{args.split}]", leave=False)):
            imgs, masks, _ = unwrap_batch(batch_data)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            outputs_for_diag = None
            if args.tta:
                logits = apply_tta(model, imgs, device=device, amp=amp_enabled, mode=args.tta_mode)
                # TTA returns averaged logits. For diagnostics, optionally do one extra clean forward.
                if args.diagnostics and (args.diag_max_batches <= 0 or batch_idx < args.diag_max_batches):
                    with amp_context(device, amp_enabled):
                        outputs_for_diag = model_forward(model, imgs)
            else:
                with amp_context(device, amp_enabled):
                    outputs_for_diag = model_forward(model, imgs)
                    logits = extract_logits(outputs_for_diag)
            if args.diagnostics and outputs_for_diag is not None and (args.diag_max_batches <= 0 or batch_idx < args.diag_max_batches):
                update_sum_dict(diag_sum, diag_count, collect_output_stats(outputs_for_diag, masks, threshold=float(args.diag_threshold)))
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

    diagnostics = avg_dict(diag_sum, diag_count)
    special_stats = special_parameter_stats(model) if args.diagnostics else {}
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
    print(f" SaveMode: {save_mode}")
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
    if args.diagnostics:
        print("--------------------------------------------------")
        print(" RD_v4 Diagnostics:")
        print(" " + format_diag(diagnostics))
        if special_stats:
            print(" SpecialParam: " + " | ".join([f"{k}={v:.6g}" for k, v in sorted(special_stats.items())]))
    result_log_path = os.path.join(save_dir, "test_results.log")
    print(f" 结果保存: {result_log_path}")
    print("==================================================")

    # 追加写入和原 test.py 一样的 test_results.log，保证多次测试不会覆盖历史结果。
    result_lines = []
    result_lines.append("\n==================================================")
    result_lines.append(f" ⏱测试时间: {now}")
    result_lines.append(f" 权重路径: {args.weights}")
    result_lines.append(f" 加载字段: {load_info['state_type']} | loaded={load_info['loaded_keys']} missing={len(load_info['missing'])} unexpected={len(load_info['unexpected'])} skipped={len(load_info['skipped'])}")
    result_lines.append(f" 数据集:   {config['dataset']['name']}")
    result_lines.append(f" Split:    {args.split}")
    result_lines.append(f" 模型:     {get_model_name(config)}")
    result_lines.append(f" Params:   total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M frozen={frozen/1e6:.2f}M")
    result_lines.append(f" TTA:      {args.tta} ({args.tta_mode}) | AMP: {amp_enabled}")
    result_lines.append(f" 脚本版本: {SCRIPT_VERSION}")
    result_lines.append(f" 保存模式: {save_mode}")
    result_lines.append("--------------------------------------------------")
    for t, m in metrics.items():
        result_lines.append(
            f" threshold={t:.3f} | IoU={m['iou']*100:.2f}% | "
            f"Precision={m['precision']*100:.2f}% | Recall={m['recall']*100:.2f}% | F1={m['f1']*100:.2f}% | "
            f"pred_fg={m['pred_fg_ratio']*100:.2f}% gt_fg={m['gt_fg_ratio']*100:.2f}%"
        )
    result_lines.append("--------------------------------------------------")
    result_lines.append(
        f" 最优阈值: {best_t:.3f} | IoU: {best_m['iou']*100:.2f}% | "
        f"Precision: {best_m['precision']*100:.2f}% | Recall: {best_m['recall']*100:.2f}% | F1: {best_m['f1']*100:.2f}%"
    )
    if args.diagnostics:
        result_lines.append("--------------------------------------------------")
        result_lines.append(" RD_v4 Diagnostics:")
        result_lines.append(" " + format_diag(diagnostics))
        if special_stats:
            result_lines.append(" SpecialParam: " + " | ".join([f"{k}={v:.6g}" for k, v in sorted(special_stats.items())]))
    result_lines.append("==================================================\n")
    with open(result_log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(result_lines))

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
        "diagnostics_enabled": bool(args.diagnostics),
        "diagnostics": diagnostics,
        "special_parameter_stats": special_stats,
        "script_version": SCRIPT_VERSION,
        "save_mode": save_mode,
        "result_log_path": result_log_path,
    }
    # 详细结果文件加上 split 前缀，避免覆盖训练阶段的 metrics.csv。
    metrics_json_path = os.path.join(save_dir, f"{args.split}_metrics.json")
    metrics_csv_path = os.path.join(save_dir, f"{args.split}_metrics.csv")
    summary_path = os.path.join(save_dir, f"{args.split}_model_summary_test.txt")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_metrics_csv(metrics_csv_path, metrics)
    write_model_summary(summary_path, model, config)
    if args.diagnostics:
        diag_json_path = os.path.join(save_dir, f"{args.split}_diagnostics.json")
        diag_csv_path = os.path.join(save_dir, f"{args.split}_diagnostics.csv")
        with open(diag_json_path, "w", encoding="utf-8") as f:
            json.dump({"diagnostics": diagnostics, "special_parameter_stats": special_stats}, f, indent=2, ensure_ascii=False)
        flat = {}
        flat.update({f"diag_{k}": v for k, v in diagnostics.items()})
        flat.update({f"special_{k}": v for k, v in special_stats.items()})
        if flat:
            write_flat_csv(diag_csv_path, flat)


if __name__ == "__main__":
    main()
