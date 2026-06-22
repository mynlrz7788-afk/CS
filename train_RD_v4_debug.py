# -*- coding: utf-8 -*-
"""RD_v4-friendly generic training script for RD-family segmentation models.

适配对象：RD_v1 / RD_v2 / RD_v3 / RD_v4... 这类返回 dict 的模型，也兼容普通 tensor 输出模型。
核心特性：
- 支持 PEFT-DINO + DLAEM + 多尺度 decoder 输出的通用 loss；
- 支持 model.get_param_groups(base_lr, weight_decay)，自动按模型内部参数组微调；
- 支持 --finetune 加载预训练权重，只加载模型参数，自动跳过 shape mismatch；
- 支持 --resume 完整恢复 optimizer / scheduler / scaler；
- 支持冻结/解冻模块，便于做分阶段微调；
- AMP-safe boundary loss；
- 保存 train.log / metrics.csv / branch_stats.csv / model_summary.txt / best_model.pth / best_ema_model.pth / last_checkpoint.pth。
"""

import os
import csv
import json
import time
import datetime
import math
import copy
import random
import argparse
from contextlib import nullcontext
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.road_dataset import RoadDataset
from dataloaders.drive_dataset import DRIVEDataset
from models import get_model

SCRIPT_VERSION = "20260602_TRAIN_TIME_NO_IMPROVE_V2"


# =========================================================
# 1. Basic utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 训练分割模型通常更需要速度；需要完全可复现时可在 config 里自行关闭 benchmark。
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_model_name(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model", {})
    return model_cfg.get("name") or model_cfg.get("model_name") or config.get("model_name") or "RD"


def log_print(msg: str, log_file=None):
    print(msg, end="")
    if log_file is not None:
        log_file.write(msg)
        log_file.flush()


def unwrap_batch(batch_data):
    if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
        return batch_data[0], batch_data[1]
    if isinstance(batch_data, dict):
        img = batch_data.get("image") or batch_data.get("img") or batch_data.get("images")
        mask = batch_data.get("mask") or batch_data.get("label") or batch_data.get("masks")
        if img is not None and mask is not None:
            return img, mask
    raise RuntimeError(f"无法解析 batch 数据结构: {type(batch_data)}")


def amp_context(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return nullcontext()


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


def append_csv(path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.isfile(path)
    fieldnames = list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def safe_float_dict(prefix: str, values: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    for k, v in values.items():
        fv = tensor_to_float(v)
        if math.isfinite(fv):
            out[f"{prefix}{k}"] = fv
    return out


def get_loss_cfg(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return loss config while staying compatible with two config styles.

    Supported:
    1) Top-level:  config["loss"]
    2) Nested:     config["training"]["loss"]

    Nested values override top-level values when both exist. This keeps old RD configs
    working and also supports the RD_v4_CHN6_CUG.json style.
    """
    out: Dict[str, Any] = {}
    if isinstance(config.get("loss"), dict):
        out.update(config.get("loss", {}))
    train_loss = config.get("training", {}).get("loss", {})
    if isinstance(train_loss, dict):
        out.update(train_loss)
    return out


def cfg_get_first(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


# =========================================================
# 2. Dataset / model construction
# =========================================================
def build_dataset(config: Dict[str, Any], mode: str):
    ds_cfg = config["dataset"]
    name = ds_cfg["name"]
    root = ds_cfg["root_path"]
    img_size = int(ds_cfg.get("input_size", 1024))
    if name.lower() == "drive":
        return DRIVEDataset(root, name, mode=mode, img_size=img_size)
    return RoadDataset(root, name, mode=mode, img_size=img_size)


def create_exp_dir(config: Dict[str, Any], tag: str = "") -> str:
    dataset_name = config["dataset"]["name"]
    model_name = get_model_name(config)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    suffix = f"_{tag}" if tag else ""
    exp_dir = os.path.join("saved_runs", dataset_name, f"{model_name}{suffix}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    return exp_dir


def set_model_return_aux(model: nn.Module, enabled: bool):
    # RD 系列通常有 return_aux；训练时尽量打开，测试/纯验证可关闭。
    for m in [model]:
        if hasattr(m, "return_aux"):
            try:
                setattr(m, "return_aux", bool(enabled))
            except Exception:
                pass


def forward_train(model: nn.Module, imgs: torch.Tensor):
    # RD_v1/RD_v2 可能提供 forward_train，强制返回 dict 以计算 aux/prior/boundary loss。
    if hasattr(model, "forward_train"):
        return model.forward_train(imgs)
    return model(imgs)


def extract_logits(outputs):
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if isinstance(outputs, dict):
        for k in ("final_logits", "logits", "out", "pred", "prediction"):
            if k in outputs and torch.is_tensor(outputs[k]):
                return outputs[k]
        # 兜底：找第一个 B×1×H×W tensor。
        for v in outputs.values():
            if torch.is_tensor(v) and v.dim() == 4 and v.shape[1] == 1:
                return v
    raise RuntimeError(f"无法从模型输出中提取 logits: {type(outputs)}")


# =========================================================
# 3. Checkpoint / fine-tuning helpers
# =========================================================
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


def extract_state_dict(ckpt_obj: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    info = {}
    if isinstance(ckpt_obj, dict):
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                for meta in ("epoch", "metrics", "config"):
                    if meta in ckpt_obj:
                        info[meta] = ckpt_obj[meta]
                return strip_module_prefix(ckpt_obj[key]), info
    if isinstance(ckpt_obj, dict):
        return strip_module_prefix(ckpt_obj), info
    raise RuntimeError("checkpoint 格式无法解析")


def load_model_weights(
    model: nn.Module,
    weight_path: str,
    strict: bool = False,
    skip_mismatch: bool = True,
    log_file=None,
    prefix: str = "",
) -> Dict[str, Any]:
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state, info = extract_state_dict(ckpt)
    model_state = model.state_dict()
    if skip_mismatch:
        filtered = OrderedDict()
        skipped = []
        for k, v in state.items():
            if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
                filtered[k] = v
            else:
                skipped.append(k)
        state = filtered
    else:
        skipped = []
    missing, unexpected = model.load_state_dict(state, strict=strict)
    msg = (
        f"{prefix}Load weights: {weight_path}\n"
        f"{prefix}  loaded_keys={len(state)}, missing={len(missing)}, unexpected={len(unexpected)}, skipped_mismatch={len(skipped)}\n"
    )
    log_print(msg, log_file)
    if skipped:
        log_print(f"{prefix}  skipped examples: {skipped[:10]}\n", log_file)
    return {"missing": list(missing), "unexpected": list(unexpected), "skipped": skipped, "info": info}


def apply_freeze_unfreeze(model: nn.Module, config: Dict[str, Any], args, log_file=None):
    ft_cfg = config.get("finetune", {})
    freeze_modules = list(ft_cfg.get("freeze_modules", []))
    unfreeze_modules = list(ft_cfg.get("unfreeze_modules", []))
    if args.freeze_modules:
        freeze_modules += [x.strip() for x in args.freeze_modules.split(",") if x.strip()]
    if args.unfreeze_modules:
        unfreeze_modules += [x.strip() for x in args.unfreeze_modules.split(",") if x.strip()]

    def match(name: str, patterns: List[str]) -> bool:
        return any(p and p in name for p in patterns)

    if freeze_modules:
        for name, p in model.named_parameters():
            if match(name, freeze_modules):
                p.requires_grad = False
        log_print(f"Freeze modules by patterns: {freeze_modules}\n", log_file)
    if unfreeze_modules:
        for name, p in model.named_parameters():
            if match(name, unfreeze_modules):
                p.requires_grad = True
        log_print(f"Unfreeze modules by patterns: {unfreeze_modules}\n", log_file)


# =========================================================
# 4. Loss / metrics
# =========================================================
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        targets = targets.float()
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
        return self.bce_weight * bce + self.dice_weight * dice


def soft_boundary(mask: torch.Tensor, k: int = 3):
    mask = mask.float()
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
    ero = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=k // 2)
    return (dil - ero).clamp(0, 1)


def find_aux_logits(outputs: Dict[str, Any]) -> List[Tuple[str, torch.Tensor]]:
    aux = []
    for key in ("logit16", "logit8", "logit4", "logit2", "aux16", "aux8", "aux4", "aux2"):
        if key in outputs and torch.is_tensor(outputs[key]):
            aux.append((key, outputs[key]))
    # 兼容 aux_logits / aux_outputs 列表。
    for key in ("aux_logits", "aux_outputs", "deep_supervision"):
        if key in outputs:
            value = outputs[key]
            if isinstance(value, (list, tuple)):
                for i, t in enumerate(value):
                    if torch.is_tensor(t):
                        aux.append((f"{key}{i}", t))
    return aux


def aux_weight_for_key(key: str, loss_cfg: Dict[str, Any]) -> float:
    key_l = key.lower()
    aux_weights = loss_cfg.get("aux_weights", {})
    if isinstance(aux_weights, dict):
        if key in aux_weights:
            return float(aux_weights[key])
        if key_l in aux_weights:
            return float(aux_weights[key_l])
    if "16" in key_l:
        return float(loss_cfg.get("aux16_weight", loss_cfg.get("aux_weight", 0.15)))
    if "8" in key_l:
        return float(loss_cfg.get("aux8_weight", loss_cfg.get("aux_weight", 0.25)))
    if "4" in key_l:
        return float(loss_cfg.get("aux4_weight", loss_cfg.get("aux_weight", 0.15)))
    if "2" in key_l:
        return float(loss_cfg.get("aux2_weight", loss_cfg.get("aux_weight", 0.08)))
    return float(loss_cfg.get("aux_weight", 0.1))


def find_prior_logits(outputs: Dict[str, Any]) -> List[Tuple[str, torch.Tensor]]:
    priors = []
    for k, v in outputs.items():
        if torch.is_tensor(v) and "prior" in k.lower() and "logit" in k.lower() and v.dim() == 4:
            priors.append((k, v))
    return priors


def rd_loss(outputs, masks, criterion: nn.Module, cfg: Dict[str, Any]):
    # 普通模型输出 tensor 时，只计算主损失。
    if not isinstance(outputs, dict):
        loss = criterion(extract_logits(outputs), masks)
        return loss, {"main": loss.detach().item(), "total": loss.detach().item()}

    loss_cfg = get_loss_cfg(cfg)
    logits = extract_logits(outputs)
    main = criterion(logits, masks)
    total = main
    parts = {"main": main.detach().item()}

    # 多尺度 decoder auxiliary logits（辅助输出）。
    for key, aux_logit in find_aux_logits(outputs):
        w = aux_weight_for_key(key, loss_cfg)
        if w <= 0:
            continue
        l = criterion(aux_logit, masks)
        total = total + w * l
        parts[key] = l.detach().item()

    # Road prior heads（道路先验头）。
    prior_weight = float(loss_cfg.get("prior_weight", 0.0))
    prior_weights = loss_cfg.get("prior_weights", {})
    priors = find_prior_logits(outputs)
    if priors:
        if isinstance(prior_weights, dict) and prior_weights:
            weighted_sum = 0.0
            weight_sum = 0.0
            raw_sum = 0.0
            raw_count = 0
            for name, p in priors:
                w = float(prior_weights.get(name, prior_weights.get(name.lower(), 0.0)))
                if w <= 0:
                    continue
                y = F.interpolate(masks.float(), size=p.shape[-2:], mode="nearest")
                l = criterion(p, y)
                weighted_sum = weighted_sum + w * l
                weight_sum += w
                raw_sum = raw_sum + l.detach()
                raw_count += 1
                parts[name] = l.detach().item()
            if weight_sum > 0:
                total = total + weighted_sum
                parts["prior"] = (raw_sum / max(raw_count, 1)).detach().item() if torch.is_tensor(raw_sum) else float("nan")
        elif prior_weight > 0:
            ploss = 0.0
            for _, p in priors:
                y = F.interpolate(masks.float(), size=p.shape[-2:], mode="nearest")
                ploss = ploss + criterion(p, y)
            ploss = ploss / max(len(priors), 1)
            total = total + prior_weight * ploss
            parts["prior"] = ploss.detach().item()

    # AMP-safe boundary loss（边界损失）。
    boundary_weight = float(loss_cfg.get("boundary_weight", 0.0))
    if boundary_weight > 0:
        k = int(loss_cfg.get("boundary_kernel", 3))
        # binary_cross_entropy(prob, target) 在 autocast 下不安全，因此转成 logits 后用 BCEWithLogits。
        with torch.amp.autocast("cuda", enabled=False) if logits.is_cuda else nullcontext():
            target_edge = soft_boundary(masks.float(), k=k)
            pred_prob = torch.sigmoid(logits.float())
            pred_edge = soft_boundary(pred_prob, k=k).clamp(1e-5, 1.0 - 1e-5)
            pred_edge_logits = torch.logit(pred_edge)
            bl = F.binary_cross_entropy_with_logits(pred_edge_logits, target_edge.float())
        total = total + boundary_weight * bl
        parts["boundary"] = bl.detach().item()

    parts["total"] = total.detach().item()
    return total, parts


@torch.no_grad()
def compute_counts_from_logits(logits: torch.Tensor, masks: torch.Tensor, threshold: float):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks = masks.float()
    tp = (preds * masks).sum().item()
    fp = (preds * (1.0 - masks)).sum().item()
    fn = ((1.0 - preds) * masks).sum().item()
    return tp, fp, fn, probs


def metrics_from_counts(tp: float, fp: float, fn: float) -> Dict[str, float]:
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}


# =========================================================
# 5. Branch / parameter diagnostics
# =========================================================
@torch.no_grad()
def collect_output_stats(outputs, masks=None, threshold: float = 0.5) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if not isinstance(outputs, dict):
        return stats

    # 直接收集标量类 tensor，例如 dlaem8_alpha / dlaem16_gate_mean。
    for k, v in outputs.items():
        lk = k.lower()
        if torch.is_tensor(v) and v.numel() <= 16 and any(s in lk for s in ("alpha", "gate", "weight", "lambda")):
            if v.dim() <= 2:
                stats[k] = tensor_to_float(v)
        if isinstance(v, dict):
            for kk, vv in v.items():
                lkk = kk.lower()
                if torch.is_tensor(vv) and vv.numel() <= 16 and any(s in lkk for s in ("alpha", "gate", "weight", "lambda")):
                    stats[f"{k}_{kk}"] = tensor_to_float(vv)

    # decoder fusion weights（多尺度输出融合权重）。
    for key in ("decoder_fuse_weights", "fuse_weights", "fusion_weights"):
        if key in outputs and torch.is_tensor(outputs[key]):
            w = outputs[key].detach().float().flatten().cpu().tolist()
            for i, v in enumerate(w):
                stats[f"{key}_{i}"] = float(v)

    # prior 分布。
    for k, v in outputs.items():
        if torch.is_tensor(v) and "prior" in k.lower() and "logit" in k.lower() and v.dim() == 4:
            p = torch.sigmoid(v.detach().float())
            stats[f"{k}_mean"] = float(p.mean().cpu().item())
            stats[f"{k}_std"] = float(p.std().cpu().item())

    logits = extract_logits(outputs).detach().float()
    prob = torch.sigmoid(logits)

    # Output distribution diagnostics（输出分布诊断）：
    # 用来判断模型是“整体偏保守/偏激进”、logits 是否饱和、阈值 0.5 是否合适。
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


def count_params(model: nn.Module) -> Tuple[int, int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def module_param_summary(model: nn.Module) -> List[Dict[str, Any]]:
    rows = []
    for name, module in model.named_children():
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        rows.append({"module": name, "total": total, "trainable": trainable, "frozen": total - trainable})
    return rows


def optimizer_group_summary(optimizer) -> List[Dict[str, Any]]:
    rows = []
    for i, g in enumerate(optimizer.param_groups):
        params = g.get("params", [])
        n = sum(p.numel() for p in params if p.requires_grad)
        rows.append({
            "idx": i,
            "name": g.get("name", f"group{i}"),
            "params": n,
            "lr": float(g.get("lr", 0.0)),
            "weight_decay": float(g.get("weight_decay", 0.0)),
        })
    return rows


def grad_norm_from_params(params: Iterable[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    for p in params:
        if p.grad is None:
            continue
        try:
            v = float(p.grad.detach().float().norm(2).cpu().item())
            total_sq += v * v
        except Exception:
            pass
    return math.sqrt(total_sq)


def grad_diagnostics_from_params(params: Iterable[torch.nn.Parameter]) -> Dict[str, float]:
    """Return detailed gradient diagnostics for one optimizer group.

    这些统计只用于日志诊断，不参与 loss / backward / optimizer.step，
    因此不会改变训练逻辑。
    """
    total_sq = 0.0
    max_abs = 0.0
    abs_sum = 0.0
    elem_count = 0
    trainable_tensors = 0
    with_grad_tensors = 0
    none_grad_tensors = 0
    zero_grad_tensors = 0

    for p in params:
        if p is None or not p.requires_grad:
            continue
        trainable_tensors += 1
        if p.grad is None:
            none_grad_tensors += 1
            continue
        try:
            g = p.grad.detach().float()
            norm = float(g.norm(2).cpu().item())
            total_sq += norm * norm
            if norm == 0.0:
                zero_grad_tensors += 1
            with_grad_tensors += 1
            abs_g = g.abs()
            max_abs = max(max_abs, float(abs_g.max().cpu().item()))
            abs_sum += float(abs_g.sum().cpu().item())
            elem_count += int(g.numel())
        except Exception:
            pass

    return {
        "norm": math.sqrt(total_sq),
        "mean_abs": abs_sum / max(elem_count, 1),
        "max_abs": max_abs,
        "trainable_tensors": float(trainable_tensors),
        "with_grad_tensors": float(with_grad_tensors),
        "none_grad_tensors": float(none_grad_tensors),
        "zero_grad_tensors": float(zero_grad_tensors),
    }


def grad_norm_by_optimizer_groups(optimizer) -> Dict[str, float]:
    out = {}
    for g in optimizer.param_groups:
        name = str(g.get("name", "group"))
        diag = grad_diagnostics_from_params(g.get("params", []))
        # 保留旧字段名，兼容原日志和 CSV。
        out[f"grad_{name}"] = diag["norm"]
        # 新增细粒度诊断字段。
        out[f"grad_{name}_mean_abs"] = diag["mean_abs"]
        out[f"grad_{name}_max_abs"] = diag["max_abs"]
        out[f"grad_{name}_with_grad_tensors"] = diag["with_grad_tensors"]
        out[f"grad_{name}_none_grad_tensors"] = diag["none_grad_tensors"]
        out[f"grad_{name}_zero_grad_tensors"] = diag["zero_grad_tensors"]
    return out


def param_norm_by_optimizer_groups(optimizer) -> Dict[str, float]:
    """Parameter norm diagnostics by optimizer group.

    用来判断某个分支虽然有梯度，但参数是否几乎不变化或数值异常。
    """
    out = {}
    for g in optimizer.param_groups:
        name = str(g.get("name", "group"))
        total_sq = 0.0
        abs_sum = 0.0
        elem_count = 0
        max_abs = 0.0
        for p in g.get("params", []):
            if p is None or not p.requires_grad:
                continue
            try:
                v = p.detach().float()
                n = float(v.norm(2).cpu().item())
                total_sq += n * n
                av = v.abs()
                abs_sum += float(av.sum().cpu().item())
                elem_count += int(v.numel())
                max_abs = max(max_abs, float(av.max().cpu().item()))
            except Exception:
                pass
        out[f"param_{name}_norm"] = math.sqrt(total_sq)
        out[f"param_{name}_mean_abs"] = abs_sum / max(elem_count, 1)
        out[f"param_{name}_max_abs"] = max_abs
    return out


def grad_diagnostics_by_named_modules(model: nn.Module) -> Dict[str, float]:
    """Fine-grained gradient diagnostics for RD_v4 modules.

    Optimizer group `dlaem` intentionally merges DLAEM4/8/16, but RD_v4 needs to know
    whether the new shallow DLAEM4 has gradient. This function only reads gradients
    after backward/unscale and does not change training.
    """
    pattern_map = {
        "named_dlaem4": ["dlaem4."],
        "named_dlaem8": ["dlaem8."],
        "named_dlaem16": ["dlaem16."],
        "named_dino_adapter": ["adapter_attn", "adapter_ffn", "gamma_attn", "gamma_ffn"],
        "named_priors": ["prior4.", "prior8.", "prior16."],
    }
    buckets: Dict[str, List[nn.Parameter]] = {k: [] for k in pattern_map}
    for name, p in model.named_parameters():
        for key, pats in pattern_map.items():
            if any(pat in name for pat in pats):
                buckets[key].append(p)
    out: Dict[str, float] = {}
    for key, params in buckets.items():
        if not params:
            continue
        diag = grad_diagnostics_from_params(params)
        out[f"grad_{key}"] = diag["norm"]
        out[f"grad_{key}_mean_abs"] = diag["mean_abs"]
        out[f"grad_{key}_max_abs"] = diag["max_abs"]
        out[f"grad_{key}_with_grad_tensors"] = diag["with_grad_tensors"]
        out[f"grad_{key}_none_grad_tensors"] = diag["none_grad_tensors"]
        out[f"grad_{key}_zero_grad_tensors"] = diag["zero_grad_tensors"]
    return out


def special_parameter_stats(model: nn.Module) -> Dict[str, float]:
    """Collect RD/DINO-specific parameter stats.

    主要用于检查 DINO adapter 的 gamma / adapter 权重是否真的在训练，
    以及 alpha raw 参数是否出现异常。字段不存在时自动跳过。
    """
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


def build_optimizer(model: nn.Module, config: Dict[str, Any]):
    train_cfg = config.get("training", {})
    base_lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    if hasattr(model, "get_param_groups"):
        groups = model.get_param_groups(base_lr=base_lr, weight_decay=weight_decay)
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=base_lr, weight_decay=weight_decay)


def write_model_summary(exp_dir: str, model: nn.Module, optimizer, config: Dict[str, Any]):
    total, trainable, frozen = count_params(model)
    path = os.path.join(exp_dir, "model_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("========== RD Generic Model Summary ==========" + "\n")
        f.write(f"Model: {get_model_name(config)}\n")
        f.write(f"Total params:     {total/1e6:.4f} M\n")
        f.write(f"Trainable params: {trainable/1e6:.4f} M\n")
        f.write(f"Frozen params:    {frozen/1e6:.4f} M\n\n")
        f.write("[Top-level modules]\n")
        for r in module_param_summary(model):
            f.write(f"{r['module']:<22} total={r['total']/1e6:8.4f}M trainable={r['trainable']/1e6:8.4f}M frozen={r['frozen']/1e6:8.4f}M\n")
        f.write("\n[Optimizer groups]\n")
        for r in optimizer_group_summary(optimizer):
            f.write(f"{r['idx']:02d} {r['name']:<18} params={r['params']/1e6:8.4f}M lr={r['lr']:.3e} wd={r['weight_decay']:.3e}\n")
        f.write("\n[Model config]\n")
        f.write(json.dumps(config.get("model", {}), indent=2, ensure_ascii=False))
        f.write("\n")


def format_stats(stats: Dict[str, float], max_items: int = 24) -> str:
    if not stats:
        return ""
    # 优先展示最有诊断意义的项。
    priority = [
        "dlaem4_alpha", "dlaem8_alpha", "dlaem16_alpha",
        "dlaem4_gate_mean", "dlaem8_gate_mean", "dlaem16_gate_mean",
        "dlaem4_lambda_prior", "dlaem8_lambda_prior", "dlaem16_lambda_prior",
        "prob_mean", "prob_std", "pred_ratio", "target_ratio", "pred_target_gap",
        "fg_prob_mean", "bg_prob_mean", "fg_bg_prob_gap",
        "fp_ratio", "fn_ratio", "tp_ratio",
        "prior4_logits_mean", "prior8_logits_mean", "prior16_logits_mean",
    ]
    keys = [k for k in priority if k in stats] + [k for k in sorted(stats.keys()) if k not in priority]
    parts = []
    for k in keys[:max_items]:
        parts.append(f"{k}={stats[k]:.4f}")
    return " | ".join(parts)


def overfit_message(train_metrics: Dict[str, float], val_metrics: Dict[str, Any]) -> str:
    train_iou = float(train_metrics.get("iou", float("nan")))
    val_iou = float(val_metrics.get("iou", float("nan")))
    train_main = float(train_metrics.get("loss_main", train_metrics.get("loss", float("nan"))))
    val_main = float(val_metrics.get("loss", float("nan")))
    iou_gap = train_iou - val_iou
    loss_gap = val_main - train_main if math.isfinite(train_main) and math.isfinite(val_main) else float("nan")
    flag = "正常"
    if math.isfinite(iou_gap) and iou_gap > 0.08 and math.isfinite(loss_gap) and loss_gap > 0.03:
        flag = "可能过拟合"
    elif math.isfinite(train_iou) and train_iou < 0.2 and math.isfinite(val_iou) and val_iou < 0.2:
        flag = "可能欠拟合/未收敛"
    return f"OverfitCheck[{flag}]: train_IoU={train_iou*100:.2f} val_IoU={val_iou*100:.2f} gap={iou_gap*100:.2f} | train_main={train_main:.4f} val_main={val_main:.4f} val-train={loss_gap:.4f}"


def diagnose_epoch(train_stats: Dict[str, float], val_stats: Dict[str, float], grad_stats: Dict[str, float]) -> str:
    """Generate a compact human-readable diagnosis for one epoch.

    只基于日志统计做提示，不改变训练。
    """
    notes = []

    for name in ("dlaem4", "dlaem8", "dlaem16"):
        alpha = train_stats.get(f"{name}_alpha", float("nan"))
        gnorm = grad_stats.get("grad_dlaem", float("nan"))
        if math.isfinite(alpha) and alpha < 1e-3:
            notes.append(f"{name}_alpha过小({alpha:.2e})")
        if math.isfinite(gnorm) and gnorm < 1e-8:
            notes.append("DLAEM梯度接近0")
            break

    dino_g = grad_stats.get("grad_dino_adapters", float("nan"))
    if math.isfinite(dino_g) and dino_g < 1e-8:
        notes.append("DINO-adapter梯度接近0")

    pred_gap = val_stats.get("pred_target_gap", float("nan"))
    if math.isfinite(pred_gap):
        if pred_gap > 0.015:
            notes.append(f"预测道路偏多(+{pred_gap*100:.2f}%)")
        elif pred_gap < -0.015:
            notes.append(f"预测道路偏少({pred_gap*100:.2f}%)")

    fg_bg_gap = val_stats.get("fg_bg_prob_gap", float("nan"))
    if math.isfinite(fg_bg_gap) and fg_bg_gap < 0.35:
        notes.append(f"前景/背景概率分离不足({fg_bg_gap:.3f})")

    fp = val_stats.get("fp_ratio", float("nan"))
    fn = val_stats.get("fn_ratio", float("nan"))
    if math.isfinite(fp) and math.isfinite(fn):
        if fp > fn * 1.25:
            notes.append("误检偏多_FP>FN")
        elif fn > fp * 1.25:
            notes.append("漏检偏多_FN>FP")

    if not notes:
        return "Diagnosis[暂无明显异常]"
    # 去重但保留顺序
    uniq = []
    for x in notes:
        if x not in uniq:
            uniq.append(x)
    return "Diagnosis[" + "；".join(uniq[:8]) + "]"


# =========================================================
# 6. EMA / validation / saving
# =========================================================
class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd and v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            elif k in msd:
                v.copy_(msd[k])


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, metrics, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "config": config,
    }, path)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, config, device, threshold=0.5, desc="Valid"):
    model.eval()
    amp_enabled = bool(config.get("training", {}).get("amp", True))
    val_main_loss = 0.0
    val_full_loss = 0.0
    loss_sum: Dict[str, float] = {}
    loss_count: Dict[str, int] = {}
    stat_sum: Dict[str, float] = {}
    stat_count: Dict[str, int] = {}
    TP = FP = FN = 0.0

    sweep_values = config.get("logging", {}).get("threshold_sweep", [0.35, 0.4, 0.45, 0.5])
    sweep_counts = {str(t): {"TP": 0.0, "FP": 0.0, "FN": 0.0} for t in sweep_values}

    for batch_data in tqdm(loader, desc=desc, leave=False):
        imgs, masks = unwrap_batch(batch_data)
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with amp_context(device, amp_enabled):
            outputs = forward_train(model, imgs)
            logits = extract_logits(outputs)
            main_loss = criterion(logits, masks)
            full_loss, loss_parts = rd_loss(outputs, masks, criterion, config)

        val_main_loss += float(main_loss.item())
        val_full_loss += float(full_loss.item())
        update_sum_dict(loss_sum, loss_count, loss_parts)
        update_sum_dict(stat_sum, stat_count, collect_output_stats(outputs, masks, threshold=threshold))

        tp, fp, fn, probs = compute_counts_from_logits(logits, masks, threshold)
        TP += tp
        FP += fp
        FN += fn
        masks_f = masks.float()
        for t in sweep_values:
            key = str(t)
            p = (probs > float(t)).float()
            sweep_counts[key]["TP"] += (p * masks_f).sum().item()
            sweep_counts[key]["FP"] += (p * (1.0 - masks_f)).sum().item()
            sweep_counts[key]["FN"] += ((1.0 - p) * masks_f).sum().item()

    metrics = metrics_from_counts(TP, FP, FN)
    sweep_metrics = {}
    best_sweep = {"threshold": threshold, "iou": metrics["iou"], "f1": metrics["f1"]}
    for key, c in sweep_counts.items():
        m = metrics_from_counts(c["TP"], c["FP"], c["FN"])
        sweep_metrics[f"thr{key}_iou"] = m["iou"]
        sweep_metrics[f"thr{key}_f1"] = m["f1"]
        if m["iou"] > best_sweep["iou"]:
            best_sweep = {"threshold": float(key), "iou": m["iou"], "f1": m["f1"]}

    metrics.update({
        "loss": val_main_loss / max(len(loader), 1),
        "full_loss": val_full_loss / max(len(loader), 1),
        "loss_parts": avg_dict(loss_sum, loss_count),
        "branch_stats": avg_dict(stat_sum, stat_count),
        "threshold_sweep": sweep_metrics,
        "best_sweep": best_sweep,
    })
    return metrics


# =========================================================
# 7. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="json config path")
    parser.add_argument("--resume", type=str, default="", help="resume full checkpoint, including optimizer/scheduler/scaler")
    parser.add_argument("--finetune", type=str, default="", help="load model weights only for fine-tuning")
    parser.add_argument("--tag", type=str, default="", help="optional experiment name suffix")
    parser.add_argument("--freeze-modules", type=str, default="", help="comma separated parameter-name substrings to freeze")
    parser.add_argument("--unfreeze-modules", type=str, default="", help="comma separated parameter-name substrings to unfreeze after freezing")
    parser.add_argument("--strict-load", action="store_true", help="strict load for --finetune")
    parser.add_argument("--no-skip-mismatch", action="store_true", help="do not skip mismatched checkpoint keys")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = int(config.get("training", {}).get("seed", config.get("seed", 42)))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = create_exp_dir(config, tag=args.tag)
    log_file = open(os.path.join(exp_dir, "train.log"), "w", encoding="utf-8")
    metrics_csv = os.path.join(exp_dir, "metrics.csv")
    branch_csv = os.path.join(exp_dir, "branch_stats.csv")

    # 记录系统真实训练开始时间，和普通 train.py 保持一致，便于回看实验耗时。
    start_time_raw = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Dataset
    train_set = build_dataset(config, mode="train")
    val_set = build_dataset(config, mode="val")
    ds_cfg = config["dataset"]
    batch_size = int(ds_cfg.get("batch_size", 2))
    val_batch_size = int(ds_cfg.get("val_batch_size", batch_size))
    num_workers = int(ds_cfg.get("num_workers", 8))
    img_size = int(ds_cfg.get("input_size", 1024))

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=bool(ds_cfg.get("drop_last", True)),
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, val_batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    model = get_model(config["model"], img_size=img_size).to(device)
    set_model_return_aux(model, True)

    if args.finetune:
        load_model_weights(
            model,
            args.finetune,
            strict=bool(args.strict_load),
            skip_mismatch=not bool(args.no_skip_mismatch),
            log_file=log_file,
            prefix="[Finetune] ",
        )
    elif config.get("finetune", {}).get("weights"):
        load_model_weights(
            model,
            config["finetune"]["weights"],
            strict=bool(config.get("finetune", {}).get("strict", False)),
            skip_mismatch=bool(config.get("finetune", {}).get("skip_mismatch", True)),
            log_file=log_file,
            prefix="[Finetune] ",
        )

    apply_freeze_unfreeze(model, config, args, log_file=log_file)

    total, trainable, frozen = count_params(model)
    log_print("\n========== RD Generic Training ==========" + "\n", log_file)
    log_print(f"Exp dir: {exp_dir}\n", log_file)
    log_print(f"训练开始时间: {start_time_str}\n", log_file)
    log_print(f"Model: {get_model_name(config)}\n", log_file)
    log_print(f"Device: {device}\n", log_file)
    if torch.cuda.is_available():
        log_print(f"CUDA: {torch.cuda.get_device_name(0)} | capability={torch.cuda.get_device_capability(0)}\n", log_file)
    log_print(f"Dataset: {ds_cfg['name']} | train={len(train_set)} val={len(val_set)} | img={img_size}\n", log_file)
    log_print(f"Batch: train={batch_size} val={val_batch_size} workers={num_workers}\n", log_file)
    log_print(f"Params: total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M frozen={frozen/1e6:.2f}M\n", log_file)

    max_trainable = float(config.get("model", {}).get("max_trainable_params_m", 1e9)) * 1e6
    if max_trainable < 1e9 and trainable > max_trainable:
        log_print(f"⚠️ Trainable params exceed cap: {trainable/1e6:.2f}M > {max_trainable/1e6:.2f}M. Frozen DINO is not counted.\n", log_file)
    elif max_trainable < 1e9:
        log_print(f"Trainable-param cap: {max_trainable/1e6:.2f}M，当前满足要求；冻结 DINO 不计入上限。\n", log_file)

    log_print("\n[Top-level parameter summary]\n", log_file)
    for r in module_param_summary(model):
        log_print(f"  {r['module']:<22} total={r['total']/1e6:7.3f}M trainable={r['trainable']/1e6:7.3f}M frozen={r['frozen']/1e6:7.3f}M\n", log_file)

    loss_cfg = get_loss_cfg(config)
    criterion = BCEDiceLoss(
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
    ).to(device)

    optimizer = build_optimizer(model, config)
    train_cfg = config.get("training", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=str(train_cfg.get("scheduler_mode", "min")),
        factor=float(cfg_get_first(train_cfg, ["lr_factor", "scheduler_factor"], 0.5)),
        patience=int(cfg_get_first(train_cfg, ["lr_patience", "scheduler_patience"], 5)),
        min_lr=float(train_cfg.get("min_lr", 1e-7)),
    )
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    log_print("\n[Optimizer groups]\n", log_file)
    for r in optimizer_group_summary(optimizer):
        log_print(f"  {r['idx']:02d} {r['name']:<18} params={r['params']/1e6:7.3f}M lr={r['lr']:.3e} wd={r['weight_decay']:.3e}\n", log_file)
    write_model_summary(exp_dir, model, optimizer, config)

    ema = None
    ema_flag = bool(cfg_get_first(train_cfg, ["ema_enabled", "use_ema", "ema"], True))
    if ema_flag:
        ema = ModelEMA(model, decay=float(train_cfg.get("ema_decay", 0.999)))

    start_epoch = 1
    best_iou = -1.0
    best_ema_iou = -1.0
    best_epoch = -1
    best_ema_epoch = -1
    no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        state, _ = extract_state_dict(ckpt)
        model.load_state_dict(state, strict=True)
        if isinstance(ckpt, dict):
            if ckpt.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_iou = float(ckpt.get("metrics", {}).get("best_iou", best_iou))
        log_print(f"Resume from {args.resume}, start_epoch={start_epoch}\n", log_file)

    epochs = int(train_cfg.get("epochs", 200))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    threshold = float(train_cfg.get("threshold", 0.5))
    early_stop = int(train_cfg.get("early_stop_patience", 25))
    ema_start = int(train_cfg.get("ema_start_epoch", 3))
    log_cfg = config.get("logging", {})
    log_grad_norm = bool(log_cfg.get("log_grad_norm", True))

    log_print("\n[Training hyperparameters]\n", log_file)
    log_print(f"  epochs={epochs} | start_epoch={start_epoch} | threshold={threshold:.3f} | amp={amp_enabled}\n", log_file)
    log_print(f"  grad_clip={grad_clip} | early_stop_patience={early_stop} | ema_enabled={ema is not None} | ema_start_epoch={ema_start}\n", log_file)
    log_print(f"  scheduler=ReduceLROnPlateau(mode={train_cfg.get('scheduler_mode', 'min')}, factor={cfg_get_first(train_cfg, ['lr_factor', 'scheduler_factor'], 0.5)}, patience={cfg_get_first(train_cfg, ['lr_patience', 'scheduler_patience'], 5)})\n", log_file)
    log_print("--------------------------------------------------\n", log_file)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        set_model_return_aux(model, True)
        running = 0.0
        loss_sum: Dict[str, float] = {}
        loss_count: Dict[str, int] = {}
        stat_sum: Dict[str, float] = {}
        stat_count: Dict[str, int] = {}
        grad_sum: Dict[str, float] = {}
        grad_count: Dict[str, int] = {}
        TP = FP = FN = 0.0

        tbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}] Train", leave=False)
        for batch_data in tbar:
            imgs, masks = unwrap_batch(batch_data)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp_enabled):
                outputs = forward_train(model, imgs)
                logits = extract_logits(outputs)
                loss, loss_parts = rd_loss(outputs, masks, criterion, config)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if log_grad_norm:
                update_sum_dict(grad_sum, grad_count, grad_norm_by_optimizer_groups(optimizer))
                update_sum_dict(grad_sum, grad_count, grad_diagnostics_by_named_modules(model))

            total_norm_value = float("nan")
            if grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                total_norm_value = tensor_to_float(total_norm)
                update_sum_dict(grad_sum, grad_count, {"grad_total_preclip": total_norm_value})

            scaler.step(optimizer)
            scaler.update()
            if ema is not None and epoch >= ema_start:
                ema.update(model)

            running += float(loss.item())
            update_sum_dict(loss_sum, loss_count, loss_parts)
            update_sum_dict(stat_sum, stat_count, collect_output_stats(outputs, masks, threshold=threshold))

            with torch.no_grad():
                tp, fp, fn, _ = compute_counts_from_logits(logits.detach(), masks, threshold)
                TP += tp
                FP += fp
                FN += fn

            tbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "main": f"{loss_parts.get('main', 0):.4f}",
                "g": f"{total_norm_value:.2f}" if math.isfinite(total_norm_value) else "nan",
            })

        train_loss = running / max(len(train_loader), 1)
        train_metrics = metrics_from_counts(TP, FP, FN)
        train_loss_parts = avg_dict(loss_sum, loss_count)
        train_branch_stats = avg_dict(stat_sum, stat_count)
        train_grad_stats = avg_dict(grad_sum, grad_count)
        param_stats = param_norm_by_optimizer_groups(optimizer)
        special_param_stats = special_parameter_stats(model)
        train_epoch_metrics = {"loss": train_loss, **train_metrics}
        for k, v in train_loss_parts.items():
            train_epoch_metrics[f"loss_{k}"] = v

        raw = validate_one_epoch(model, val_loader, criterion, config, device, threshold=threshold, desc=f"Epoch [{epoch}/{epochs}] Valid Raw")
        ema_metrics = None
        if ema is not None and epoch >= ema_start:
            ema_metrics = validate_one_epoch(ema.ema, val_loader, criterion, config, device, threshold=threshold, desc=f"Epoch [{epoch}/{epochs}] Valid EMA")

        scheduler.step(raw["loss"])
        lrs = {g.get("name", f"group{i}"): float(g.get("lr", 0.0)) for i, g in enumerate(optimizer.param_groups)}
        lr_now = optimizer.param_groups[0]["lr"]

        msg = (
            f"Epoch {epoch:03d}/{epochs} | lr={lr_now:.2e} | "
            f"train: loss={train_loss:.4f} IoU={train_metrics['iou']*100:.2f} P={train_metrics['precision']*100:.2f} R={train_metrics['recall']*100:.2f} F1={train_metrics['f1']*100:.2f} | "
            f"raw: main_loss={raw['loss']:.4f} full_loss={raw['full_loss']:.4f} IoU={raw['iou']*100:.2f} P={raw['precision']*100:.2f} R={raw['recall']*100:.2f} F1={raw['f1']*100:.2f} "
            f"bestThr={raw['best_sweep']['threshold']:.2f}/IoU={raw['best_sweep']['iou']*100:.2f}"
        )
        if ema_metrics is not None:
            msg += f" | ema: IoU={ema_metrics['iou']*100:.2f} F1={ema_metrics['f1']*100:.2f} bestThr={ema_metrics['best_sweep']['threshold']:.2f}/IoU={ema_metrics['best_sweep']['iou']*100:.2f}"
        msg += "\n"
        log_print(msg, log_file)
        log_print("  " + overfit_message(train_epoch_metrics, raw) + "\n", log_file)
        log_print("  TrainBranch: " + format_stats(train_branch_stats) + "\n", log_file)
        log_print("  ValBranch:   " + format_stats(raw.get("branch_stats", {})) + "\n", log_file)
        if train_grad_stats:
            # 主梯度范数用 6 位小数，避免 DINO/DLAEM 小梯度被打印成 0.000。
            main_grad_keys = [k for k in sorted(train_grad_stats.keys()) if k.startswith("grad_") and not any(s in k for s in ("mean_abs", "max_abs", "tensors"))]
            grad_str = " | ".join([f"{k}={train_grad_stats[k]:.6f}" for k in main_grad_keys])
            log_print("  GradNorm: " + grad_str + "\n", log_file)

            detail_keys = [k for k in sorted(train_grad_stats.keys()) if any(s in k for s in ("mean_abs", "max_abs", "none_grad_tensors", "zero_grad_tensors"))]
            detail_show = [k for k in detail_keys if ("dino_adapters" in k or "named_dino" in k or "dlaem" in k or "named_dlaem" in k or "cnn_encoder" in k or "decoder" in k)][:32]
            if detail_show:
                detail_str = " | ".join([f"{k}={train_grad_stats[k]:.6g}" for k in detail_show])
                log_print("  GradDetail: " + detail_str + "\n", log_file)

        if param_stats:
            p_show = [k for k in sorted(param_stats.keys()) if k.endswith("_norm")][:16]
            p_str = " | ".join([f"{k}={param_stats[k]:.4f}" for k in p_show])
            log_print("  ParamNorm: " + p_str + "\n", log_file)

        if special_param_stats:
            sp_str = " | ".join([f"{k}={v:.6g}" for k, v in sorted(special_param_stats.items())])
            log_print("  SpecialParam: " + sp_str + "\n", log_file)

        log_print("  " + diagnose_epoch(train_branch_stats, raw.get("branch_stats", {}), train_grad_stats) + "\n", log_file)

        lr_str = " | ".join([f"{k}={v:.2e}" for k, v in lrs.items()])
        log_print("  LRGroups: " + lr_str + "\n", log_file)

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_iou": train_metrics["iou"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_f1": train_metrics["f1"],
            "val_loss": raw["loss"],
            "val_full_loss": raw["full_loss"],
            "val_iou": raw["iou"],
            "val_precision": raw["precision"],
            "val_recall": raw["recall"],
            "val_f1": raw["f1"],
            "val_best_thr": raw["best_sweep"]["threshold"],
            "val_best_thr_iou": raw["best_sweep"]["iou"],
            "best_iou_so_far": max(best_iou, raw["iou"]),
            "no_improve": no_improve,
        }
        row.update({f"train_{k}": v for k, v in train_loss_parts.items()})
        row.update({f"train_{k}": v for k, v in train_branch_stats.items()})
        row.update({f"val_{k}": v for k, v in raw.get("branch_stats", {}).items()})
        row.update({f"val_{k}": v for k, v in raw.get("threshold_sweep", {}).items()})
        row.update({f"grad_{k}": v for k, v in train_grad_stats.items()})
        row.update({f"param_{k}": v for k, v in param_stats.items()})
        row.update({f"special_{k}": v for k, v in special_param_stats.items()})
        append_csv(metrics_csv, row)

        branch_row = {"epoch": epoch}
        branch_row.update({f"train_{k}": v for k, v in train_branch_stats.items()})
        branch_row.update({f"val_{k}": v for k, v in raw.get("branch_stats", {}).items()})
        branch_row.update({f"grad_{k}": v for k, v in train_grad_stats.items()})
        branch_row.update({f"param_{k}": v for k, v in param_stats.items()})
        branch_row.update({f"special_{k}": v for k, v in special_param_stats.items()})
        append_csv(branch_csv, branch_row)

        improved = False
        if raw["iou"] > best_iou:
            best_iou = raw["iou"]
            best_epoch = epoch
            improved = True
            save_metrics = copy.deepcopy(raw)
            save_metrics["best_iou"] = best_iou
            save_metrics["best_epoch"] = best_epoch
            save_metrics["train_epoch"] = train_epoch_metrics
            save_metrics["train_branch_stats"] = train_branch_stats
            save_metrics["train_grad_stats"] = train_grad_stats
            save_metrics["param_stats"] = param_stats
            save_metrics["special_param_stats"] = special_param_stats
            save_checkpoint(os.path.join(exp_dir, "best_model.pth"), model, optimizer, scheduler, scaler, epoch, save_metrics, config)
            with open(os.path.join(exp_dir, "best_summary.json"), "w", encoding="utf-8") as f:
                json.dump(save_metrics, f, indent=2, ensure_ascii=False)
            log_print(f"  ✅ Save best raw model: IoU={best_iou*100:.2f} at epoch={best_epoch}\n", log_file)

        if ema_metrics is not None and ema_metrics["iou"] > best_ema_iou:
            best_ema_iou = ema_metrics["iou"]
            best_ema_epoch = epoch
            save_ema_metrics = copy.deepcopy(ema_metrics)
            save_ema_metrics["best_ema_iou"] = best_ema_iou
            save_ema_metrics["best_ema_epoch"] = best_ema_epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": ema.ema.state_dict(),
                "metrics": save_ema_metrics,
                "config": config,
            }, os.path.join(exp_dir, "best_ema_model.pth"))
            with open(os.path.join(exp_dir, "best_ema_summary.json"), "w", encoding="utf-8") as f:
                json.dump(save_ema_metrics, f, indent=2, ensure_ascii=False)
            log_print(f"  ✅ Save best EMA model: IoU={best_ema_iou*100:.2f} at epoch={best_ema_epoch}\n", log_file)

        last_metrics = copy.deepcopy(raw)
        last_metrics["best_iou"] = best_iou
        last_metrics["best_epoch"] = best_epoch
        last_metrics["best_ema_iou"] = best_ema_iou
        last_metrics["best_ema_epoch"] = best_ema_epoch
        last_metrics["train_epoch"] = train_epoch_metrics
        last_metrics["train_branch_stats"] = train_branch_stats
        last_metrics["train_grad_stats"] = train_grad_stats
        last_metrics["param_stats"] = param_stats
        last_metrics["special_param_stats"] = special_param_stats
        save_checkpoint(os.path.join(exp_dir, "last_checkpoint.pth"), model, optimizer, scheduler, scaler, epoch, last_metrics, config)

        no_improve = 0 if improved else no_improve + 1
        if improved:
            log_print("  ✅ 本轮 raw Val IoU 提升，连续未提升轮数重置为 0。\n", log_file)
        else:
            log_print(f"  ⚠️ 连续 {no_improve} 轮 raw Val IoU 没有提升。当前 best_raw_epoch={best_epoch}, best_raw_IoU={best_iou*100:.2f}%。\n", log_file)

        if no_improve >= early_stop:
            log_print(f"Early stop at epoch {epoch}: no improvement for {early_stop} epochs. Best raw epoch={best_epoch}, best EMA epoch={best_ema_epoch}.\n", log_file)
            break

    # --- 训练彻底结束后的时间统计 ---
    end_time_raw = time.time()
    end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration_sec = end_time_raw - start_time_raw
    hours, rem = divmod(duration_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    duration_str = f"{int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒"

    final_msg = (
        "--------------------------------------------------\n"
        f"训练开始时间: {start_time_str}\n"
        f"训练结束时间: {end_time_str}\n"
        f"整个实验总耗时: {duration_str}\n"
        f"最终连续未提升轮数: {no_improve}\n"
        f"Training done. Best raw IoU={best_iou*100:.2f} at epoch={best_epoch}, "
        f"best EMA IoU={best_ema_iou*100:.2f} at epoch={best_ema_epoch}\n"
        f"实验目录: {exp_dir}\n"
        "--------------------------------------------------\n"
    )
    log_print(final_msg, log_file)
    log_file.close()


if __name__ == "__main__":
    main()
