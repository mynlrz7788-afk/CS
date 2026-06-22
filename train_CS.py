# -*- coding: utf-8 -*-
"""train_CS.py

Generic training script for curvilinear-structure segmentation models in the CS project.

设计目标：
1) 保持原 train.py 的保存逻辑：saved_runs/{dataset}/{model}_{timestamp}/，断点续训继续写原目录；
2) 始终保存 latest_model.pth，验证 IoU 提升时保存 best_model.pth；
3) 支持断点续训、AMP、ReduceLROnPlateau、Early Stopping、模型自定义参数组；
4) 兼容普通 Tensor 输出、tuple/list 输出、dict 输出模型；
5) 记录更多曲线结构分割诊断指标，方便后续调参。
"""

import os
try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass

import csv
import copy
import json
import time
import math
import random
import argparse
import datetime
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from thop import profile
except Exception:
    profile = None

from dataloaders.road_dataset import RoadDataset
from models import get_model


# =========================================================
# 1. Basic utils
# =========================================================
def set_seed(seed: int = 3407, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




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


def format_input_size(input_size) -> str:
    h, w = parse_input_size(input_size)
    return f"{h}x{w}"


def log_print(msg: str, log_file=None):
    print(msg, end="")
    if log_file is not None:
        log_file.write(msg)
        log_file.flush()


def get_model_name(config: Dict[str, Any]) -> str:
    m = config.get("model", {})
    return m.get("name") or m.get("model_name") or config.get("model_name") or "CSModel"


def create_experiment_dir(config: Dict[str, Any]) -> str:
    dataset_name = config["dataset"]["name"]
    model_name = get_model_name(config)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_dir = os.path.join("saved_runs", dataset_name, f"{model_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    return exp_dir


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


# =========================================================
# 2. Dataset / model helpers
# =========================================================
def build_dataset(config: Dict[str, Any], mode: str):
    ds_cfg = config["dataset"]
    root = ds_cfg["root_path"]
    name = ds_cfg["name"]
    input_size = ds_cfg.get("input_size", 1024)
    return RoadDataset(root, name, mode=mode, img_size=input_size)


def set_model_return_aux(model: nn.Module, enabled: bool):
    if hasattr(model, "return_aux"):
        try:
            setattr(model, "return_aux", bool(enabled))
        except Exception:
            pass


def forward_train(model: nn.Module, imgs: torch.Tensor):
    if hasattr(model, "forward_train") and callable(getattr(model, "forward_train")):
        return model.forward_train(imgs)
    return model(imgs)


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


# =========================================================
# 3. Checkpoint helpers
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
        for meta in ("epoch", "metrics", "config", "best_val_iou", "epochs_without_improvement"):
            if meta in ckpt_obj:
                info[meta] = ckpt_obj[meta]
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return strip_module_prefix(ckpt_obj[key]), info
        return strip_module_prefix(ckpt_obj), info
    raise RuntimeError("checkpoint 格式无法解析")


def load_weights_flexible(model: nn.Module, weight_path: str, strict: bool = False, skip_mismatch: bool = True, log_file=None, prefix: str = ""):
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state, info = extract_state_dict(ckpt)
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
    msg = (
        f"{prefix}加载权重: {weight_path}\n"
        f"{prefix}  loaded={len(state)} missing={len(missing)} unexpected={len(unexpected)} skipped_mismatch={len(skipped)}\n"
    )
    log_print(msg, log_file)
    if skipped:
        log_print(f"{prefix}  skipped 示例: {skipped[:10]}\n", log_file)
    return {"missing": list(missing), "unexpected": list(unexpected), "skipped": skipped, "info": info}


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
        dims = tuple(range(1, probs.dim()))
        inter = (probs * targets).sum(dim=dims)
        union = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
        return self.bce_weight * bce + self.dice_weight * dice


def soft_boundary(mask: torch.Tensor, k: int = 3):
    mask = mask.float()
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
    ero = -F.max_pool2d(-mask, kernel_size=k, stride=1, padding=k // 2)
    return (dil - ero).clamp(0, 1)


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    if img.shape[1] != 1:
        return torch.cat([soft_erode(img[:, i:i + 1]) for i in range(img.shape[1])], dim=1)
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skeletonize(img: torch.Tensor, iters: int = 10) -> torch.Tensor:
    img = img.clamp(0, 1)
    skel = F.relu(img - soft_open(img))
    for _ in range(int(iters)):
        img = soft_erode(img)
        opened = soft_open(img)
        delta = F.relu(img - opened)
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0, 1)


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor, iters: int = 10, eps: float = 1e-6) -> torch.Tensor:
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    pred = torch.sigmoid(logits)
    target = target.float().clamp(0, 1)
    skel_pred = soft_skeletonize(pred, iters=iters)
    skel_true = soft_skeletonize(target, iters=iters)
    dims = tuple(range(1, pred.dim()))
    tprec = (skel_pred * target).sum(dim=dims) / (skel_pred.sum(dim=dims) + eps)
    tsens = (skel_true * pred).sum(dim=dims) / (skel_true.sum(dim=dims) + eps)
    cl = (2.0 * tprec * tsens + eps) / (tprec + tsens + eps)
    return (1.0 - cl).mean()


def find_aux_logits(outputs: Dict[str, Any]) -> List[Tuple[str, torch.Tensor]]:
    aux = []
    for key in ("logit16", "logit8", "logit4", "logit2", "aux16", "aux8", "aux4", "aux2"):
        if key in outputs and torch.is_tensor(outputs[key]):
            aux.append((key, outputs[key]))
    for key in ("aux_logits", "aux_outputs", "deep_supervision"):
        if key in outputs and isinstance(outputs[key], (list, tuple)):
            for i, t in enumerate(outputs[key]):
                if torch.is_tensor(t):
                    aux.append((f"{key}{i}", t))
    return aux


def aux_weight_for_key(key: str, loss_cfg: Dict[str, Any]) -> float:
    key_l = key.lower()
    if "16" in key_l:
        return float(loss_cfg.get("aux16_weight", loss_cfg.get("aux_weight", 0.15)))
    if "8" in key_l:
        return float(loss_cfg.get("aux8_weight", loss_cfg.get("aux_weight", 0.25)))
    if "4" in key_l:
        return float(loss_cfg.get("aux4_weight", loss_cfg.get("aux_weight", 0.15)))
    if "2" in key_l:
        return float(loss_cfg.get("aux2_weight", loss_cfg.get("aux_weight", 0.08)))
    return float(loss_cfg.get("aux_weight", 0.10))


def find_prior_logits(outputs: Dict[str, Any]) -> List[Tuple[str, torch.Tensor]]:
    priors = []
    for k, v in outputs.items():
        lk = k.lower()
        if torch.is_tensor(v) and "prior" in lk and "logit" in lk and v.dim() == 4:
            priors.append((k, v))
    return priors


def cs_loss(outputs, masks, criterion: nn.Module, cfg: Dict[str, Any]):
    if not isinstance(outputs, dict):
        loss = criterion(extract_logits(outputs), masks)
        return loss, {"main": tensor_to_float(loss), "total": tensor_to_float(loss)}

    loss_cfg = cfg.get("loss", {})
    logits = extract_logits(outputs)
    main = criterion(logits, masks)
    total = main
    parts = {"main": tensor_to_float(main)}

    for key, aux_logit in find_aux_logits(outputs):
        w = aux_weight_for_key(key, loss_cfg)
        if w <= 0:
            continue
        l = criterion(aux_logit, masks)
        total = total + w * l
        parts[key] = tensor_to_float(l)

    prior_weight = float(loss_cfg.get("prior_weight", 0.0))
    priors = find_prior_logits(outputs)
    if prior_weight > 0 and priors:
        ploss = 0.0
        for _, p in priors:
            y = F.interpolate(masks.float(), size=p.shape[-2:], mode="nearest")
            ploss = ploss + criterion(p, y)
        ploss = ploss / max(len(priors), 1)
        total = total + prior_weight * ploss
        parts["structure_prior"] = tensor_to_float(ploss)

    boundary_weight = float(loss_cfg.get("boundary_weight", 0.0))
    if boundary_weight > 0:
        k = int(loss_cfg.get("boundary_kernel", 3))
        with torch.amp.autocast("cuda", enabled=False) if logits.is_cuda else nullcontext():
            target_edge = soft_boundary(masks.float(), k=k)
            pred_prob = torch.sigmoid(logits.float())
            pred_edge = soft_boundary(pred_prob, k=k).clamp(1e-5, 1.0 - 1e-5)
            pred_edge_logits = torch.logit(pred_edge)
            bl = F.binary_cross_entropy_with_logits(pred_edge_logits, target_edge.float())
        total = total + boundary_weight * bl
        parts["boundary"] = tensor_to_float(bl)

    cldice_weight = float(loss_cfg.get("cldice_weight", 0.0))
    if cldice_weight > 0:
        cl_iters = int(loss_cfg.get("cldice_iters", 10))
        with torch.amp.autocast("cuda", enabled=False) if logits.is_cuda else nullcontext():
            cl = soft_cldice_loss(logits.float(), masks.float(), iters=cl_iters)
        total = total + cldice_weight * cl
        parts["cldice"] = tensor_to_float(cl)

    parts["total"] = tensor_to_float(total)
    return total, parts


@torch.no_grad()
def compute_counts_from_logits(logits: torch.Tensor, masks: torch.Tensor, threshold: float):
    if logits.shape[-2:] != masks.shape[-2:]:
        logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
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
# 5. Diagnostics
# =========================================================
@torch.no_grad()
def collect_output_stats(outputs, masks=None, threshold: float = 0.5) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if not isinstance(outputs, dict):
        logits = extract_logits(outputs).detach().float()
    else:
        for k, v in outputs.items():
            lk = k.lower()
            if torch.is_tensor(v) and v.numel() <= 16 and any(s in lk for s in ("alpha", "gate", "weight", "reliability", "lambda")):
                if v.dim() <= 2:
                    stats[k] = tensor_to_float(v)
            if torch.is_tensor(v) and "prior" in lk and "logit" in lk and v.dim() == 4:
                p = torch.sigmoid(v.detach().float())
                stats[f"{k}_mean"] = float(p.mean().cpu().item())
                stats[f"{k}_std"] = float(p.std().cpu().item())
            if torch.is_tensor(v) and "reliability" in lk and v.dim() == 4:
                rv = v.detach().float()
                stats[f"{k}_mean"] = float(rv.mean().cpu().item())
                stats[f"{k}_std"] = float(rv.std().cpu().item())
            if torch.is_tensor(v) and "disagreement" in lk and v.dim() == 4:
                dv = v.detach().float()
                stats[f"{k}_mean"] = float(dv.mean().cpu().item())
                stats[f"{k}_std"] = float(dv.std().cpu().item())
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
    stats["pred_structure_ratio"] = float((prob > threshold).float().mean().cpu().item())

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
        stats["target_structure_ratio"] = float(m.mean().cpu().item())
        stats["pred_target_gap"] = stats["pred_structure_ratio"] - stats["target_structure_ratio"]
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


def grad_diagnostics_from_params(params: Iterable[torch.nn.Parameter]) -> Dict[str, float]:
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
        g = p.grad.detach().float()
        norm = float(g.norm(2).cpu().item())
        total_sq += norm * norm
        if norm == 0.0:
            zero_grad_tensors += 1
        with_grad_tensors += 1
        ag = g.abs()
        max_abs = max(max_abs, float(ag.max().cpu().item()))
        abs_sum += float(ag.sum().cpu().item())
        elem_count += int(g.numel())

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
    for i, g in enumerate(optimizer.param_groups):
        name = str(g.get("name", f"group{i}"))
        diag = grad_diagnostics_from_params(g.get("params", []))
        out[f"grad_{name}"] = diag["norm"]
        out[f"grad_{name}_mean_abs"] = diag["mean_abs"]
        out[f"grad_{name}_max_abs"] = diag["max_abs"]
        out[f"grad_{name}_none_grad_tensors"] = diag["none_grad_tensors"]
        out[f"grad_{name}_zero_grad_tensors"] = diag["zero_grad_tensors"]
    return out


def param_norm_by_optimizer_groups(optimizer) -> Dict[str, float]:
    out = {}
    for i, g in enumerate(optimizer.param_groups):
        name = str(g.get("name", f"group{i}"))
        total_sq = 0.0
        abs_sum = 0.0
        elem_count = 0
        max_abs = 0.0
        for p in g.get("params", []):
            if p is None or not p.requires_grad:
                continue
            v = p.detach().float()
            n = float(v.norm(2).cpu().item())
            total_sq += n * n
            av = v.abs()
            abs_sum += float(av.sum().cpu().item())
            elem_count += int(v.numel())
            max_abs = max(max_abs, float(av.max().cpu().item()))
        out[f"param_{name}_norm"] = math.sqrt(total_sq)
        out[f"param_{name}_mean_abs"] = abs_sum / max(elem_count, 1)
        out[f"param_{name}_max_abs"] = max_abs
    return out


def format_stats(stats: Dict[str, float], max_items: int = 24) -> str:
    if not stats:
        return ""
    priority = [
        "prob_mean", "prob_std", "pred_structure_ratio", "target_structure_ratio", "pred_target_gap",
        "fg_prob_mean", "bg_prob_mean", "fg_bg_prob_gap", "fp_ratio", "fn_ratio", "tp_ratio",
        "reliability4_mean", "reliability8_mean", "reliability16_mean",
        "disagreement4_mean", "disagreement8_mean", "disagreement16_mean",
    ]
    keys = [k for k in priority if k in stats] + [k for k in sorted(stats.keys()) if k not in priority]
    return " | ".join([f"{k}={stats[k]:.4f}" for k in keys[:max_items]])


def diagnose_epoch(train_stats: Dict[str, float], val_stats: Dict[str, float], grad_stats: Dict[str, float]) -> str:
    notes = []
    pred_gap = val_stats.get("pred_target_gap", float("nan"))
    if math.isfinite(pred_gap):
        if pred_gap > 0.015:
            notes.append(f"曲线结构预测偏多(+{pred_gap * 100:.2f}%)，可能误检背景细线")
        elif pred_gap < -0.015:
            notes.append(f"曲线结构预测偏少({pred_gap * 100:.2f}%)，可能漏检细小分支")

    fg_bg_gap = val_stats.get("fg_bg_prob_gap", float("nan"))
    if math.isfinite(fg_bg_gap) and fg_bg_gap < 0.30:
        notes.append(f"前景/背景概率分离不足({fg_bg_gap:.3f})")

    fp = val_stats.get("fp_ratio", float("nan"))
    fn = val_stats.get("fn_ratio", float("nan"))
    if math.isfinite(fp) and math.isfinite(fn):
        if fp > fn * 1.25:
            notes.append("结构误检偏多：背景线状干扰可能较强")
        elif fn > fp * 1.25:
            notes.append("结构漏检偏多：细弱曲线或断裂区域可能没有恢复")

    for k, v in grad_stats.items():
        if k.startswith("grad_") and not any(s in k for s in ("mean_abs", "max_abs", "tensors")):
            if math.isfinite(v) and v < 1e-10:
                notes.append(f"{k.replace('grad_', '')} 梯度接近 0")
                break

    if not notes:
        return "Diagnosis[暂无明显异常]"
    uniq = []
    for x in notes:
        if x not in uniq:
            uniq.append(x)
    return "Diagnosis[" + "；".join(uniq[:8]) + "]"


def overfit_message(train_metrics: Dict[str, float], val_metrics: Dict[str, Any]) -> str:
    train_iou = float(train_metrics.get("iou", float("nan")))
    val_iou = float(val_metrics.get("iou", float("nan")))
    train_loss = float(train_metrics.get("loss", float("nan")))
    val_loss = float(val_metrics.get("loss", float("nan")))
    iou_gap = train_iou - val_iou
    loss_gap = val_loss - train_loss if math.isfinite(train_loss) and math.isfinite(val_loss) else float("nan")
    flag = "正常"
    if math.isfinite(iou_gap) and iou_gap > 0.08 and math.isfinite(loss_gap) and loss_gap > 0.03:
        flag = "可能过拟合"
    elif math.isfinite(train_iou) and train_iou < 0.20 and math.isfinite(val_iou) and val_iou < 0.20:
        flag = "可能欠拟合/未收敛"
    return f"FitCheck[{flag}]: train_IoU={train_iou * 100:.2f} val_IoU={val_iou * 100:.2f} gap={iou_gap * 100:.2f} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} val-train={loss_gap:.4f}"


# =========================================================
# 6. Optimizer / validation / saving
# =========================================================
def build_optimizer(model: nn.Module, config: Dict[str, Any], weight_decay: float):
    train_cfg = config.get("training", {})
    base_lr = float(train_cfg.get("lr", 1e-4))
    optimizer_name = str(train_cfg.get("optimizer", "AdamW")).lower()

    if hasattr(model, "get_param_groups") and callable(getattr(model, "get_param_groups")):
        try:
            raw_groups = model.get_param_groups(base_lr=base_lr, weight_decay=weight_decay)
        except TypeError:
            raw_groups = model.get_param_groups()

        param_groups = []
        used = set()
        for group in raw_groups:
            if "params" not in group:
                continue
            clean = []
            for p in group["params"]:
                if p is None or not p.requires_grad:
                    continue
                if id(p) in used:
                    continue
                clean.append(p)
                used.add(id(p))
            if not clean:
                continue
            new_group = dict(group)
            new_group["params"] = clean
            new_group.setdefault("lr", base_lr)
            new_group.setdefault("weight_decay", weight_decay)
            param_groups.append(new_group)

        missing = [p for p in model.parameters() if p.requires_grad and id(p) not in used]
        if missing:
            param_groups.append({"name": "others", "params": missing, "lr": base_lr, "weight_decay": weight_decay})

        if optimizer_name == "adam":
            optimizer = optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)
            opt_type = "Adam with model.get_param_groups()"
        else:
            optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
            opt_type = "AdamW with model.get_param_groups()"
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        param_groups = [{"name": "all_trainable", "params": params, "lr": base_lr, "weight_decay": weight_decay}]
        if optimizer_name == "adam":
            optimizer = optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)
            opt_type = "Adam"
        else:
            optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
            opt_type = "AdamW"

    lines = [" 优化器设置:\n", f" - 类型: {opt_type}\n", f" - 参数组数量: {len(optimizer.param_groups)}\n"]
    for r in optimizer_group_summary(optimizer):
        lines.append(f"   Group {r['idx']:02d} {r['name']:<18} lr={r['lr']:.3e} wd={r['weight_decay']:.3e} params={r['params'] / 1e6:.3f}M\n")
    lines.append("--------------------------------------------------\n")
    return optimizer, "".join(lines)


class ProfileWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return extract_logits(self.model(x))


def evaluate_model_complexity(model: nn.Module, device: torch.device, img_size, log_file=None, enabled: bool = True):
    total, trainable, frozen = count_params(model)
    params, flops, fps = total, float("nan"), float("nan")
    if not enabled:
        return params, flops, fps

    img_h, img_w = parse_input_size(img_size)
    dummy = torch.randn(1, 3, img_h, img_w, device=device)
    was_training = model.training
    model.eval()

    if profile is not None:
        try:
            model_copy = copy.deepcopy(model).to(device).eval()
            set_model_return_aux(model_copy, False)
            macs, thop_params = profile(ProfileWrapper(model_copy), inputs=(dummy,), verbose=False)
            flops = float(macs)
            params = int(thop_params)
            del model_copy
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            log_print(f"⚠️ THOP 复杂度统计失败，继续训练。原因: {repr(e)}\n", log_file)

    try:
        with torch.no_grad():
            for _ in range(10):
                _ = extract_logits(model(dummy))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(30):
                _ = extract_logits(model(dummy))
        if device.type == "cuda":
            torch.cuda.synchronize()
        fps = 30.0 / max(time.time() - t0, 1e-6)
    except Exception as e:
        log_print(f"⚠️ FPS 统计失败，继续训练。原因: {repr(e)}\n", log_file)

    model.train(was_training)
    return params, flops, fps


def write_model_summary(exp_dir: str, model: nn.Module, optimizer, config: Dict[str, Any]):
    total, trainable, frozen = count_params(model)
    with open(os.path.join(exp_dir, "model_summary.txt"), "w", encoding="utf-8") as f:
        f.write("========== Curvilinear Structure Segmentation Model Summary ==========\n")
        f.write(f"Model: {get_model_name(config)}\n")
        f.write(f"Total params:     {total / 1e6:.4f} M\n")
        f.write(f"Trainable params: {trainable / 1e6:.4f} M\n")
        f.write(f"Frozen params:    {frozen / 1e6:.4f} M\n\n")
        f.write("[Top-level modules]\n")
        for r in module_param_summary(model):
            f.write(f"{r['module']:<24} total={r['total'] / 1e6:8.4f}M trainable={r['trainable'] / 1e6:8.4f}M frozen={r['frozen'] / 1e6:8.4f}M\n")
        f.write("\n[Optimizer groups]\n")
        for r in optimizer_group_summary(optimizer):
            f.write(f"{r['idx']:02d} {r['name']:<20} params={r['params'] / 1e6:8.4f}M lr={r['lr']:.3e} wd={r['weight_decay']:.3e}\n")
        f.write("\n[Config]\n")
        f.write(json.dumps(config, indent=2, ensure_ascii=False))
        f.write("\n")


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, config, device, threshold=0.5, desc="Valid"):
    model.eval()
    amp_enabled = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
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
            full_loss, loss_parts = cs_loss(outputs, masks, criterion, config)

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


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best_val_iou, epochs_without_improvement, metrics, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_iou": best_val_iou,
        "epochs_without_improvement": epochs_without_improvement,
        "metrics": metrics,
        "config": config,
    }, path)


# =========================================================
# 7. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
    parser.add_argument("-r", "--resume", type=str, default=None, help="Path to latest_model.pth to resume")
    parser.add_argument("--finetune", type=str, default="", help="Only load model weights, not optimizer/scheduler/scaler")
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--no-skip-mismatch", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 3407))
    deterministic = bool(train_cfg.get("deterministic", True))
    set_seed(seed, deterministic=deterministic)

    if args.resume and os.path.isfile(args.resume):
        exp_dir = os.path.dirname(args.resume)
        log_file = open(os.path.join(exp_dir, "train.log"), "a", encoding="utf-8")
        resume_msg = f"\n{'=' * 60}\n🔌 触发断点续训，继续向原目录记录日志\n{'=' * 60}\n"
        log_print(resume_msg, log_file)
    else:
        exp_dir = create_experiment_dir(config)
        log_file = open(os.path.join(exp_dir, "train.log"), "w", encoding="utf-8")

    start_time_raw = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f" 实验开始，日志和权重保存在: {exp_dir}\n 训练开始时间: {start_time_str}\n", log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        log_print(f" 训练设备: CUDA | {torch.cuda.get_device_name(0)}\n", log_file)
    else:
        log_print(" 训练设备: CPU\n", log_file)

    ds_cfg = config["dataset"]
    dataset_name = ds_cfg["name"]
    input_size = ds_cfg.get("input_size", 1024)
    img_h, img_w = parse_input_size(input_size)
    if img_h % 16 != 0 or img_w % 16 != 0:
        raise ValueError(f"CS/RD/DINO 模型要求输入 H/W 能被 16 整除，当前 H={img_h}, W={img_w}")
    if img_h % 32 != 0 or img_w % 32 != 0:
        print(f"⚠️ 当前输入 H={img_h}, W={img_w} 不能被 32 整除，部分 CNN/decoder 可能需要插值对齐")
    train_dataset = build_dataset(config, mode="train")
    val_dataset = build_dataset(config, mode="val")

    n_workers = int(ds_cfg.get("num_workers", 8))
    batch_size = int(ds_cfg.get("batch_size", 2))
    val_batch_size = int(ds_cfg.get("val_batch_size", batch_size))
    drop_last = bool(ds_cfg.get("drop_last", False))
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, val_batch_size),
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
    )

    try:
        model = get_model(config["model"], img_size=input_size).to(device)
    except TypeError:
        # 兼容只接受 int img_size 的旧模型
        model = get_model(config["model"], img_size=max(img_h, img_w)).to(device)
    set_model_return_aux(model, True)

    if args.finetune:
        load_weights_flexible(
            model,
            args.finetune,
            strict=bool(args.strict_load),
            skip_mismatch=not bool(args.no_skip_mismatch),
            log_file=log_file,
            prefix="[Finetune] ",
        )
    elif config.get("finetune", {}).get("weights"):
        load_weights_flexible(
            model,
            config["finetune"]["weights"],
            strict=bool(config.get("finetune", {}).get("strict", False)),
            skip_mismatch=bool(config.get("finetune", {}).get("skip_mismatch", True)),
            log_file=log_file,
            prefix="[Finetune] ",
        )

    total, trainable, frozen = count_params(model)
    log_print("\n========== Curvilinear Structure Training ==========" + "\n", log_file)
    log_print(f"Dataset: {dataset_name} | train={len(train_dataset)} val={len(val_dataset)} | img={img_size}\n", log_file)
    log_print(f"Batch: train={batch_size} val={val_batch_size} workers={n_workers}\n", log_file)
    log_print(f"Model: {get_model_name(config)}\n", log_file)
    log_print(f"Params: total={total / 1e6:.2f}M trainable={trainable / 1e6:.2f}M frozen={frozen / 1e6:.2f}M\n", log_file)

    for r in module_param_summary(model):
        log_print(f"  {r['module']:<24} total={r['total'] / 1e6:7.3f}M trainable={r['trainable'] / 1e6:7.3f}M frozen={r['frozen'] / 1e6:7.3f}M\n", log_file)

    profile_enabled = bool(train_cfg.get("profile_model", True))
    params_prof, flops, fps = evaluate_model_complexity(model, device=device, img_size=input_size, log_file=log_file, enabled=profile_enabled)
    complexity_msg = (
        f"--------------------------------------------------\n"
        f"📊 模型复杂度 @ 输入尺寸: {img_h}x{img_w}\n"
        f"   - 参数量 Params:       {params_prof / 1e6:.2f} M\n"
        f"   - 浮点运算量 FLOPs:    {flops / 1e9:.2f} G\n" if math.isfinite(flops) else
        f"--------------------------------------------------\n📊 模型复杂度 @ 输入尺寸: {img_h}x{img_w}\n   - 参数量 Params:       {params_prof / 1e6:.2f} M\n   - 浮点运算量 FLOPs:    统计失败/未启用\n"
    )
    complexity_msg += f"   - 推理速度 FPS:        {fps:.2f} 张/秒\n" if math.isfinite(fps) else "   - 推理速度 FPS:        统计失败/未启用\n"
    complexity_msg += "--------------------------------------------------\n"
    log_print(complexity_msg, log_file)

    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    lr_factor = float(train_cfg.get("lr_factor", 0.2))
    lr_patience = int(train_cfg.get("lr_patience", 5))
    min_lr = float(train_cfg.get("min_lr", 1e-7))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 15))
    epochs = int(train_cfg.get("epochs", 200))
    threshold = float(train_cfg.get("threshold", 0.5))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    log_grad_norm = bool(config.get("logging", {}).get("log_grad_norm", True))

    optimizer, optimizer_msg = build_optimizer(model, config, weight_decay)
    log_print(optimizer_msg, log_file)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
        min_lr=min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    loss_cfg = config.get("loss", {})
    criterion = BCEDiceLoss(
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
    ).to(device)

    start_epoch = 1
    best_val_iou = 0.0
    best_epoch = -1
    epochs_without_improvement = 0

    if args.resume:
        if os.path.isfile(args.resume):
            log_print(f"🔄 发现断点文件，正在恢复: {args.resume}\n", log_file)
            ckpt = torch.load(args.resume, map_location=device, weights_only=False)
            state, _ = extract_state_dict(ckpt)
            model.load_state_dict(state, strict=True)
            if ckpt.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("scaler_state_dict") is not None:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_iou = float(ckpt.get("best_val_iou", 0.0))
            epochs_without_improvement = int(ckpt.get("epochs_without_improvement", 0))
            best_epoch = int(ckpt.get("metrics", {}).get("best_epoch", -1)) if isinstance(ckpt.get("metrics", {}), dict) else -1
            log_print(f"🔄 恢复完成: start_epoch={start_epoch}, best_val_iou={best_val_iou:.4f}, 已连续未提升={epochs_without_improvement} 轮\n", log_file)
        else:
            log_print(f"⚠️ 指定 resume 文件不存在，将从头训练: {args.resume}\n", log_file)

    hyper_msg = (
        f" 训练超参数设置:\n"
        f" - 主学习率 lr: {float(train_cfg.get('lr', 1e-4))}\n"
        f" - 权重衰减 Weight Decay: {weight_decay}\n"
        f" - AMP: {amp_enabled}\n"
        f" - Grad Clip: {grad_clip}\n"
        f" - 阈值 threshold: {threshold}\n"
        f" - 学习率调度: ReduceLROnPlateau(factor={lr_factor}, patience={lr_patience}, min_lr={min_lr})\n"
        f" - 早停机制: 连续 {early_stop_patience} 轮 Val IoU 未提升则停止\n"
        f"--------------------------------------------------\n"
    )
    log_print(hyper_msg, log_file)
    write_model_summary(exp_dir, model, optimizer, config)

    metrics_csv = os.path.join(exp_dir, "metrics.csv")
    branch_csv = os.path.join(exp_dir, "branch_stats.csv")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        set_model_return_aux(model, True)
        train_loss = 0.0
        TP = FP = FN = 0.0
        loss_sum: Dict[str, float] = {}
        loss_count: Dict[str, int] = {}
        stat_sum: Dict[str, float] = {}
        stat_count: Dict[str, int] = {}
        grad_sum: Dict[str, float] = {}
        grad_count: Dict[str, int] = {}

        train_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}] Train", leave=False)
        for batch_data in train_tqdm:
            imgs, masks = unwrap_batch(batch_data)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp_enabled):
                outputs = forward_train(model, imgs)
                logits = extract_logits(outputs)
                loss, loss_parts = cs_loss(outputs, masks, criterion, config)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if log_grad_norm:
                update_sum_dict(grad_sum, grad_count, grad_norm_by_optimizer_groups(optimizer))

            total_norm_value = float("nan")
            if grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                total_norm_value = tensor_to_float(total_norm)
                update_sum_dict(grad_sum, grad_count, {"grad_total_preclip": total_norm_value})

            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            update_sum_dict(loss_sum, loss_count, loss_parts)
            update_sum_dict(stat_sum, stat_count, collect_output_stats(outputs, masks, threshold=threshold))

            with torch.no_grad():
                tp, fp, fn, _ = compute_counts_from_logits(logits.detach(), masks, threshold)
                TP += tp
                FP += fp
                FN += fn

            train_tqdm.set_postfix({
                "loss": f"{loss.item():.4f}",
                "main": f"{loss_parts.get('main', 0.0):.4f}",
                "g": f"{total_norm_value:.2f}" if math.isfinite(total_norm_value) else "nan",
            })

        avg_train_loss = train_loss / max(len(train_loader), 1)
        train_metrics = metrics_from_counts(TP, FP, FN)
        train_loss_parts = avg_dict(loss_sum, loss_count)
        train_stats = avg_dict(stat_sum, stat_count)
        train_grad_stats = avg_dict(grad_sum, grad_count)
        param_stats = param_norm_by_optimizer_groups(optimizer)

        raw = validate_one_epoch(model, val_loader, criterion, config, device, threshold=threshold, desc=f"Epoch [{epoch}/{epochs}] Valid")
        scheduler.step(raw["loss"])

        is_best = raw["iou"] > best_val_iou
        if is_best:
            best_val_iou = raw["iou"]
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        lr_now = optimizer.param_groups[0]["lr"]
        train_epoch_metrics = {"loss": avg_train_loss, **train_metrics}
        msg = (
            f"Epoch [{epoch:03d}/{epochs}] | lr={lr_now:.2e} | "
            f"Train Loss={avg_train_loss:.4f} IoU={train_metrics['iou'] * 100:.2f}% P={train_metrics['precision'] * 100:.2f}% R={train_metrics['recall'] * 100:.2f}% F1={train_metrics['f1'] * 100:.2f}% | "
            f"Val MainLoss={raw['loss']:.4f} FullLoss={raw['full_loss']:.4f} IoU={raw['iou'] * 100:.2f}% P={raw['precision'] * 100:.2f}% R={raw['recall'] * 100:.2f}% F1={raw['f1'] * 100:.2f}% | "
            f"BestThr={raw['best_sweep']['threshold']:.2f}/IoU={raw['best_sweep']['iou'] * 100:.2f}% | "
            f"BestIoU={best_val_iou * 100:.2f}%@epoch{best_epoch}\n"
        )
        log_print(msg, log_file)
        log_print("  " + overfit_message(train_epoch_metrics, raw) + "\n", log_file)
        log_print("  TrainStruct: " + format_stats(train_stats) + "\n", log_file)
        log_print("  ValStruct:   " + format_stats(raw.get("branch_stats", {})) + "\n", log_file)
        log_print("  " + diagnose_epoch(train_stats, raw.get("branch_stats", {}), train_grad_stats) + "\n", log_file)

        if train_grad_stats:
            main_grad_keys = [k for k in sorted(train_grad_stats.keys()) if k.startswith("grad_") and not any(s in k for s in ("mean_abs", "max_abs", "tensors"))]
            grad_str = " | ".join([f"{k}={train_grad_stats[k]:.6f}" for k in main_grad_keys])
            log_print("  GradNorm: " + grad_str + "\n", log_file)

        lrs = {g.get("name", f"group{i}"): float(g.get("lr", 0.0)) for i, g in enumerate(optimizer.param_groups)}
        lr_str = " | ".join([f"{k}={v:.2e}" for k, v in lrs.items()])
        log_print("  LRGroups: " + lr_str + "\n", log_file)

        save_metrics = copy.deepcopy(raw)
        save_metrics["best_val_iou"] = best_val_iou
        save_metrics["best_epoch"] = best_epoch
        save_metrics["train_epoch"] = train_epoch_metrics
        save_metrics["train_loss_parts"] = train_loss_parts
        save_metrics["train_structure_stats"] = train_stats
        save_metrics["train_grad_stats"] = train_grad_stats
        save_metrics["param_stats"] = param_stats

        latest_path = os.path.join(exp_dir, "latest_model.pth")
        save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, best_val_iou, epochs_without_improvement, save_metrics, config)

        if is_best:
            best_path = os.path.join(exp_dir, "best_model.pth")
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_val_iou, epochs_without_improvement, save_metrics, config)
            with open(os.path.join(exp_dir, "best_summary.json"), "w", encoding="utf-8") as f:
                json.dump(save_metrics, f, indent=2, ensure_ascii=False)
            log_print(f"      ✅ 已保存最佳权重！Val IoU={best_val_iou * 100:.2f}%\n", log_file)
        else:
            log_print(f"      ⚠️ 连续 {epochs_without_improvement} 轮没有提升了。\n", log_file)

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": avg_train_loss,
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
            "best_val_iou": best_val_iou,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        }
        row.update({f"train_loss_{k}": v for k, v in train_loss_parts.items()})
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"val_{k}": v for k, v in raw.get("branch_stats", {}).items()})
        row.update({f"val_{k}": v for k, v in raw.get("threshold_sweep", {}).items()})
        row.update({f"grad_{k}": v for k, v in train_grad_stats.items()})
        row.update({f"param_{k}": v for k, v in param_stats.items()})
        append_csv(metrics_csv, row)

        branch_row = {"epoch": epoch}
        branch_row.update({f"train_{k}": v for k, v in train_stats.items()})
        branch_row.update({f"val_{k}": v for k, v in raw.get("branch_stats", {}).items()})
        branch_row.update({f"grad_{k}": v for k, v in train_grad_stats.items()})
        append_csv(branch_csv, branch_row)

        if epochs_without_improvement >= early_stop_patience:
            log_print(f"🚫 连续 {early_stop_patience} 轮 Val IoU 未提升，触发早停机制，训练提前结束！最佳 epoch={best_epoch}，最佳 Val IoU={best_val_iou * 100:.2f}%\n", log_file)
            break

    end_time_raw = time.time()
    end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration_sec = end_time_raw - start_time_raw
    hours, rem = divmod(duration_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    duration_str = f"{int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒"

    end_msg = (
        f"--------------------------------------------------\n"
        f"训练结束时间: {end_time_str}\n"
        f"整个实验总耗时: {duration_str}\n"
        f"最佳 Val IoU: {best_val_iou * 100:.2f}% @ epoch {best_epoch}\n"
        f"最终连续未提升轮数: {epochs_without_improvement}\n"
        f"🎉 实验完成！前往 {exp_dir} 查看结果。\n"
        f"--------------------------------------------------\n"
    )
    log_print(end_msg, log_file)
    log_file.close()


if __name__ == "__main__":
    main()
