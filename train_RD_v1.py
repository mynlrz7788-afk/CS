# -*- coding: utf-8 -*-
"""Verbose train script for RD_v1.

特点：
- 支持 RD_v1 返回 dict；
- 主损失 + 多尺度 decoder 辅助监督 + road prior 监督 + AMP-safe boundary loss；
- 支持 model.get_param_groups(base_lr, weight_decay)；
- 支持 EMA、AMP、梯度裁剪、ReduceLROnPlateau；
- best_model.pth 按 raw validation IoU 保存，best_ema_model.pth 按 EMA IoU 保存；
- 额外保存 metrics.csv / branch_stats.csv / param_groups.txt / model_summary.txt / best_summary.json；
- 额外打印 train-vs-val gap、分支 alpha/gate、decoder fusion weight、prior/logit 分布、各参数组 grad norm。
"""

import os
import csv
import json
import time
import math
import random
import argparse
import copy
from typing import Any, Dict, Tuple, Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.road_dataset import RoadDataset
from dataloaders.drive_dataset import DRIVEDataset
from models import get_model


# =========================================================
# 1. Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================================================
# 2. Dataset / helpers
# =========================================================
def _unpack_batch(batch_data):
    if isinstance(batch_data, (list, tuple)):
        if len(batch_data) >= 2:
            return batch_data[0], batch_data[1]
    raise RuntimeError(f"无法解析 batch: {type(batch_data)}")


def build_dataset(config, mode: str):
    name = config["dataset"]["name"]
    root = config["dataset"]["root_path"]
    img_size = int(config["dataset"].get("input_size", 1024))
    if name.lower() == "drive":
        return DRIVEDataset(root, name, mode=mode, img_size=img_size)
    return RoadDataset(root, name, mode=mode, img_size=img_size)


def create_exp_dir(config: Dict[str, Any]):
    dataset_name = config["dataset"]["name"]
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("name") or model_cfg.get("model_name") or config.get("model_name") or "RD_v1"
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_dir = os.path.join("saved_runs", dataset_name, f"{model_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    return exp_dir


def log_print(msg: str, log_file=None):
    print(msg, end="")
    if log_file is not None:
        log_file.write(msg)
        log_file.flush()


def extract_logits(outputs):
    if isinstance(outputs, torch.Tensor):
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
        if isinstance(v, (float, int)) and math.isfinite(float(v)):
            sum_dict[k] = sum_dict.get(k, 0.0) + float(v)
            count_dict[k] = count_dict.get(k, 0) + 1


def avg_dict(sum_dict: Dict[str, float], count_dict: Dict[str, int]) -> Dict[str, float]:
    return {k: sum_dict[k] / max(count_dict.get(k, 1), 1) for k in sum_dict.keys()}


def append_csv(path: str, row: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.isfile(path)
    # 如果后续 row 出现新增字段，重写整个 CSV 会复杂；这里固定当前 row 的 key 顺序。
    fieldnames = list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def flatten_metrics(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            out.update(flatten_metrics(f"{prefix}{k}_", v))
        else:
            out[f"{prefix}{k}"] = v
    return out


# =========================================================
# 3. Loss / metrics
# =========================================================
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)

    def forward(self, logits, targets):
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


def rd_v1_loss(outputs, masks, criterion: nn.Module, cfg: Dict[str, Any]):
    if not isinstance(outputs, dict):
        loss = criterion(outputs, masks)
        return loss, {"main": loss.detach().item()}

    loss_cfg = cfg.get("loss", {})
    w_aux8 = float(loss_cfg.get("aux8_weight", 0.25))
    w_aux16 = float(loss_cfg.get("aux16_weight", 0.15))
    w_aux4 = float(loss_cfg.get("aux4_weight", 0.20))
    w_aux2 = float(loss_cfg.get("aux2_weight", 0.10))
    w_prior = float(loss_cfg.get("prior_weight", 0.05))
    w_boundary = float(loss_cfg.get("boundary_weight", 0.03))

    main = criterion(outputs["logits"], masks)
    total = main
    parts = {"main": main.detach().item()}

    aux_terms = [
        ("logit8", w_aux8),
        ("logit16", w_aux16),
        ("logit4", w_aux4),
        ("logit2", w_aux2),
    ]
    for key, w in aux_terms:
        if w > 0 and key in outputs:
            l = criterion(outputs[key], masks)
            total = total + w * l
            parts[key] = l.detach().item()

    prior_loss = 0.0
    prior_count = 0
    for key in ("prior4_logits", "prior8_logits", "prior16_logits"):
        if key in outputs:
            p = outputs[key]
            y = F.interpolate(masks.float(), size=p.shape[-2:], mode="nearest")
            prior_loss = prior_loss + criterion(p, y)
            prior_count += 1
    if prior_count > 0:
        prior_loss = prior_loss / prior_count
        total = total + w_prior * prior_loss
        parts["prior"] = prior_loss.detach().item()

    if w_boundary > 0:
        # AMP/autocast-safe boundary loss.
        k = int(loss_cfg.get("boundary_kernel", 3))
        with torch.cuda.amp.autocast(enabled=False):
            target_edge = soft_boundary(masks.float(), k=k)
            pred_prob = torch.sigmoid(outputs["logits"].float())
            pred_edge = soft_boundary(pred_prob, k=k).clamp(1e-5, 1.0 - 1e-5)
            pred_edge_logits = torch.logit(pred_edge)
            boundary = F.binary_cross_entropy_with_logits(pred_edge_logits, target_edge.float())
        total = total + w_boundary * boundary
        parts["boundary"] = boundary.detach().item()

    parts["total"] = total.detach().item()
    return total, parts


@torch.no_grad()
def compute_metrics_from_logits(logits, masks, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    masks = masks.float()
    tp = (preds * masks).sum().item()
    fp = (preds * (1.0 - masks)).sum().item()
    fn = ((1.0 - preds) * masks).sum().item()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return precision, recall, f1, iou


@torch.no_grad()
def collect_output_stats(outputs, masks=None, threshold: float = 0.5) -> Dict[str, float]:
    stats: Dict[str, float] = {}
    if not isinstance(outputs, dict):
        return stats

    for k in ("dlaem8_alpha", "dlaem16_alpha", "dlaem8_gate_mean", "dlaem16_gate_mean"):
        if k in outputs:
            stats[k] = tensor_to_float(outputs[k])

    if "decoder_fuse_weights" in outputs:
        w = outputs["decoder_fuse_weights"].detach().float().flatten().cpu().tolist()
        for i, v in enumerate(w):
            stats[f"decoder_w{i}"] = float(v)

    for k in ("prior4_logits", "prior8_logits", "prior16_logits"):
        if k in outputs:
            p = torch.sigmoid(outputs[k].detach().float())
            stats[f"{k}_mean"] = float(p.mean().cpu().item())
            stats[f"{k}_std"] = float(p.std().cpu().item())

    logits = extract_logits(outputs).detach().float()
    prob = torch.sigmoid(logits)
    stats["prob_mean"] = float(prob.mean().cpu().item())
    stats["prob_std"] = float(prob.std().cpu().item())
    stats["pred_ratio"] = float((prob > threshold).float().mean().cpu().item())
    if masks is not None:
        stats["target_ratio"] = float(masks.detach().float().mean().cpu().item())
    return stats


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
        imgs, masks = _unpack_batch(batch_data)
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(imgs)
            logits = extract_logits(outputs)
            main_loss = criterion(logits, masks)
            full_loss, loss_parts = rd_v1_loss(outputs, masks, criterion, config)
        val_main_loss += main_loss.item()
        val_full_loss += full_loss.item()
        update_sum_dict(loss_sum, loss_count, loss_parts)
        update_sum_dict(stat_sum, stat_count, collect_output_stats(outputs, masks, threshold=threshold))

        probs = torch.sigmoid(logits)
        masks_f = masks.float()
        preds = (probs > threshold).float()
        TP += (preds * masks_f).sum().item()
        FP += (preds * (1.0 - masks_f)).sum().item()
        FN += ((1.0 - preds) * masks_f).sum().item()

        for t in sweep_values:
            key = str(t)
            p = (probs > float(t)).float()
            sweep_counts[key]["TP"] += (p * masks_f).sum().item()
            sweep_counts[key]["FP"] += (p * (1.0 - masks_f)).sum().item()
            sweep_counts[key]["FN"] += ((1.0 - p) * masks_f).sum().item()

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)

    sweep_metrics = {}
    best_sweep = {"threshold": threshold, "iou": iou, "f1": f1}
    for key, c in sweep_counts.items():
        p = c["TP"] / (c["TP"] + c["FP"] + 1e-6)
        r = c["TP"] / (c["TP"] + c["FN"] + 1e-6)
        f = 2.0 * p * r / (p + r + 1e-6)
        j = c["TP"] / (c["TP"] + c["FP"] + c["FN"] + 1e-6)
        sweep_metrics[f"thr{key}_iou"] = j
        sweep_metrics[f"thr{key}_f1"] = f
        if j > best_sweep["iou"]:
            best_sweep = {"threshold": float(key), "iou": j, "f1": f}

    return {
        "loss": val_main_loss / max(len(loader), 1),
        "full_loss": val_full_loss / max(len(loader), 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "loss_parts": avg_dict(loss_sum, loss_count),
        "branch_stats": avg_dict(stat_sum, stat_count),
        "threshold_sweep": sweep_metrics,
        "best_sweep": best_sweep,
    }


# =========================================================
# 4. EMA / optimizer / diagnostics
# =========================================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd and v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            elif k in msd:
                v.copy_(msd[k])


def build_optimizer(model, config):
    train_cfg = config.get("training", {})
    base_lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    if hasattr(model, "get_param_groups"):
        groups = model.get_param_groups(base_lr=base_lr, weight_decay=weight_decay)
        return torch.optim.AdamW(groups, lr=base_lr, weight_decay=weight_decay)
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=base_lr, weight_decay=weight_decay)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, total - trainable


def module_param_summary(model) -> List[Dict[str, Any]]:
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


def grad_norm_by_optimizer_groups(optimizer) -> Dict[str, float]:
    out = {}
    for g in optimizer.param_groups:
        name = g.get("name", "group")
        out[f"grad_{name}"] = grad_norm_from_params(g.get("params", []))
    return out


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, metrics, config):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "metrics": metrics,
        "config": config,
    }, path)


def write_model_summary(exp_dir: str, model, optimizer, config: Dict[str, Any]):
    path = os.path.join(exp_dir, "model_summary.txt")
    total, trainable, frozen = count_params(model)
    with open(path, "w", encoding="utf-8") as f:
        f.write("========== RD_v1 Model Summary ==========\n")
        f.write(f"Total params:     {total/1e6:.4f} M\n")
        f.write(f"Trainable params: {trainable/1e6:.4f} M\n")
        f.write(f"Frozen params:    {frozen/1e6:.4f} M\n\n")
        f.write("[Top-level modules]\n")
        for r in module_param_summary(model):
            f.write(f"{r['module']:<18} total={r['total']/1e6:8.4f}M trainable={r['trainable']/1e6:8.4f}M frozen={r['frozen']/1e6:8.4f}M\n")
        f.write("\n[Optimizer groups]\n")
        for r in optimizer_group_summary(optimizer):
            f.write(f"{r['idx']:02d} {r['name']:<16} params={r['params']/1e6:8.4f}M lr={r['lr']:.3e} wd={r['weight_decay']:.3e}\n")
        f.write("\n[Config model]\n")
        f.write(json.dumps(config.get("model", {}), indent=2, ensure_ascii=False))
        f.write("\n")


def format_branch_stats(stats: Dict[str, float]) -> str:
    keys = [
        "dlaem8_alpha", "dlaem16_alpha", "dlaem8_gate_mean", "dlaem16_gate_mean",
        "decoder_w0", "decoder_w1", "decoder_w2", "decoder_w3",
        "pred_ratio", "target_ratio", "prob_mean",
        "prior4_logits_mean", "prior8_logits_mean", "prior16_logits_mean",
    ]
    parts = []
    for k in keys:
        if k in stats:
            parts.append(f"{k}={stats[k]:.4f}")
    return " | ".join(parts)


def overfit_message(train_metrics: Dict[str, float], val_metrics: Dict[str, Any]) -> str:
    train_iou = train_metrics.get("iou", float("nan"))
    val_iou = val_metrics.get("iou", float("nan"))
    train_main = train_metrics.get("loss_main", float("nan"))
    val_main = val_metrics.get("loss", float("nan"))
    iou_gap = train_iou - val_iou
    loss_gap = val_main - train_main if math.isfinite(train_main) and math.isfinite(val_main) else float("nan")
    flag = "正常"
    if math.isfinite(iou_gap) and iou_gap > 0.08 and math.isfinite(loss_gap) and loss_gap > 0.03:
        flag = "可能过拟合"
    elif math.isfinite(train_iou) and train_iou < 0.2 and math.isfinite(val_iou) and val_iou < 0.2:
        flag = "可能欠拟合/未收敛"
    return f"OverfitCheck[{flag}]: train_IoU={train_iou*100:.2f} val_IoU={val_iou*100:.2f} gap={iou_gap*100:.2f} | train_main={train_main:.4f} val_main={val_main:.4f} val-train={loss_gap:.4f}"


# =========================================================
# 5. Main train
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    set_seed(int(config.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = create_exp_dir(config)
    log_path = os.path.join(exp_dir, "train.log")
    metrics_csv = os.path.join(exp_dir, "metrics.csv")
    branch_csv = os.path.join(exp_dir, "branch_stats.csv")
    log_file = open(log_path, "w", encoding="utf-8")

    train_set = build_dataset(config, mode="train")
    val_set = build_dataset(config, mode="val")
    batch_size = int(config["dataset"].get("batch_size", 2))
    num_workers = int(config["dataset"].get("num_workers", 8))
    g = torch.Generator()
    g.manual_seed(int(config.get("seed", 42)))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, int(config["dataset"].get("val_batch_size", batch_size))),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    img_size = int(config["dataset"].get("input_size", 1024))
    model = get_model(config["model"], img_size=img_size).to(device)
    total, trainable, frozen = count_params(model)

    log_print(f"\n========== RD_v1 Training ==========" "\n", log_file)
    log_print(f"Exp dir: {exp_dir}\n", log_file)
    log_print(f"Device: {device}\n", log_file)
    if torch.cuda.is_available():
        log_print(f"CUDA: {torch.cuda.get_device_name(0)} | capability={torch.cuda.get_device_capability(0)}\n", log_file)
    log_print(f"Dataset: {config['dataset']['name']} | train={len(train_set)} val={len(val_set)} | img={img_size}\n", log_file)
    log_print(f"Batch: train={batch_size} val={max(1, int(config['dataset'].get('val_batch_size', batch_size)))} workers={num_workers}\n", log_file)
    log_print(f"Params: total={total/1e6:.2f}M trainable={trainable/1e6:.2f}M frozen={frozen/1e6:.2f}M\n", log_file)
    max_trainable = float(config["model"].get("max_trainable_params_m", config["model"].get("max_total_params_m", 33.0))) * 1e6
    if trainable > max_trainable:
        log_print("⚠️ 可训练参数量超过配置上限；冻结 DINO 参数不计入该上限。请检查 adapter_bottleneck/encoder_channels/decoder_channels。\n", log_file)
    else:
        log_print(f"Trainable-param cap: {max_trainable/1e6:.2f}M，当前可训练参数满足要求；冻结 DINO 不计入上限。\n", log_file)

    log_print("\n[Top-level parameter summary]\n", log_file)
    for r in module_param_summary(model):
        log_print(f"  {r['module']:<18} total={r['total']/1e6:7.3f}M trainable={r['trainable']/1e6:7.3f}M frozen={r['frozen']/1e6:7.3f}M\n", log_file)

    criterion = BCEDiceLoss(
        bce_weight=float(config.get("loss", {}).get("bce_weight", 0.5)),
        dice_weight=float(config.get("loss", {}).get("dice_weight", 0.5)),
    ).to(device)

    optimizer = build_optimizer(model, config)
    train_cfg = config.get("training", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg.get("lr_factor", 0.5)),
        patience=int(train_cfg.get("lr_patience", 5)),
        min_lr=float(train_cfg.get("min_lr", 1e-7)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)))

    log_print("\n[Optimizer groups]\n", log_file)
    for r in optimizer_group_summary(optimizer):
        log_print(f"  {r['idx']:02d} {r['name']:<16} params={r['params']/1e6:7.3f}M lr={r['lr']:.3e} wd={r['weight_decay']:.3e}\n", log_file)
    write_model_summary(exp_dir, model, optimizer, config)

    ema = None
    if bool(train_cfg.get("ema_enabled", True)):
        ema = ModelEMA(model, decay=float(train_cfg.get("ema_decay", 0.999)))

    start_epoch = 1
    best_iou = -1.0
    best_ema_iou = -1.0
    best_epoch = -1
    best_ema_epoch = -1
    no_improve = 0

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
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
    amp_enabled = bool(train_cfg.get("amp", True))
    log_cfg = config.get("logging", {})
    log_grad_norm = bool(log_cfg.get("log_grad_norm", True))

    for epoch in range(start_epoch, epochs + 1):
        model.train()
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
            imgs, masks = _unpack_batch(batch_data)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(imgs)
                logits = extract_logits(outputs)
                loss, loss_parts = rd_v1_loss(outputs, masks, criterion, config)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if log_grad_norm:
                gstats = grad_norm_by_optimizer_groups(optimizer)
                update_sum_dict(grad_sum, grad_count, gstats)

            total_norm_value = float("nan")
            if grad_clip > 0:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                total_norm_value = tensor_to_float(total_norm)
                update_sum_dict(grad_sum, grad_count, {"grad_total_preclip": total_norm_value})

            scaler.step(optimizer)
            scaler.update()
            if ema is not None and epoch >= ema_start:
                ema.update(model)

            running += loss.item()
            update_sum_dict(loss_sum, loss_count, loss_parts)
            update_sum_dict(stat_sum, stat_count, collect_output_stats(outputs, masks, threshold=threshold))

            with torch.no_grad():
                probs = torch.sigmoid(logits.detach())
                preds = (probs > threshold).float()
                masks_f = masks.float()
                TP += (preds * masks_f).sum().item()
                FP += (preds * (1.0 - masks_f)).sum().item()
                FN += ((1.0 - preds) * masks_f).sum().item()

            tbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "main": f"{loss_parts.get('main', 0):.4f}",
                "g": f"{total_norm_value:.2f}" if math.isfinite(total_norm_value) else "nan",
            })

        train_loss = running / max(len(train_loader), 1)
        train_precision = TP / (TP + FP + 1e-6)
        train_recall = TP / (TP + FN + 1e-6)
        train_f1 = 2.0 * train_precision * train_recall / (train_precision + train_recall + 1e-6)
        train_iou = TP / (TP + FP + FN + 1e-6)
        train_loss_parts = avg_dict(loss_sum, loss_count)
        train_branch_stats = avg_dict(stat_sum, stat_count)
        train_grad_stats = avg_dict(grad_sum, grad_count)
        train_epoch_metrics = {
            "loss": train_loss,
            "iou": train_iou,
            "precision": train_precision,
            "recall": train_recall,
            "f1": train_f1,
        }
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
            f"Epoch {epoch:03d}/{epochs} | lr={lr_now:.2e} | train: loss={train_loss:.4f} IoU={train_iou*100:.2f} P={train_precision*100:.2f} R={train_recall*100:.2f} F1={train_f1*100:.2f} | "
            f"raw: main_loss={raw['loss']:.4f} full_loss={raw['full_loss']:.4f} IoU={raw['iou']*100:.2f} P={raw['precision']*100:.2f} R={raw['recall']*100:.2f} F1={raw['f1']*100:.2f} "
            f"bestThr={raw['best_sweep']['threshold']:.2f}/IoU={raw['best_sweep']['iou']*100:.2f}"
        )
        if ema_metrics is not None:
            msg += f" | ema: IoU={ema_metrics['iou']*100:.2f} F1={ema_metrics['f1']*100:.2f} bestThr={ema_metrics['best_sweep']['threshold']:.2f}/IoU={ema_metrics['best_sweep']['iou']*100:.2f}"
        msg += "\n"
        log_print(msg, log_file)
        log_print("  " + overfit_message(train_epoch_metrics, raw) + "\n", log_file)
        log_print("  TrainBranch: " + format_branch_stats(train_branch_stats) + "\n", log_file)
        log_print("  ValBranch:   " + format_branch_stats(raw.get("branch_stats", {})) + "\n", log_file)
        if train_grad_stats:
            grad_str = " | ".join([f"{k}={v:.3f}" for k, v in sorted(train_grad_stats.items())])
            log_print("  GradNorm: " + grad_str + "\n", log_file)
        lr_str = " | ".join([f"{k}={v:.2e}" for k, v in lrs.items()])
        log_print("  LRGroups: " + lr_str + "\n", log_file)

        # CSV：一行一个 epoch，方便训练后画过拟合曲线和分支有效性曲线。
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
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
        append_csv(metrics_csv, row)

        branch_row = {"epoch": epoch}
        branch_row.update({f"train_{k}": v for k, v in train_branch_stats.items()})
        branch_row.update({f"val_{k}": v for k, v in raw.get("branch_stats", {}).items()})
        branch_row.update({f"grad_{k}": v for k, v in train_grad_stats.items()})
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

        # 保存最新断点。
        last_metrics = copy.deepcopy(raw)
        last_metrics["best_iou"] = best_iou
        last_metrics["best_epoch"] = best_epoch
        last_metrics["best_ema_iou"] = best_ema_iou
        last_metrics["best_ema_epoch"] = best_ema_epoch
        last_metrics["train_epoch"] = train_epoch_metrics
        last_metrics["train_branch_stats"] = train_branch_stats
        last_metrics["train_grad_stats"] = train_grad_stats
        save_checkpoint(os.path.join(exp_dir, "last_checkpoint.pth"), model, optimizer, scheduler, scaler, epoch, last_metrics, config)

        no_improve = 0 if improved else no_improve + 1
        if no_improve >= early_stop:
            log_print(f"Early stop at epoch {epoch}: no improvement for {early_stop} epochs. Best raw epoch={best_epoch}, best EMA epoch={best_ema_epoch}.\n", log_file)
            break

    log_print(f"Training done. Best raw IoU={best_iou*100:.2f} at epoch={best_epoch}, best EMA IoU={best_ema_iou*100:.2f} at epoch={best_ema_epoch}\n", log_file)
    log_file.close()


if __name__ == "__main__":
    main()
