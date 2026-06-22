# -*- coding: utf-8 -*-
"""
train_RD_v3A.py

RD_v3A 专用训练脚本：
1) 只使用最终主输出 logits / final_logits 计算 BCE+Dice 主损失；
2) 不读取、不计算 logit16/logit8/logit4/logit2 辅助监督；
3) 不读取、不计算 prior4/prior8/prior16 先验监督；
4) 兼容模型输出 tensor 或 dict；
5) 兼容模型自带 get_param_groups() 的分组学习率。

用法示例：
python train_RD_v3A.py -c configs/RD_v3A_CHN6_CUG.json
"""

import os
import json
import csv
import time
import random
import argparse
import datetime
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.road_dataset import RoadDataset
from models import get_model


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_string() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def append_log(msg: str, log_path: str) -> None:
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def extract_main_logits(outputs: Any) -> torch.Tensor:
    """只取最终主输出，不读取任何辅助输出。"""
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, dict):
        if "logits" in outputs:
            return outputs["logits"]
        if "final_logits" in outputs:
            return outputs["final_logits"]
    raise RuntimeError(
        "模型输出必须是 Tensor，或包含 'logits' / 'final_logits' 的 dict。"
    )


def unpack_batch(batch_data: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """兼容 Dataset 返回 (img, mask) 或 (img, mask, name)。"""
    if isinstance(batch_data, (list, tuple)):
        if len(batch_data) >= 2:
            return batch_data[0], batch_data[1]
    raise RuntimeError("DataLoader batch 格式异常，期望至少包含 images 和 masks。")


# -----------------------------
# Loss and metrics
# -----------------------------
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)

        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.contiguous().view(probs.shape[0], -1)
        targets_flat = targets.contiguous().view(targets.shape[0], -1)
        inter = (probs * targets_flat).sum(dim=1)
        union = probs.sum(dim=1) + targets_flat.sum(dim=1)
        dice = 1.0 - ((2.0 * inter + self.smooth) / (union + self.smooth)).mean()
        return self.bce_weight * bce + self.dice_weight * dice


class Evaluator:
    """Dataset-level metrics: accumulate TP/FP/FN over the full set."""
    def __init__(self, threshold: float = 0.5):
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        if logits.shape[-2:] != targets.shape[-2:]:
            logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()
        targets = targets.float()
        self.TP += (preds * targets).sum().item()
        self.FP += (preds * (1.0 - targets)).sum().item()
        self.FN += ((1.0 - preds) * targets).sum().item()

    def get_metrics(self) -> Dict[str, float]:
        eps = 1e-6
        precision = self.TP / (self.TP + self.FP + eps)
        recall = self.TP / (self.TP + self.FN + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        iou = self.TP / (self.TP + self.FP + self.FN + eps)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }


# -----------------------------
# Training / validation
# -----------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Train", ncols=120)
    for batch_data in pbar:
        imgs, masks = unpack_batch(batch_data)
        imgs = imgs.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            if hasattr(model, "forward_train"):
                outputs = model.forward_train(imgs)
            else:
                outputs = model(imgs)
            logits = extract_main_logits(outputs)
            loss = criterion(logits, masks)

        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss is not finite: {loss.item()}")

        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    evaluator = Evaluator(threshold=threshold)
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc="Val", ncols=120)
    for batch_data in pbar:
        imgs, masks = unpack_batch(batch_data)
        imgs = imgs.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        outputs = model(imgs)
        logits = extract_main_logits(outputs)
        loss = criterion(logits, masks)

        total_loss += loss.item()
        n_batches += 1
        evaluator.update(logits, masks)
        metrics = evaluator.get_metrics()
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{metrics['iou']:.4f}")

    return total_loss / max(n_batches, 1), evaluator.get_metrics()


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any], log_path: str = "") -> torch.optim.Optimizer:
    train_cfg = config.get("training", {})
    base_lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))

    if hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(base_lr=base_lr, weight_decay=weight_decay)
        lines = ["", "[Optimizer param groups]"]
        for g in param_groups:
            num = sum(p.numel() for p in g["params"] if p.requires_grad)
            lines.append(
                f"  - {g.get('name', 'group')}: lr={g.get('lr', base_lr):.3e}, "
                f"wd={g.get('weight_decay', weight_decay):.3e}, params={num/1e6:.3f}M"
            )
        if log_path:
            append_log("\n".join(lines), log_path)
        else:
            print("\n".join(lines))
        return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)

    params = [p for p in model.parameters() if p.requires_grad]
    if log_path:
        append_log(
            "\n[Optimizer]\n"
            f"  - AdamW 普通参数模式: lr={base_lr:.3e}, wd={weight_decay:.3e}, "
            f"params={sum(p.numel() for p in params)/1e6:.3f}M",
            log_path,
        )
    return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_iou: float,
    best_epoch: int,
    no_improve: int,
    config: Dict[str, Any],
    metrics: Dict[str, Any] = None,
) -> None:
    state = {
        "epoch": epoch,
        "best_iou": best_iou,
        "best_epoch": best_epoch,
        "no_improve": no_improve,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }
    if metrics is not None:
        state["metrics"] = metrics
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--resume", type=str, default="", help="可选：继续训练的 checkpoint 路径")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = int(config.get("seed", 42))
    set_seed(seed)

    dataset_cfg = config["dataset"]
    train_cfg = config.get("training", {})
    model_cfg = config["model"]

    dataset_name = dataset_cfg["name"]
    model_name = model_cfg["name"]
    img_size = int(dataset_cfg.get("input_size", 512))

    resume_mode = bool(args.resume and os.path.isfile(args.resume))
    if resume_mode:
        # 断点续训时继续写入原实验目录，避免新建目录导致日志和权重分散。
        save_dir = os.path.dirname(args.resume)
    else:
        run_name = f"{model_name}_{now_string()}"
        save_root = train_cfg.get("save_root", "saved_runs")
        save_dir = os.path.join(save_root, dataset_name, run_name)
    ensure_dir(save_dir)

    log_path = os.path.join(save_dir, "train.log")
    metrics_csv = os.path.join(save_dir, "metrics.csv")
    if not resume_mode:
        write_json(config, os.path.join(save_dir, "config.json"))

    train_start_time = time.time()
    train_start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    append_log("=" * 80, log_path)
    if resume_mode:
        append_log("触发断点续训，继续向原目录记录日志", log_path)
    append_log(f"实验开始，日志和权重保存在: {save_dir}", log_path)
    append_log(f"训练开始时间: {train_start_str}", log_path)
    append_log(f"模型: {model_name} | 数据集: {dataset_name} | img_size={img_size}", log_path)
    append_log("RD_v3A 训练模式：只计算 final logits 的主 BCE+Dice loss；不使用 aux/prior supervision。", log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = RoadDataset(dataset_cfg["root_path"], dataset_name, mode="train", img_size=img_size)
    val_dataset = RoadDataset(dataset_cfg["root_path"], dataset_name, mode="val", img_size=img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(dataset_cfg.get("batch_size", 8)),
        shuffle=True,
        num_workers=int(dataset_cfg.get("num_workers", 8)),
        pin_memory=True,
        drop_last=bool(dataset_cfg.get("drop_last", True)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(dataset_cfg.get("val_batch_size", dataset_cfg.get("batch_size", 8))),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 8)),
        pin_memory=True,
        drop_last=False,
    )

    model = get_model(model_cfg, img_size=img_size).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    append_log(f"Total Params: {total_params / 1e6:.2f}M | Trainable Params: {trainable_params / 1e6:.2f}M", log_path)

    criterion = BCEDiceLoss(
        bce_weight=float(train_cfg.get("bce_weight", 0.5)),
        dice_weight=float(train_cfg.get("dice_weight", 0.5)),
        smooth=float(train_cfg.get("dice_smooth", 1.0)),
    )
    optimizer = build_optimizer(model, config, log_path=log_path)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg.get("lr_factor", 0.5)),
        patience=int(train_cfg.get("lr_patience", 5)),
        min_lr=float(train_cfg.get("min_lr", 1e-7)),
    )

    use_amp = bool(train_cfg.get("amp", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    threshold = float(train_cfg.get("threshold", 0.5))

    start_epoch = 1
    best_iou = -1.0
    best_epoch = -1
    no_improve = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_iou = float(ckpt.get("best_iou", -1.0))
        best_epoch = int(ckpt.get("best_epoch", -1))
        no_improve = int(ckpt.get("no_improve", 0))
        append_log(
            f"从 checkpoint 恢复: {args.resume}, start_epoch={start_epoch}, "
            f"best_iou={best_iou:.4f}, best_epoch={best_epoch}, 连续未提升={no_improve}",
            log_path,
        )

    if not (resume_mode and os.path.isfile(metrics_csv)):
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss", "precision", "recall", "f1", "iou",
                "lr", "best_iou", "best_epoch", "no_improve", "epoch_time_min", "total_time"
            ])
    else:
        append_log(f"继续追加 metrics.csv: {metrics_csv}", log_path)

    epochs = int(train_cfg.get("epochs", 200))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 25))

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        append_log(f"\nEpoch [{epoch}/{epochs}] | lr={current_lr:.3e}", log_path)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=grad_clip,
        )
        val_loss, metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
        )
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        msg = (
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"P={metrics['precision']*100:.2f}% R={metrics['recall']*100:.2f}% "
            f"F1={metrics['f1']*100:.2f}% IoU={metrics['iou']*100:.2f}% | "
            f"time={elapsed/60:.1f}min"
        )
        append_log(msg, log_path)

        # best by validation IoU
        improved = metrics["iou"] > best_iou
        if improved:
            best_iou = metrics["iou"]
            best_epoch = epoch
            no_improve = 0
            summary = {
                "best_epoch": best_epoch,
                "best_iou": best_iou,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "val_loss": val_loss,
                "train_loss": train_loss,
                "epoch_time_min": elapsed / 60.0,
                "total_time": format_duration(time.time() - train_start_time),
                "note": "RD_v3A uses only final logits main BCE+Dice loss; no aux/prior supervision.",
            }
            save_checkpoint(
                os.path.join(save_dir, "best_model.pth"),
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_iou,
                best_epoch,
                no_improve,
                config,
                metrics=summary,
            )
            write_json(summary, os.path.join(save_dir, "best_summary.json"))
            append_log(
                f"✅ New best IoU: {best_iou*100:.2f}% @ epoch {epoch} | "
                f"连续未提升: {no_improve}/{early_stop_patience}",
                log_path,
            )
        else:
            no_improve += 1
            append_log(
                f"No improvement: {no_improve}/{early_stop_patience} | "
                f"best_iou={best_iou*100:.2f}% @ epoch {best_epoch}",
                log_path,
            )

        total_elapsed_str = format_duration(time.time() - train_start_time)
        with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{metrics['precision']:.6f}",
                f"{metrics['recall']:.6f}",
                f"{metrics['f1']:.6f}",
                f"{metrics['iou']:.6f}",
                f"{current_lr:.8e}",
                f"{best_iou:.6f}",
                best_epoch,
                no_improve,
                f"{elapsed/60.0:.4f}",
                total_elapsed_str,
            ])

        # save latest checkpoint with updated best_iou / best_epoch / no_improve.
        latest_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "iou": metrics["iou"],
            "best_iou": best_iou,
            "best_epoch": best_epoch,
            "no_improve": no_improve,
            "epoch_time_min": elapsed / 60.0,
            "total_time": total_elapsed_str,
        }
        save_checkpoint(
            os.path.join(save_dir, "last_model.pth"),
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_iou,
            best_epoch,
            no_improve,
            config,
            metrics=latest_metrics,
        )

        if no_improve >= early_stop_patience:
            append_log(f"Early stopping triggered at epoch {epoch}. 连续未提升: {no_improve}/{early_stop_patience}", log_path)
            break

    train_end_time = time.time()
    train_end_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    append_log("\n" + "=" * 80, log_path)
    append_log("训练结束", log_path)
    append_log(f"训练结束时间: {train_end_str}", log_path)
    append_log(f"训练总耗时: {format_duration(train_end_time - train_start_time)}", log_path)
    append_log(f"Best IoU: {best_iou*100:.2f}% @ epoch {best_epoch}", log_path)
    append_log(f"最终连续未提升轮数: {no_improve}/{early_stop_patience}", log_path)
    append_log(f"权重目录: {save_dir}", log_path)


if __name__ == "__main__":
    main()
