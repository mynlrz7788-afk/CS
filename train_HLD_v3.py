import os
import cv2
cv2.setNumThreads(0)
import time
import json
import copy
import math
import random
import argparse
import datetime
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from thop import profile

from dataloaders.road_dataset import RoadDataset
from dataloaders.drive_dataset import DRIVEDataset
from models import get_model
from core.loss import BCEDiceLoss
from core.metrics import Evaluator


# =========================================================
# 1. Reproducibility
# =========================================================
def set_seed(seed=3407):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =========================================================
# 2. Experiment / logging helpers
# =========================================================
def create_experiment_dir(config):
    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_dir = os.path.join("saved_runs", dataset_name, f"{model_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    return exp_dir


def log_print(msg: str, log_file=None, end: str = ""):
    print(msg, end=end)
    if log_file is not None:
        log_file.write(msg + end)
        log_file.flush()


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


# =========================================================
# 3. Model stats / complexity
# =========================================================
def count_model_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    dino_total = 0
    dino_trainable = 0
    if hasattr(model, "dino"):
        dino_total = sum(p.numel() for p in model.dino.parameters())
        dino_trainable = sum(p.numel() for p in model.dino.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "dino_total": dino_total,
        "dino_trainable": dino_trainable,
    }


def evaluate_model_complexity(model, device, img_size):
    """thop 失败时不终止训练。"""
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    param_stats = count_model_parameters(model)
    params = param_stats["total"]
    flops = float("nan")
    profile_error = None

    try:
        model_for_profile = copy.deepcopy(model).to(device)
        model_for_profile.eval()
        macs, thop_params = profile(model_for_profile, inputs=(dummy_input,), verbose=False)
        flops = macs
        params = thop_params
        del model_for_profile
        torch.cuda.empty_cache()
    except Exception as e:
        profile_error = repr(e)

    fps = float("nan")
    try:
        warmup_iters = 20
        test_iters = 50
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            for _ in range(test_iters):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        end_time = time.time()
        fps = test_iters / max(end_time - start_time, 1e-12)
    except Exception as e:
        if profile_error is None:
            profile_error = f"FPS eval error: {repr(e)}"
        else:
            profile_error += f" | FPS eval error: {repr(e)}"

    model.train()
    return params, flops, fps, profile_error, param_stats


# =========================================================
# 4. Optimizer with param groups
# =========================================================
def _append_clean_group(param_groups, used_param_ids, params, lr, weight_decay, name):
    clean_params = []
    for p in params:
        if p is None or not p.requires_grad:
            continue
        pid = id(p)
        if pid in used_param_ids:
            continue
        clean_params.append(p)
        used_param_ids.add(pid)

    if len(clean_params) > 0:
        param_groups.append({
            "params": clean_params,
            "lr": lr,
            "weight_decay": weight_decay,
            "name": name,
        })


def _build_config_param_groups(model, config, base_lr, base_wd):
    """
    当模型没有 get_param_groups() 时，根据 config['model']['param_group_lrs']
    和 HLD_v1 的模块名构造参数组。
    """
    model_cfg = config.get("model", {})
    lr_cfg = model_cfg.get("param_group_lrs", {})
    wd_cfg = model_cfg.get("param_group_weight_decays", {})

    if not lr_cfg.get("enabled", False):
        return None

    param_groups = []
    used_param_ids = set()

    module_specs = [
        ("road_branch", "road_branch_lr", "road_branch_wd"),
        ("prompt_extractor", "prompt_extractor_lr", "prompt_extractor_wd"),
        ("wpda_adapters", "wpda_lr", "wpda_wd"),
        ("readout", "hld_readout_lr", "hld_readout_wd"),
        ("decoder", "decoder_lr", "decoder_wd"),
    ]

    for module_name, lr_key, wd_key in module_specs:
        if not hasattr(model, module_name):
            continue
        module = getattr(model, module_name)
        lr = float(lr_cfg.get(lr_key, base_lr))
        wd = float(wd_cfg.get(wd_key, base_wd))
        _append_clean_group(
            param_groups,
            used_param_ids,
            module.parameters(),
            lr=lr,
            weight_decay=wd,
            name=module_name,
        )

    # 防止漏掉可训练参数。
    missing_params = []
    for p in model.parameters():
        if p.requires_grad and id(p) not in used_param_ids:
            missing_params.append(p)
    _append_clean_group(
        param_groups,
        used_param_ids,
        missing_params,
        lr=base_lr,
        weight_decay=base_wd,
        name="others",
    )

    if len(param_groups) == 0:
        return None
    return param_groups


def build_optimizer(model, config, weight_decay):
    base_lr = float(config["training"]["lr"])
    optimizer_name = config["training"].get("optimizer", "AdamW").lower()

    param_groups = None
    opt_type = None

    # 1) 优先使用模型自己定义的参数组。
    if hasattr(model, "get_param_groups") and callable(getattr(model, "get_param_groups")):
        raw_param_groups = model.get_param_groups()
        param_groups = []
        used_param_ids = set()

        for group in raw_param_groups:
            if "params" not in group:
                continue
            clean_params = []
            for p in group["params"]:
                if p is None or not p.requires_grad:
                    continue
                pid = id(p)
                if pid in used_param_ids:
                    continue
                clean_params.append(p)
                used_param_ids.add(pid)

            if len(clean_params) == 0:
                continue

            new_group = dict(group)
            new_group["params"] = clean_params
            if "lr" not in new_group:
                new_group["lr"] = base_lr
            if "weight_decay" not in new_group:
                new_group["weight_decay"] = weight_decay
            if "name" not in new_group:
                new_group["name"] = f"group{len(param_groups)}"
            param_groups.append(new_group)

        missing_params = []
        for p in model.parameters():
            if p.requires_grad and id(p) not in used_param_ids:
                missing_params.append(p)
        if len(missing_params) > 0:
            param_groups.append({
                "params": missing_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "missing_params",
            })
        opt_type = "model.get_param_groups()"

    # 2) 如果模型没有 get_param_groups，则根据 json 的模块名分组。
    if param_groups is None:
        param_groups = _build_config_param_groups(model, config, base_lr, weight_decay)
        if param_groups is not None:
            opt_type = "config param_group_lrs"

    # 3) 兜底：普通优化器。
    if param_groups is None:
        param_groups = model.parameters()
        opt_type = "plain parameters"

    if optimizer_name == "adam":
        optimizer = optim.Adam(param_groups, lr=base_lr, weight_decay=weight_decay)
        opt_name = "Adam"
    else:
        optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)
        opt_name = "AdamW"

    msg_lines = []
    msg_lines.append(" 优化器设置:\n")
    msg_lines.append(f" - 类型: {opt_name} with {opt_type}\n")

    if isinstance(param_groups, list):
        msg_lines.append(f" - 参数组数量: {len(param_groups)}\n")
        for idx, group in enumerate(param_groups):
            group_params = group["params"]
            group_lr = group.get("lr", base_lr)
            group_wd = group.get("weight_decay", weight_decay)
            group_name = group.get("name", f"group{idx}")
            group_param_num = sum(p.numel() for p in group_params)
            msg_lines.append(
                f"   Group {idx} [{group_name}]: lr={group_lr}, weight_decay={group_wd}, "
                f"params={group_param_num / 1e6:.3f}M\n"
            )
    else:
        msg_lines.append(f" - lr: {base_lr}\n")
        msg_lines.append(f" - weight_decay: {weight_decay}\n")

    msg_lines.append("--------------------------------------------------\n")
    return optimizer, "".join(msg_lines)


# =========================================================
# 5. EMA
# =========================================================
class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.ema = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        model_state = model.state_dict()
        ema_state = self.ema.state_dict()

        for k, ema_v in ema_state.items():
            if k not in model_state:
                continue
            model_v = model_state[k].detach()
            if torch.is_floating_point(ema_v):
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)

    def state_dict(self):
        return self.ema.state_dict()

    def load_state_dict(self, state_dict):
        self.ema.load_state_dict(state_dict, strict=True)


# =========================================================
# 6. Validation / metrics
# =========================================================
def _unpack_batch(batch_data):
    if len(batch_data) == 3:
        imgs, masks, _ = batch_data
    else:
        imgs, masks = batch_data
    return imgs, masks


def _extract_logits(preds):
    if isinstance(preds, dict):
        if "final_logits" in preds:
            return preds["final_logits"]
        if "logits" in preds:
            return preds["logits"]
        # 兜底取第一个 Tensor
        for v in preds.values():
            if torch.is_tensor(v):
                return v
        raise RuntimeError("模型返回 dict，但找不到 logits Tensor。")
    if isinstance(preds, (tuple, list)):
        return preds[0]
    return preds



# =========================================================
# 6.1 HLD_v3 auxiliary losses: road prior + skeleton prior
# =========================================================
def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, kernel_size=(3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-img, kernel_size=(1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skeletonize(mask: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """
    Torch morphology skeletonization for GT masks.
    输入 mask: B×1×H×W, float in [0,1]. 输出 soft skeleton in [0,1].
    """
    mask = mask.float().clamp(0.0, 1.0)
    skel = F.relu(mask - _soft_open(mask))
    img = mask
    for _ in range(int(iterations)):
        img = _soft_erode(img)
        delta = F.relu(img - _soft_open(img))
        skel = skel + F.relu(delta - skel * delta)
    return skel.clamp(0.0, 1.0)


def make_skeleton_target(mask_4: torch.Tensor, iterations: int = 20, dilate_kernel: int = 3) -> torch.Tensor:
    """从 H/4 mask 生成膨胀后的 skeleton target。"""
    skel = soft_skeletonize(mask_4, iterations=iterations)
    k = int(dilate_kernel)
    if k > 1:
        pad = k // 2
        skel = F.max_pool2d(skel, kernel_size=k, stride=1, padding=pad)
    return skel.clamp(0.0, 1.0)


def get_skeleton_weight(epoch: int, config: Dict[str, Any]) -> float:
    loss_cfg = config.get("training", {}).get("hld_v3_loss", {})
    target = float(loss_cfg.get("skeleton_weight", 0.05))
    warmup_start = int(loss_cfg.get("skeleton_warmup_start", 10))
    warmup_end = int(loss_cfg.get("skeleton_warmup_end", 30))
    if epoch <= warmup_start:
        return 0.0
    if epoch >= warmup_end:
        return target
    ratio = (epoch - warmup_start) / max(warmup_end - warmup_start, 1)
    return float(target * ratio)


def compute_hld_v3_loss(preds, masks, criterion, epoch: int, config: Dict[str, Any]):
    """
    L_total = L_final + w_half*L_half + w_prior*L_prior + w_skel*L_skeleton
    如果模型不是 HLD_v3 dict 输出，则自动退化成普通 final loss。
    """
    loss_cfg = config.get("training", {}).get("hld_v3_loss", {})
    final_logits = _extract_logits(preds)
    loss_final = criterion(final_logits, masks)
    total_loss = loss_final
    items = {"final": float(loss_final.detach().item())}

    if not isinstance(preds, dict):
        items.update({"half": 0.0, "prior": 0.0, "skeleton": 0.0, "skeleton_weight": 0.0})
        return total_loss, items

    # H/2 auxiliary segmentation loss
    half_weight = float(loss_cfg.get("half_weight", 0.20))
    if half_weight > 0 and "logits_half" in preds:
        logits_half = preds["logits_half"]
        mask_half = F.interpolate(masks.float(), size=logits_half.shape[-2:], mode="nearest")
        loss_half = criterion(logits_half, mask_half)
        total_loss = total_loss + half_weight * loss_half
        items["half"] = float(loss_half.detach().item())
    else:
        items["half"] = 0.0

    # DINO road prior loss
    prior_weight = float(loss_cfg.get("prior_weight", 0.10))
    if prior_weight > 0 and "road_prior_logits" in preds:
        road_prior_logits = preds["road_prior_logits"]
        mask_4 = F.interpolate(masks.float(), size=road_prior_logits.shape[-2:], mode="nearest")
        loss_prior = criterion(road_prior_logits, mask_4)
        total_loss = total_loss + prior_weight * loss_prior
        items["prior"] = float(loss_prior.detach().item())
    else:
        items["prior"] = 0.0

    # DINO skeleton prior loss, with warmup
    skeleton_weight = get_skeleton_weight(epoch, config)
    if skeleton_weight > 0 and "skeleton_logits" in preds:
        skeleton_logits = preds["skeleton_logits"]
        mask_4 = F.interpolate(masks.float(), size=skeleton_logits.shape[-2:], mode="nearest")
        skel_iters = int(loss_cfg.get("skeleton_iterations", 20))
        dilate_kernel = int(loss_cfg.get("skeleton_dilate_kernel", 3))
        with torch.no_grad():
            skel_gt = make_skeleton_target(mask_4, iterations=skel_iters, dilate_kernel=dilate_kernel)
        loss_skel = criterion(skeleton_logits, skel_gt)
        total_loss = total_loss + skeleton_weight * loss_skel
        items["skeleton"] = float(loss_skel.detach().item())
    else:
        items["skeleton"] = 0.0
    items["skeleton_weight"] = float(skeleton_weight)
    items["total"] = float(total_loss.detach().item())
    return total_loss, items


def validate_one_epoch(model, val_loader, criterion, device="cuda", threshold=0.5, desc="Valid"):
    model.eval()
    val_loss = 0.0
    evaluator = Evaluator(num_class=2)
    evaluator.reset()

    pred_fg_sum = 0.0
    gt_fg_sum = 0.0
    pixel_sum = 0.0

    with torch.no_grad():
        loader_tqdm = tqdm(val_loader, desc=desc, leave=False)
        for batch_data in loader_tqdm:
            imgs, masks = _unpack_batch(batch_data)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                preds = _extract_logits(model(imgs))
                loss = criterion(preds, masks)

            val_loss += loss.item()
            probs = torch.sigmoid(preds)
            preds_bin_t = (probs > threshold).float()

            pred_fg_sum += preds_bin_t.sum().item()
            gt_fg_sum += masks.float().sum().item()
            pixel_sum += masks.numel()

            preds_bin = preds_bin_t.detach().cpu().numpy().astype(int)
            masks_int = masks.detach().cpu().numpy().astype(int)
            evaluator.add_batch(masks_int, preds_bin)
            loader_tqdm.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_val_loss = val_loss / max(len(val_loader), 1)

    precision = evaluator.Pixel_Precision()
    recall = evaluator.Pixel_Recall()
    f1 = evaluator.Pixel_F1()
    iou = evaluator.Intersection_over_Union()

    pred_fg_ratio = pred_fg_sum / max(pixel_sum, 1.0)
    gt_fg_ratio = gt_fg_sum / max(pixel_sum, 1.0)

    return {
        "loss": float(avg_val_loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "pred_fg_ratio": float(pred_fg_ratio),
        "gt_fg_ratio": float(gt_fg_ratio),
    }


def compute_balanced_score(metrics: Dict[str, float], config: Dict[str, Any]) -> float:
    sel_cfg = config.get("training", {}).get("balanced_selection", {})
    precision_weight = float(sel_cfg.get("precision_weight", 0.10))
    pr_gap_penalty = float(sel_cfg.get("pr_gap_penalty", 0.05))
    fg_ratio_penalty = float(sel_cfg.get("fg_ratio_penalty", 0.03))

    iou = metrics.get("iou", 0.0)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)
    pred_fg_ratio = metrics.get("pred_fg_ratio", 0.0)
    gt_fg_ratio = metrics.get("gt_fg_ratio", 0.0)

    pr_gap = abs(recall - precision)
    over_pred_ratio = max(0.0, pred_fg_ratio - gt_fg_ratio) / max(gt_fg_ratio, 1e-6)

    score = iou + precision_weight * precision - pr_gap_penalty * pr_gap - fg_ratio_penalty * over_pred_ratio
    return float(score)


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    return (
        f"{prefix} Loss: {metrics['loss']:.4f} | "
        f"P: {metrics['precision']:.4f} | "
        f"R: {metrics['recall']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"IoU: {metrics['iou']:.4f} | "
        f"PredFG: {metrics['pred_fg_ratio']:.5f} | "
        f"GtFG: {metrics['gt_fg_ratio']:.5f}"
    )


# =========================================================
# 7. Checkpoint manager
# =========================================================
def save_eval_checkpoint(path, epoch, state_dict, metrics, config, tag):
    torch.save({
        "epoch": epoch,
        "model_state_dict": state_dict,
        "metrics": metrics,
        "config": config,
        "tag": tag,
    }, path)


def save_latest_checkpoint(path, epoch, model, optimizer, scheduler, scaler, ema, state_tracker):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "state_tracker": state_tracker,
    }
    if ema is not None:
        ckpt["ema_state_dict"] = ema.state_dict()
    torch.save(ckpt, path)


def _safe_remove(path: Optional[str]):
    if path is not None and os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _metric_for_topk(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name == "balanced_score":
        return float(metrics.get("balanced_score", -1e9))
    return float(metrics.get(metric_name, -1e9))


def update_topk_bank(
    bank: List[Dict[str, Any]],
    limit: int,
    score: float,
    path: str,
    epoch: int,
    kind: str,
):
    bank.append({"score": float(score), "path": path, "epoch": int(epoch), "kind": kind})
    bank.sort(key=lambda x: x["score"], reverse=True)
    removed = []
    while len(bank) > limit:
        item = bank.pop(-1)
        removed.append(item)
        _safe_remove(item.get("path"))
    return bank, removed


def average_checkpoints(checkpoint_paths: List[str], out_path: str, config: Dict[str, Any], tag: str):
    if len(checkpoint_paths) == 0:
        return False

    avg_state = None
    first_epoch = None
    metrics_list = []

    for idx, path in enumerate(checkpoint_paths):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]
        if first_epoch is None:
            first_epoch = ckpt.get("epoch", -1)
        metrics_list.append(ckpt.get("metrics", {}))

        if avg_state is None:
            avg_state = {}
            for k, v in state.items():
                if torch.is_floating_point(v):
                    avg_state[k] = v.detach().clone().float()
                else:
                    avg_state[k] = v.detach().clone()
        else:
            for k, v in state.items():
                if torch.is_floating_point(v) and k in avg_state and torch.is_floating_point(avg_state[k]):
                    avg_state[k] += v.detach().float()

    n = float(len(checkpoint_paths))
    for k, v in avg_state.items():
        if torch.is_floating_point(v):
            avg_state[k] = (v / n)

    torch.save({
        "epoch": first_epoch,
        "model_state_dict": avg_state,
        "metrics": {"source_checkpoints": checkpoint_paths, "source_metrics": metrics_list},
        "config": config,
        "tag": tag,
    }, out_path)
    return True


# =========================================================
# 8. Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
    parser.add_argument("-r", "--resume", type=str, default=None, help="Path to latest_model.pth to resume")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    seed = int(config["training"].get("seed", 3407))
    set_seed(seed)

    if args.resume and os.path.isfile(args.resume):
        exp_dir = os.path.dirname(args.resume)
        log_file = open(os.path.join(exp_dir, "train.log"), "a", encoding="utf-8")
        resume_msg = f"\n{'=' * 50}\n 🔌 触发断点续训，继续向原目录记录日志 \n{'=' * 50}\n"
        log_print(resume_msg, log_file)
    else:
        exp_dir = create_experiment_dir(config)
        log_file = open(os.path.join(exp_dir, "train.log"), "w", encoding="utf-8")

    start_time_raw = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f" 实验开始，日志和权重保存在: {exp_dir}\n 训练开始时间: {start_time_str}\n", log_file)

    # Runtime info
    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    runtime_msg = (
        f"--------------------------------------------------\n"
        f" 🧾 实验基本信息\n"
        f" - 配置文件: {args.config}\n"
        f" - 随机种子: {seed}\n"
        f" - PyTorch版本: {torch.__version__}\n"
        f" - CUDA可用: {torch.cuda.is_available()}\n"
        f" - 当前设备: {cuda_name}\n"
        f" - cuDNN benchmark: {torch.backends.cudnn.benchmark}\n"
        f" - cuDNN deterministic: {torch.backends.cudnn.deterministic}\n"
        f" - AMP混合精度: 开启\n"
        f"--------------------------------------------------\n"
    )
    log_print(runtime_msg, log_file)

    # Data
    dataset_name = config["dataset"]["name"]
    root_path = config["dataset"]["root_path"]
    img_size = int(config["dataset"].get("input_size", 1024))

    if dataset_name.lower() == "drive":
        train_dataset = DRIVEDataset(root_path, dataset_name, mode="train", img_size=img_size)
        val_dataset = DRIVEDataset(root_path, dataset_name, mode="val", img_size=img_size)
    else:
        train_dataset = RoadDataset(root_path, dataset_name, mode="train", img_size=img_size)
        val_dataset = RoadDataset(root_path, dataset_name, mode="val", img_size=img_size)

    n_workers = int(config["dataset"].get("num_workers", 8))
    batch_size = int(config["dataset"]["batch_size"])

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    data_msg = (
        f"--------------------------------------------------\n"
        f" 📚 数据集与DataLoader设置\n"
        f" - 数据集名称: {dataset_name}\n"
        f" - 数据根目录: {root_path}\n"
        f" - 输入尺寸: {img_size}x{img_size}\n"
        f" - 训练样本数: {len(train_dataset)}\n"
        f" - 验证样本数: {len(val_dataset)}\n"
        f" - Batch Size: {batch_size}\n"
        f" - Num Workers: {n_workers}\n"
        f" - Train Iter/Epoch: {len(train_loader)}\n"
        f" - Val Iter/Epoch: {len(val_loader)}\n"
        f"--------------------------------------------------\n"
    )
    log_print(data_msg, log_file)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(config["model"], img_size=img_size).to(device)

    model_cfg_msg = (
        f"--------------------------------------------------\n"
        f" 🧠 模型配置\n"
        f" - 模型名称: {config['model'].get('name', 'Unknown')}\n"
        f" - 类别数: {config['model'].get('num_classes', config['model'].get('n_classes', 1))}\n"
        f" - 预训练CNN: {config['model'].get('pretrained', True)}\n"
        f" - return_aux: {config['model'].get('return_aux', False)}\n"
        f" - DINO模型: {config['model'].get('dino_model_name', 'N/A')}\n"
        f" - DINO层: {config['model'].get('dino_layers', 'N/A')}\n"
        f" - DINO权重: {config['model'].get('dino_ckpt_path', 'N/A')}\n"
        f" - DINO adapter: alpha_max={config['model'].get('adapter_alpha_max', 'N/A')}, dim={config['model'].get('adapter_dim', 'N/A')}\n"
        f" - HLD_v3 loss: {config.get('training', {}).get('hld_v3_loss', {})}\n"
        f"--------------------------------------------------\n"
    )
    log_print(model_cfg_msg, log_file)

    params, flops, fps, profile_error, param_stats = evaluate_model_complexity(model, device=device, img_size=img_size)
    flops_text = f"{flops / 1e9:.2f} G" if np.isfinite(flops) else "thop统计失败"
    fps_text = f"{fps:.2f} 张/秒" if np.isfinite(fps) else "FPS统计失败"
    profile_error_text = f"    - 复杂度统计警告: {profile_error}\n" if profile_error is not None else ""

    complexity_msg = (
        f"--------------------------------------------------\n"
        f" 📊 模型复杂度 @ 输入尺寸: {img_size}x{img_size}\n"
        f"    - 参数量 (Params/thop):       {params / 1e6:.2f} M\n"
        f"    - 总参数量 (direct):          {param_stats['total'] / 1e6:.2f} M\n"
        f"    - 可训练参数量 (Trainable):   {param_stats['trainable'] / 1e6:.2f} M\n"
        f"    - 冻结参数量 (Frozen):        {param_stats['frozen'] / 1e6:.2f} M\n"
        f"    - DINO总参数量:               {param_stats['dino_total'] / 1e6:.2f} M\n"
        f"    - DINO可训练参数量:           {param_stats['dino_trainable'] / 1e6:.2f} M\n"
        f"    - 浮点运算量 (FLOPs):         {flops_text}\n"
        f"    - 推理速度 (FPS):             {fps_text}\n"
        f"{profile_error_text}"
        f"--------------------------------------------------\n"
    )
    log_print(complexity_msg, log_file)

    # Training configs
    train_cfg = config["training"]
    epochs = int(train_cfg["epochs"])
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    lr_factor = float(train_cfg.get("lr_factor", 0.5))
    lr_patience = int(train_cfg.get("lr_patience", 6))
    early_stop_patience = int(train_cfg.get("early_stop_patience", 25))

    optimizer, optimizer_msg = build_optimizer(model, config, weight_decay)
    log_print(optimizer_msg, log_file)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )
    criterion = BCEDiceLoss().to(device)
    scaler = torch.amp.GradScaler("cuda")

    # EMA configs
    ema_cfg = train_cfg.get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.9995))
    ema_start_epoch = int(ema_cfg.get("start_epoch", 1))
    ema_eval = bool(ema_cfg.get("eval_ema", True))
    ema = ModelEMA(model, decay=ema_decay) if ema_enabled else None

    # Grad clipping configs
    grad_cfg = train_cfg.get("gradient_clip", {})
    grad_clip_enabled = bool(grad_cfg.get("enabled", False))
    grad_clip_max_norm = float(grad_cfg.get("max_norm", 1.0))

    # Checkpoint configs
    ckpt_cfg = train_cfg.get("checkpoint", {})
    save_top_k = int(ckpt_cfg.get("save_top_k", 5))
    top_k_metric = str(ckpt_cfg.get("top_k_metric", "balanced_score"))
    save_window_best = bool(ckpt_cfg.get("save_window_best", True))
    window_size = int(ckpt_cfg.get("window_size", 20))
    window_metric = str(ckpt_cfg.get("window_metric", "balanced_score"))
    save_window_raw = bool(ckpt_cfg.get("save_window_raw", True))
    save_window_ema = bool(ckpt_cfg.get("save_window_ema", ema_enabled))
    create_weight_avg = bool(ckpt_cfg.get("create_weight_avg", True))
    candidate_start_epoch = int(ckpt_cfg.get("candidate_start_epoch", 1))

    hyper_msg = (
        f"--------------------------------------------------\n"
        f" ⚙️ 训练超参数设置\n"
        f" - Epochs: {epochs}\n"
        f" - Loss: {train_cfg.get('loss_type', 'BCE_Dice')}\n"
        f" - 主学习率配置 lr: {train_cfg['lr']}\n"
        f" - 优化器: {train_cfg.get('optimizer', 'AdamW')}\n"
        f" - 权重衰减 (Weight Decay): {weight_decay}\n"
        f" - 学习率调度: ReduceLROnPlateau (factor={lr_factor}, patience={lr_patience})\n"
        f" - 早停: {early_stop_patience} 轮\n"
        f" - EMA: {ema_enabled}, decay={ema_decay}\n"
        f" - Gradient Clip: {grad_clip_enabled}, max_norm={grad_clip_max_norm}\n"
        f" - TopK: {save_top_k}, metric={top_k_metric}\n"
        f" - Window Best: {save_window_best}, window_size={window_size}, metric={window_metric}\n"
        f" - best_model.pth 默认指向: best_ema_balanced 若 EMA 开启，否则 best_raw_balanced\n"
        f"--------------------------------------------------\n"
    )
    log_print(hyper_msg, log_file)

    # State trackers
    start_epoch = 1
    epochs_without_improvement = 0
    state_tracker = {
        "best_raw_iou": -1.0,
        "best_ema_iou": -1.0,
        "best_raw_balanced": -1e9,
        "best_ema_balanced": -1e9,
        "topk_raw": [],
        "topk_ema": [],
        "window_best_raw": {},
        "window_best_ema": {},
    }

    # Resume
    if args.resume:
        if os.path.isfile(args.resume):
            msg = f"🔄 发现断点文件，正在恢复: {args.resume}\n"
            log_print(msg, log_file)
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            start_epoch = int(checkpoint["epoch"]) + 1
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            if ema is not None and "ema_state_dict" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state_dict"])
            if "state_tracker" in checkpoint:
                state_tracker = checkpoint["state_tracker"]
            epochs_without_improvement = int(state_tracker.get("epochs_without_improvement", 0))
            log_print(f"成功恢复，将从第 {start_epoch} 轮继续训练。\n", log_file)
        else:
            log_print(f"找不到断点文件: {args.resume}，将从头开始训练。\n", log_file)

    # Main loop
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        grad_norm_sum = 0.0
        grad_norm_count = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch}/{epochs}] Train", leave=False)
        for batch_data in train_loader_tqdm:
            imgs, masks = _unpack_batch(batch_data)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                model_out = model(imgs)
                loss, loss_items = compute_hld_v3_loss(model_out, masks, criterion, epoch=epoch, config=config)

            scaler.scale(loss).backward()

            grad_norm = None
            skip_step = False
            if grad_clip_enabled:
                scaler.unscale_(optimizer)
                grad_norm_t = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
                if torch.is_tensor(grad_norm_t):
                    grad_norm = float(grad_norm_t.item())
                else:
                    grad_norm = float(grad_norm_t)
                if not math.isfinite(grad_norm):
                    skip_step = True
                else:
                    grad_norm_sum += grad_norm
                    grad_norm_count += 1

            if skip_step:
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
                if ema is not None and epoch >= ema_start_epoch:
                    ema.update(model)

            train_loss += loss.item()
            postfix = {"loss": f"{loss.item():.4f}", "skw": f"{loss_items.get('skeleton_weight', 0.0):.3f}"}
            if grad_norm is not None:
                postfix["grad"] = "inf" if skip_step else f"{grad_norm:.2f}"
            train_loader_tqdm.set_postfix(postfix)

        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_grad_norm = grad_norm_sum / max(grad_norm_count, 1) if grad_norm_count > 0 else float("nan")

        # Raw validation
        raw_metrics = validate_one_epoch(model, val_loader, criterion, device=device, threshold=0.5, desc=f"Epoch [{epoch}/{epochs}] Valid Raw")
        raw_metrics["balanced_score"] = compute_balanced_score(raw_metrics, config)

        # EMA validation
        ema_metrics = None
        if ema is not None and ema_eval and epoch >= ema_start_epoch:
            ema_metrics = validate_one_epoch(ema.ema, val_loader, criterion, device=device, threshold=0.5, desc=f"Epoch [{epoch}/{epochs}] Valid EMA")
            ema_metrics["balanced_score"] = compute_balanced_score(ema_metrics, config)

        # Scheduler uses raw val loss by default for continuity with old training.
        scheduler.step(raw_metrics["loss"])

        # Logging
        log_msg = (
            f"Epoch [{epoch}/{epochs}] | Train Loss: {avg_train_loss:.4f} | "
            f"GradNorm: {avg_grad_norm:.4f} | "
            f"{format_metrics('Raw Val', raw_metrics)} | "
            f"Raw Balanced: {raw_metrics['balanced_score']:.4f}\n"
        )
        log_print(log_msg, log_file)

        if ema_metrics is not None:
            ema_msg = f"                 | {format_metrics('EMA Val', ema_metrics)} | EMA Balanced: {ema_metrics['balanced_score']:.4f}\n"
            log_print(ema_msg, log_file)

        lr_msg_parts = []
        for idx, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"g{idx}")
            lr_msg_parts.append(f"{name}:{group['lr']:.2e}")
        log_print("                 | LR Groups: " + ", ".join(lr_msg_parts) + "\n", log_file)

        # Prepare checkpoint records
        latest_path = os.path.join(exp_dir, "latest_model.pth")

        # Global best raw IoU
        if raw_metrics["iou"] > state_tracker["best_raw_iou"]:
            state_tracker["best_raw_iou"] = raw_metrics["iou"]
            path = os.path.join(exp_dir, f"best_raw_iou_e{epoch:03d}_iou{raw_metrics['iou']:.4f}.pth")
            # Remove old best raw iou named file if any
            _safe_remove(state_tracker.get("best_raw_iou_path"))
            save_eval_checkpoint(path, epoch, model.state_dict(), raw_metrics, config, tag="best_raw_iou")
            state_tracker["best_raw_iou_path"] = path
            save_eval_checkpoint(os.path.join(exp_dir, "best_raw_iou.pth"), epoch, model.state_dict(), raw_metrics, config, tag="best_raw_iou")
            log_print(f"      ✅ 保存 best_raw_iou: {os.path.basename(path)}\n", log_file)

        # Global best raw balanced
        if raw_metrics["balanced_score"] > state_tracker["best_raw_balanced"]:
            state_tracker["best_raw_balanced"] = raw_metrics["balanced_score"]
            path = os.path.join(exp_dir, f"best_raw_balanced_e{epoch:03d}_s{raw_metrics['balanced_score']:.4f}.pth")
            _safe_remove(state_tracker.get("best_raw_balanced_path"))
            save_eval_checkpoint(path, epoch, model.state_dict(), raw_metrics, config, tag="best_raw_balanced")
            state_tracker["best_raw_balanced_path"] = path
            save_eval_checkpoint(os.path.join(exp_dir, "best_raw_balanced.pth"), epoch, model.state_dict(), raw_metrics, config, tag="best_raw_balanced")
            if ema is None:
                save_eval_checkpoint(os.path.join(exp_dir, "best_model.pth"), epoch, model.state_dict(), raw_metrics, config, tag="best_raw_balanced_default")
            log_print(f"      ✅ 保存 best_raw_balanced: {os.path.basename(path)}\n", log_file)

        # Global best EMA
        if ema_metrics is not None:
            ema_state = ema.ema.state_dict()

            if ema_metrics["iou"] > state_tracker["best_ema_iou"]:
                state_tracker["best_ema_iou"] = ema_metrics["iou"]
                path = os.path.join(exp_dir, f"best_ema_iou_e{epoch:03d}_iou{ema_metrics['iou']:.4f}.pth")
                _safe_remove(state_tracker.get("best_ema_iou_path"))
                save_eval_checkpoint(path, epoch, ema_state, ema_metrics, config, tag="best_ema_iou")
                state_tracker["best_ema_iou_path"] = path
                save_eval_checkpoint(os.path.join(exp_dir, "best_ema_iou.pth"), epoch, ema_state, ema_metrics, config, tag="best_ema_iou")
                log_print(f"      ✅ 保存 best_ema_iou: {os.path.basename(path)}\n", log_file)

            if ema_metrics["balanced_score"] > state_tracker["best_ema_balanced"]:
                state_tracker["best_ema_balanced"] = ema_metrics["balanced_score"]
                path = os.path.join(exp_dir, f"best_ema_balanced_e{epoch:03d}_s{ema_metrics['balanced_score']:.4f}.pth")
                _safe_remove(state_tracker.get("best_ema_balanced_path"))
                save_eval_checkpoint(path, epoch, ema_state, ema_metrics, config, tag="best_ema_balanced")
                state_tracker["best_ema_balanced_path"] = path
                save_eval_checkpoint(os.path.join(exp_dir, "best_ema_balanced.pth"), epoch, ema_state, ema_metrics, config, tag="best_ema_balanced")
                save_eval_checkpoint(os.path.join(exp_dir, "best_model.pth"), epoch, ema_state, ema_metrics, config, tag="best_ema_balanced_default")
                log_print(f"      ✅ 保存 best_ema_balanced，并同步为 best_model.pth: {os.path.basename(path)}\n", log_file)

        # Window best: keeps early/mid/late representative checkpoints.
        if save_window_best and epoch >= candidate_start_epoch:
            window_id = (epoch - 1) // window_size
            window_start = window_id * window_size + 1
            window_end = min((window_id + 1) * window_size, epochs)
            window_key = f"{window_start:03d}_{window_end:03d}"

            # raw window
            if save_window_raw:
                raw_score = _metric_for_topk(raw_metrics, window_metric)
                wb_raw = state_tracker["window_best_raw"].get(window_key, {"score": -1e9, "path": None})
                if raw_score > wb_raw["score"]:
                    _safe_remove(wb_raw.get("path"))
                    path = os.path.join(exp_dir, f"window_best_raw_{window_key}_e{epoch:03d}_s{raw_score:.4f}.pth")
                    save_eval_checkpoint(path, epoch, model.state_dict(), raw_metrics, config, tag=f"window_best_raw_{window_key}")
                    state_tracker["window_best_raw"][window_key] = {"score": raw_score, "epoch": epoch, "path": path}
                    log_print(f"      🧩 更新 window_best_raw {window_key}: e{epoch}, score={raw_score:.4f}\n", log_file)

            # ema window
            if save_window_ema and ema_metrics is not None:
                ema_score = _metric_for_topk(ema_metrics, window_metric)
                wb_ema = state_tracker["window_best_ema"].get(window_key, {"score": -1e9, "path": None})
                if ema_score > wb_ema["score"]:
                    _safe_remove(wb_ema.get("path"))
                    path = os.path.join(exp_dir, f"window_best_ema_{window_key}_e{epoch:03d}_s{ema_score:.4f}.pth")
                    save_eval_checkpoint(path, epoch, ema.ema.state_dict(), ema_metrics, config, tag=f"window_best_ema_{window_key}")
                    state_tracker["window_best_ema"][window_key] = {"score": ema_score, "epoch": epoch, "path": path}
                    log_print(f"      🧩 更新 window_best_ema {window_key}: e{epoch}, score={ema_score:.4f}\n", log_file)

        # Top-k bank. This is supplementary; window-best is what keeps e57-like checkpoints.
        if save_top_k > 0 and epoch >= candidate_start_epoch:
            raw_score = _metric_for_topk(raw_metrics, top_k_metric)
            raw_top_path = os.path.join(exp_dir, f"topk_raw_e{epoch:03d}_s{raw_score:.4f}.pth")
            save_eval_checkpoint(raw_top_path, epoch, model.state_dict(), raw_metrics, config, tag="topk_raw")
            state_tracker["topk_raw"], removed = update_topk_bank(
                state_tracker["topk_raw"], save_top_k, raw_score, raw_top_path, epoch, kind="raw"
            )
            for item in removed:
                log_print(f"      🗑️ 移除 topk_raw: {os.path.basename(item['path'])}\n", log_file)

            if ema_metrics is not None:
                ema_score = _metric_for_topk(ema_metrics, top_k_metric)
                ema_top_path = os.path.join(exp_dir, f"topk_ema_e{epoch:03d}_s{ema_score:.4f}.pth")
                save_eval_checkpoint(ema_top_path, epoch, ema.ema.state_dict(), ema_metrics, config, tag="topk_ema")
                state_tracker["topk_ema"], removed = update_topk_bank(
                    state_tracker["topk_ema"], save_top_k, ema_score, ema_top_path, epoch, kind="ema"
                )
                for item in removed:
                    log_print(f"      🗑️ 移除 topk_ema: {os.path.basename(item['path'])}\n", log_file)

        # Early stopping uses balanced score by default, because Val IoU alone has shown bad test selection.
        monitor_score = ema_metrics["balanced_score"] if ema_metrics is not None else raw_metrics["balanced_score"]
        best_monitor = state_tracker.get("best_monitor_score", -1e9)
        if monitor_score > best_monitor:
            state_tracker["best_monitor_score"] = monitor_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        state_tracker["epochs_without_improvement"] = epochs_without_improvement
        state_tracker["last_epoch"] = epoch

        # Save latest every epoch.
        save_latest_checkpoint(latest_path, epoch, model, optimizer, scheduler, scaler, ema, state_tracker)
        save_json(state_tracker, os.path.join(exp_dir, "checkpoint_tracker.json"))

        if epochs_without_improvement > 0:
            log_print(f"      ⚠️ balanced monitor 连续 {epochs_without_improvement} 轮没有提升。\n", log_file)

        if epochs_without_improvement >= early_stop_patience:
            log_print(f"🚫 连续 {early_stop_patience} 轮 balanced monitor 未提升，触发早停。\n", log_file)
            break

    # End training: create averaged checkpoints.
    if create_weight_avg:
        try:
            raw_top_paths = [item["path"] for item in state_tracker.get("topk_raw", []) if os.path.isfile(item.get("path", ""))]
            ema_top_paths = [item["path"] for item in state_tracker.get("topk_ema", []) if os.path.isfile(item.get("path", ""))]

            if len(raw_top_paths) >= 2:
                out_path = os.path.join(exp_dir, f"top{len(raw_top_paths)}_raw_avg.pth")
                average_checkpoints(raw_top_paths, out_path, config, tag="topk_raw_avg")
                log_print(f"✅ 已生成权重平均: {out_path}\n", log_file)

            if len(ema_top_paths) >= 2:
                out_path = os.path.join(exp_dir, f"top{len(ema_top_paths)}_ema_avg.pth")
                average_checkpoints(ema_top_paths, out_path, config, tag="topk_ema_avg")
                log_print(f"✅ 已生成权重平均: {out_path}\n", log_file)
        except Exception as e:
            log_print(f"⚠️ 权重平均生成失败: {repr(e)}\n", log_file)

    # Recommended checkpoints file.
    try:
        rec_lines = []
        rec_lines.append("# Recommended checkpoints to test\n")
        for key in [
            "best_ema_balanced_path",
            "best_ema_iou_path",
            "best_raw_balanced_path",
            "best_raw_iou_path",
        ]:
            p = state_tracker.get(key)
            if p is not None:
                rec_lines.append(f"{key}: {p}\n")

        rec_lines.append("\n# Window EMA best checkpoints\n")
        for k, v in sorted(state_tracker.get("window_best_ema", {}).items()):
            rec_lines.append(f"{k}: epoch={v.get('epoch')} score={v.get('score'):.4f} path={v.get('path')}\n")

        rec_lines.append("\n# Window RAW best checkpoints\n")
        for k, v in sorted(state_tracker.get("window_best_raw", {}).items()):
            rec_lines.append(f"{k}: epoch={v.get('epoch')} score={v.get('score'):.4f} path={v.get('path')}\n")

        with open(os.path.join(exp_dir, "recommended_checkpoints.txt"), "w", encoding="utf-8") as f:
            f.writelines(rec_lines)
    except Exception as e:
        log_print(f"⚠️ recommended_checkpoints.txt 写入失败: {repr(e)}\n", log_file)

    end_time_raw = time.time()
    end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration_sec = end_time_raw - start_time_raw
    hours, rem = divmod(duration_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    duration_str = f"{int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒"

    end_msg = (
        f"--------------------------------------------------\n"
        f" 训练结束时间: {end_time_str}\n"
        f" 整个实验总耗时: {duration_str}\n"
        f"🎉 实验完成！前往 {exp_dir} 查看结果。\n"
        f"--------------------------------------------------\n"
    )
    log_print(end_msg, log_file)
    log_file.close()


if __name__ == "__main__":
    main()
