import os
import cv2
cv2.setNumThreads(0)

import time
import json
import copy
import random
import argparse
import datetime
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.amp

from tqdm import tqdm
from thop import profile
from torch.utils.data import DataLoader

from dataloaders.road_dataset import RoadDataset
from dataloaders.drive_dataset import DRIVEDataset
from models import get_model
from core.loss import BCEDiceLoss
from core.metrics import Evaluator


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


def create_experiment_dir(config):
    dataset_name = config["dataset"]["name"]
    model_name = config["model"]["name"]
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    exp_dir = os.path.join("saved_runs", dataset_name, f"{model_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    return exp_dir


def evaluate_model_complexity(model, device, img_size):
    """
    这里只统计测试推理复杂度。
    因为 DC_v2.forward() 只走 HL_base，所以复杂度会接近 HL_base。
    DINO prior 只在训练 forward_train() 里使用。
    """
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    model_for_profile = copy.deepcopy(model)
    macs, params = profile(model_for_profile, inputs=(dummy_input,), verbose=False)
    flops = macs
    del model_for_profile

    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    torch.cuda.synchronize()
    end_time = time.time()

    fps = 50.0 / (end_time - start_time)

    model.train()
    return params, flops, fps


class PriorGuidedBCEDiceLoss(nn.Module):
    """
    用 DINO prior 生成一个稳定的权重图。
    这个 loss 只更新 HL_base 主分支。
    prior_prob 必须 detach，避免 main loss 反向干扰 DINO。
    """

    def __init__(self, strength=0.5, eps=1e-6):
        super().__init__()
        self.strength = float(strength)
        self.eps = eps

    def forward(self, logits, targets, prior_prob):
        if prior_prob.shape[-2:] != targets.shape[-2:]:
            prior_prob = F.interpolate(
                prior_prob,
                size=targets.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        prior_prob = prior_prob.detach()
        targets = targets.float()

        # 道路像素：DINO 越认为是道路，权重越高
        # 背景像素：DINO 越认为是背景，权重越高
        # 这样只强化 DINO 和 GT 一致的区域，减少错误 prior 的破坏
        pos_weight = targets * (0.5 + prior_prob)
        neg_weight = (1.0 - targets) * (0.5 + (1.0 - prior_prob))
        weight = 1.0 + self.strength * (pos_weight + neg_weight)

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        bce = (bce * weight).mean()

        probs = torch.sigmoid(logits)

        intersection = (weight * probs * targets).sum(dim=(1, 2, 3))
        union = (weight * probs).sum(dim=(1, 2, 3)) + (weight * targets).sum(dim=(1, 2, 3))
        dice = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)
        dice = dice.mean()

        return bce + dice


def build_optimizer(model, config):
    training_cfg = config["training"]
    lr = training_cfg.get("lr", 1e-4)
    weight_decay = training_cfg.get("weight_decay", 1e-4)

    # prior-only 先统一学习率，避免分组学习率导致主干太慢
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    return optimizer


def extract_outputs(model, imgs):
    """
    如果模型有 forward_train，就用 forward_train。
    否则退化成普通 forward。
    """
    if hasattr(model, "forward_train"):
        out = model.forward_train(imgs)
    else:
        out = model(imgs)

    if isinstance(out, dict):
        return out

    return {
        "final_logits": out,
        "base_logits": None,
        "dino_prior_logits": None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("-r", "--resume", type=str, default=None, help="断点路径 latest_model.pth")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    seed = config["training"].get("seed", 3407)
    set_seed(seed)

    if args.resume and os.path.isfile(args.resume):
        exp_dir = os.path.dirname(args.resume)
        log_file = open(os.path.join(exp_dir, "train.log"), "a")
        resume_msg = f"\n{'=' * 50}\n触发断点续训\n{'=' * 50}\n"
        print(resume_msg, end="")
        log_file.write(resume_msg)
    else:
        exp_dir = create_experiment_dir(config)
        log_file = open(os.path.join(exp_dir, "train.log"), "w")

    start_time_raw = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    init_msg = (
        f"实验开始，日志和权重保存在: {exp_dir}\n"
        f"训练开始时间: {start_time_str}\n"
        f"--------------------------------------------------\n"
    )
    print(init_msg, end="")
    log_file.write(init_msg)
    log_file.flush()

    dataset_name = config["dataset"]["name"]
    root_path = config["dataset"]["root_path"]
    img_size = config["dataset"].get("input_size", 1024)
    batch_size = config["dataset"].get("batch_size", 8)
    n_workers = config["dataset"].get("num_workers", 8)

    if dataset_name == "DRIVE":
        train_dataset = DRIVEDataset(root_path, dataset_name, mode="train", img_size=img_size)
        val_dataset = DRIVEDataset(root_path, dataset_name, mode="val", img_size=img_size)
    else:
        train_dataset = RoadDataset(root_path, dataset_name, mode="train", img_size=img_size)
        val_dataset = RoadDataset(root_path, dataset_name, mode="val", img_size=img_size)

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
        f"📦 数据集信息\n"
        f"    - 数据集: {dataset_name}\n"
        f"    - 根目录: {root_path}\n"
        f"    - 输入尺寸: {img_size}x{img_size}\n"
        f"    - 训练样本数: {len(train_dataset)}\n"
        f"    - 验证样本数: {len(val_dataset)}\n"
        f"    - Batch Size: {batch_size}\n"
        f"    - Num Workers: {n_workers}\n"
        f"--------------------------------------------------\n"
    )
    print(data_msg, end="")
    log_file.write(data_msg)
    log_file.flush()

    model = get_model(config["model"], img_size=img_size).cuda()

    try:
        params, flops, fps = evaluate_model_complexity(model, device="cuda", img_size=img_size)
        complexity_msg = (
            f"📊 模型复杂度 @ 输入尺寸: {img_size}x{img_size}\n"
            f"    - 参数量 (Params):    {params / 1e6:.2f} M\n"
            f"    - 浮点运算量 (FLOPs): {flops / 1e9:.2f} G\n"
            f"    - 推理速度 (FPS):     {fps:.2f} 张/秒\n"
            f"--------------------------------------------------\n"
        )
    except Exception as e:
        complexity_msg = (
            f"📊 模型复杂度统计失败: {str(e)}\n"
            f"--------------------------------------------------\n"
        )

    print(complexity_msg, end="")
    log_file.write(complexity_msg)
    log_file.flush()

    training_cfg = config["training"]

    weight_decay = training_cfg.get("weight_decay", 1e-4)
    lr = training_cfg.get("lr", 1e-4)
    lr_factor = training_cfg.get("lr_factor", 0.5)
    lr_patience = training_cfg.get("lr_patience", 8)
    early_stop_patience = training_cfg.get("early_stop_patience", 50)

    loss_weights = training_cfg.get("loss_weights", {})
    dino_prior_weight = loss_weights.get("dino_prior", 0.05)
    prior_guided_weight = loss_weights.get("prior_guided", 0.2)
    prior_guided_strength = loss_weights.get("prior_guided_strength", 0.5)

    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )

    criterion = BCEDiceLoss().cuda()
    prior_guided_criterion = PriorGuidedBCEDiceLoss(
        strength=prior_guided_strength,
    ).cuda()

    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    best_val_iou = 0.0
    best_val_f1 = 0.0
    epochs_without_improvement = 0

    evaluator = Evaluator(num_class=2)

    hyper_msg = (
        f"训练超参数设置:\n"
        f"    - Optimizer: AdamW\n"
        f"    - Optimizer 参数: 统一学习率\n"
        f"    - 初始学习率 (LR): {lr}\n"
        f"    - 权重衰减 (Weight Decay): {weight_decay}\n"
        f"    - 学习率调度: ReduceLROnPlateau(factor={lr_factor}, patience={lr_patience})\n"
        f"    - 早停机制: {early_stop_patience} 轮\n"
        f"    - 损失权重: dino_prior={dino_prior_weight}, prior_guided={prior_guided_weight}, prior_guided_strength={prior_guided_strength}\n"
        f"--------------------------------------------------\n"
    )
    print(hyper_msg, end="")
    log_file.write(hyper_msg)
    log_file.flush()

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, weights_only=False)
            start_epoch = checkpoint["epoch"] + 1
            best_val_iou = checkpoint.get("best_val_iou", 0.0)
            best_val_f1 = checkpoint.get("best_val_f1", 0.0)
            epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            msg = f"成功恢复断点，将从第 {start_epoch} 轮继续训练。\n"
            print(msg, end="")
            log_file.write(msg)
            log_file.flush()
        else:
            print(f"找不到断点文件: {args.resume}，将从头开始训练。")

    total_epochs = training_cfg["epochs"]

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()

        train_loss = 0.0
        train_main = 0.0
        train_dino = 0.0
        train_prior = 0.0

        train_tqdm = tqdm(
            train_loader,
            desc=f"Epoch [{epoch}/{total_epochs}] Train",
            leave=False,
        )

        for batch_data in train_tqdm:
            if len(batch_data) == 3:
                imgs, masks, _ = batch_data
            else:
                imgs, masks = batch_data

            imgs = imgs.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                outputs = extract_outputs(model, imgs)

                main_logits = outputs["final_logits"]
                dino_prior_logits = outputs.get("dino_prior_logits", None)

                main_loss = criterion(main_logits, masks)

                dino_loss = torch.tensor(0.0, device=imgs.device)
                prior_guided_loss = torch.tensor(0.0, device=imgs.device)

                if dino_prior_logits is not None:
                    dino_prior_up = F.interpolate(
                        dino_prior_logits,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                    dino_loss = criterion(dino_prior_up, masks)

                    prior_prob = torch.sigmoid(dino_prior_up).detach()
                    prior_guided_loss = prior_guided_criterion(
                        main_logits,
                        masks,
                        prior_prob,
                    )

                loss = (
                    main_loss
                    + dino_prior_weight * dino_loss
                    + prior_guided_weight * prior_guided_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_main += main_loss.item()
            train_dino += dino_loss.item()
            train_prior += prior_guided_loss.item()

            train_tqdm.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "main": f"{main_loss.item():.4f}",
                    "dino": f"{dino_loss.item():.4f}",
                    "prior": f"{prior_guided_loss.item():.4f}",
                }
            )

        avg_train_loss = train_loss / len(train_loader)
        avg_train_main = train_main / len(train_loader)
        avg_train_dino = train_dino / len(train_loader)
        avg_train_prior = train_prior / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_main = 0.0
        val_dino = 0.0
        val_prior = 0.0

        evaluator.reset()

        with torch.no_grad():
            val_tqdm = tqdm(
                val_loader,
                desc=f"Epoch [{epoch}/{total_epochs}] Valid",
                leave=False,
            )

            for batch_data in val_tqdm:
                if len(batch_data) == 3:
                    imgs, masks, _ = batch_data
                else:
                    imgs, masks = batch_data

                imgs = imgs.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)

                with torch.amp.autocast("cuda"):
                    outputs = extract_outputs(model, imgs)

                    main_logits = outputs["final_logits"]
                    dino_prior_logits = outputs.get("dino_prior_logits", None)

                    main_loss = criterion(main_logits, masks)

                    dino_loss = torch.tensor(0.0, device=imgs.device)
                    prior_guided_loss = torch.tensor(0.0, device=imgs.device)

                    if dino_prior_logits is not None:
                        dino_prior_up = F.interpolate(
                            dino_prior_logits,
                            size=masks.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )

                        dino_loss = criterion(dino_prior_up, masks)

                        prior_prob = torch.sigmoid(dino_prior_up).detach()
                        prior_guided_loss = prior_guided_criterion(
                            main_logits,
                            masks,
                            prior_prob,
                        )

                    loss = (
                        main_loss
                        + dino_prior_weight * dino_loss
                        + prior_guided_weight * prior_guided_loss
                    )

                val_loss += loss.item()
                val_main += main_loss.item()
                val_dino += dino_loss.item()
                val_prior += prior_guided_loss.item()

                preds_bin = (torch.sigmoid(main_logits) > 0.5).cpu().numpy().astype(int)
                masks_int = masks.cpu().numpy().astype(int)
                evaluator.add_batch(masks_int, preds_bin)

                val_tqdm.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "main": f"{main_loss.item():.4f}",
                    }
                )

        avg_val_loss = val_loss / len(val_loader)
        avg_val_main = val_main / len(val_loader)
        avg_val_dino = val_dino / len(val_loader)
        avg_val_prior = val_prior / len(val_loader)

        val_precision = evaluator.Pixel_Precision()
        val_recall = evaluator.Pixel_Recall()
        val_f1 = evaluator.Pixel_F1()
        val_iou = evaluator.Intersection_over_Union()

        current_lr = optimizer.param_groups[0]["lr"]

        log_msg = (
            f"Epoch [{epoch}/{total_epochs}] | LR: {current_lr:.8g} | "
            f"Train Loss: {avg_train_loss:.4f} "
            f"(main {avg_train_main:.4f}, dino {avg_train_dino:.4f}, prior {avg_train_prior:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} "
            f"(main {avg_val_main:.4f}, dino {avg_val_dino:.4f}, prior {avg_val_prior:.4f}) | "
            f"Val Precision: {val_precision:.4f} | "
            f"Val Recall: {val_recall:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val F1: {val_f1:.4f}\n"
        )

        print(log_msg, end="")
        log_file.write(log_msg)
        log_file.flush()

        scheduler.step(avg_val_loss)

        is_best = val_iou > best_val_iou

        if is_best:
            best_val_iou = val_iou
            best_val_f1 = val_f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_iou": best_val_iou,
            "best_val_f1": best_val_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "config": config,
        }

        torch.save(checkpoint, os.path.join(exp_dir, "latest_model.pth"))

        if is_best:
            torch.save(checkpoint, os.path.join(exp_dir, "best_model.pth"))
            msg = (
                f"      ✅ 已保存最佳权重！"
                f"Best IoU: {best_val_iou:.4f}, Best F1: {best_val_f1:.4f}\n"
            )
        else:
            msg = (
                f"     ⚠️ 连续 {epochs_without_improvement} 轮 IoU 没有提升。"
                f"当前最佳 IoU: {best_val_iou:.4f}\n"
            )

        print(msg, end="")
        log_file.write(msg)
        log_file.flush()

        if epochs_without_improvement >= early_stop_patience:
            msg = f"🚫 连续 {early_stop_patience} 轮 IoU 未提升，触发早停机制，训练提前结束！\n"
            print(msg, end="")
            log_file.write(msg)
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
        f"最佳验证 IoU: {best_val_iou:.4f}\n"
        f"最佳验证 F1:  {best_val_f1:.4f}\n"
        f"🎉 实验完成！前往 {exp_dir} 查看结果。\n"
        f"--------------------------------------------------\n"
    )

    print(end_msg, end="")
    log_file.write(end_msg)
    log_file.close()


if __name__ == "__main__":
    main()