
# -*- coding: utf-8 -*-
"""
train_dinounet.py

放置位置：
    SEG/train_dinounet.py

用途：
    使用你现有 RoadSeg/SEG 训练风格训练 DinoRoadUNet。
    不走原作者 nnU-Net 流程，保持你的 Massachusetts Roads 数据、指标、保存逻辑不变。

运行示例：
    python train_dinounet.py -c configs/baseline/DinoRoadUNet_S_Mass.json

断点续训：
    python train_dinounet.py -c configs/baseline/DinoRoadUNet_S_Mass.json -r saved_runs/Mass/DinoRoadUNet_xxx/latest_model.pth
"""

import os
import cv2
cv2.setNumThreads(0)

import time
import datetime
import json
import argparse
import random
import copy
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from thop import profile
except Exception:
    profile = None

from dataloaders.road_dataset import RoadDataset
from dataloaders.drive_dataset import DRIVEDataset
from core.loss import BCEDiceLoss
from core.metrics import Evaluator
from models.custom.DinoRoadUNet import DinoRoadUNet


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
        json.dump(config, f, indent=4, ensure_ascii=False)

    return exp_dir


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
def safe_evaluate_model_complexity(model, device, img_size, enable_profile=False):
    """
    统计参数量、FPS，并尝试统计 FLOPs。
    对于 DinoRoadUNet，FPS 比 FLOPs 更可靠。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)

    flops = None
    fps = None

    # 1. 尝试统计 FLOPs
    if enable_profile:
        if profile is None:
            print("⚠️ 没有安装 thop，跳过 FLOPs 统计。")
        else:
            try:
                model_for_profile = copy.deepcopy(model).to(device)
                model_for_profile.eval()

                with torch.no_grad():
                    macs, _ = profile(model_for_profile, inputs=(dummy_input,), verbose=False)

                flops = macs

                del model_for_profile
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"⚠️ FLOPs 统计失败，继续统计 FPS。原因: {repr(e)}")

    # 2. 统计 FPS
    try:
        torch.cuda.empty_cache()
        model.eval()

        # 预热
        with torch.no_grad():
            for _ in range(20):
                _ = model(dummy_input)

        torch.cuda.synchronize()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        repetitions = 100
        starter.record()

        with torch.no_grad():
            for _ in range(repetitions):
                _ = model(dummy_input)

        ender.record()
        torch.cuda.synchronize()

        total_time_ms = starter.elapsed_time(ender)
        avg_time_ms = total_time_ms / repetitions
        fps = 1000.0 / avg_time_ms

        print(f"✅ FPS 统计完成：单张平均耗时 {avg_time_ms:.2f} ms，FPS {fps:.2f}")

    except Exception as e:
        print(f"⚠️ FPS 统计失败。原因: {repr(e)}")

    model.train()
    return total_params, trainable_params, flops, fps




def build_dataset(config, mode):
    dataset_name = config["dataset"]["name"]
    root_path = config["dataset"]["root_path"]
    img_size = config["dataset"].get("input_size", 1024)

    if dataset_name == "DRIVE":
        return DRIVEDataset(root_path, dataset_name, mode=mode, img_size=img_size)

    return RoadDataset(root_path, dataset_name, mode=mode, img_size=img_size)


def build_model(config, img_size):
    mcfg = config["model"]

    if mcfg.get("name") != "DinoRoadUNet":
        raise ValueError("train_dinounet.py 只建议用于 DinoRoadUNet。请检查 config['model']['name']。")

    model = DinoRoadUNet(
        num_classes=mcfg.get("num_classes", 1),
        dinov3_model=mcfg.get("dinov3_model", "dinounet_s"),
        pretrained_path=mcfg.get("pretrained_path"),
        out_channels=mcfg.get("out_channels", [64, 128, 256, 512]),
        rank=mcfg.get("rank", 256),
        img_size=img_size,
        freeze_backbone=mcfg.get("freeze_backbone", True),
        imagenet_norm=mcfg.get("imagenet_norm", True),
        input_already_normalized=mcfg.get("input_already_normalized", False),
        conv_inplane=mcfg.get("conv_inplane", 64),
        deform_num_heads=mcfg.get("deform_num_heads", 16),
        n_points=mcfg.get("n_points", 4),
        with_cp=mcfg.get("with_cp", False),
    )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
    parser.add_argument("-r", "--resume", type=str, default=None, help="Path to latest_model.pth to resume")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    seed = config["training"].get("seed", 3407)
    set_seed(seed)

    if args.resume and os.path.isfile(args.resume):
        exp_dir = os.path.dirname(args.resume)
        log_file = open(os.path.join(exp_dir, "train.log"), "a")
        resume_msg = f"\n{'='*50}\n🔌 触发断点续训，继续向原目录记录日志\n{'='*50}\n"
        print(resume_msg, end="")
        log_file.write(resume_msg)
    else:
        exp_dir = create_experiment_dir(config)
        log_file = open(os.path.join(exp_dir, "train.log"), "w")

    start_time_raw = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    init_msg = f"实验开始，日志和权重保存在: {exp_dir}\n训练开始时间: {start_time_str}\n"
    print(init_msg, end="")
    log_file.write(init_msg)
    log_file.flush()

    dataset_name = config["dataset"]["name"]
    img_size = config["dataset"].get("input_size", 1024)
    n_workers = config["dataset"].get("num_workers", 8)
    batch_size = config["dataset"].get("batch_size", 1)

    train_dataset = build_dataset(config, mode="train")
    val_dataset = build_dataset(config, mode="val")

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
        drop_last=True,
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

    model = build_model(config, img_size=img_size).cuda()

    profile_complexity = config["training"].get("profile_complexity", False)
    total_params, trainable_params, flops, fps = safe_evaluate_model_complexity(
        model,
        device="cuda",
        img_size=img_size,
        enable_profile=profile_complexity,
    )

    complexity_msg = (
        f"--------------------------------------------------\n"
        f"📊 模型复杂度 @ 输入尺寸: {img_size}x{img_size}\n"
        f"   - 总参数量 (Total Params):      {total_params / 1e6:.2f} M\n"
        f"   - 可训练参数量 (Trainable):     {trainable_params / 1e6:.2f} M\n"
    )
    if flops is not None:
        complexity_msg += f"   - 浮点运算量 (FLOPs):          {flops / 1e9:.2f} G\n"
    else:
        complexity_msg += f"   - 浮点运算量 (FLOPs):          未统计，避免 MSDeformAttn 与 thop 冲突\n"

    if fps is not None:
        complexity_msg += f"   - 推理速度 (FPS):              {fps:.2f} 张/秒\n"
    else:
        complexity_msg += f"   - 推理速度 (FPS):              未统计\n"

    complexity_msg += f"--------------------------------------------------\n"
    print(complexity_msg, end="")
    log_file.write(complexity_msg)
    log_file.flush()

    tcfg = config["training"]
    lr = tcfg.get("lr", 1e-4)
    weight_decay = tcfg.get("weight_decay", 1e-4)
    lr_factor = tcfg.get("lr_factor", 0.5)
    lr_patience = tcfg.get("lr_patience", 8)
    early_stop_patience = tcfg.get("early_stop_patience", 50)
    accumulation_steps = tcfg.get("accumulation_steps", 1)
    grad_clip = tcfg.get("grad_clip", 0.0)

    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params_list, lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_factor,
        patience=lr_patience,
    )

    criterion = BCEDiceLoss().cuda()
    scaler = torch.amp.GradScaler("cuda")
    evaluator = Evaluator(num_class=2)

    hyper_msg = (
        f"训练超参数设置:\n"
        f"   - Dataset: {dataset_name}\n"
        f"   - Batch Size: {batch_size}\n"
        f"   - Accumulation Steps: {accumulation_steps}\n"
        f"   - LR: {lr}\n"
        f"   - Weight Decay: {weight_decay}\n"
        f"   - Scheduler: ReduceLROnPlateau(factor={lr_factor}, patience={lr_patience})\n"
        f"   - Early Stop Patience: {early_stop_patience}\n"
        f"   - Grad Clip: {grad_clip}\n"
        f"--------------------------------------------------\n"
    )
    print(hyper_msg, end="")
    log_file.write(hyper_msg)
    log_file.flush()

    start_epoch = 1
    best_val_iou = 0.0
    epochs_without_improvement = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"🔄 发现断点文件，正在恢复: {args.resume}")
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)

            start_epoch = checkpoint["epoch"] + 1
            best_val_iou = checkpoint.get("best_val_iou", 0.0)
            epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)

            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            msg = f"成功恢复，将从第 {start_epoch} 轮继续训练。\n"
            print(msg, end="")
            log_file.write(msg)
            log_file.flush()
        else:
            print(f"找不到断点文件: {args.resume}，将从头开始训练。")

    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        model.train()
        train_loss = 0.0

        optimizer.zero_grad(set_to_none=True)
        train_loader_tqdm = tqdm(
            train_loader,
            desc=f"Epoch [{epoch}/{tcfg['epochs']}] Train",
            leave=False,
        )

        for step, batch_data in enumerate(train_loader_tqdm, start=1):
            if len(batch_data) == 3:
                imgs, masks, _ = batch_data
            else:
                imgs, masks = batch_data

            imgs = imgs.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)

            with torch.amp.autocast("cuda"):
                preds = model(imgs)
                if isinstance(preds, (tuple, list)):
                    preds = preds[0]
                loss = criterion(preds, masks)
                loss_to_backward = loss / accumulation_steps

            scaler.scale(loss_to_backward).backward()

            if step % accumulation_steps == 0 or step == len(train_loader):
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params_list, grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item()
            train_loader_tqdm.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        evaluator.reset()

        with torch.no_grad():
            val_loader_tqdm = tqdm(
                val_loader,
                desc=f"Epoch [{epoch}/{tcfg['epochs']}] Valid",
                leave=False,
            )

            for batch_data in val_loader_tqdm:
                if len(batch_data) == 3:
                    imgs, masks, _ = batch_data
                else:
                    imgs, masks = batch_data

                imgs = imgs.cuda(non_blocking=True)
                masks = masks.cuda(non_blocking=True)

                with torch.amp.autocast("cuda"):
                    preds = model(imgs)
                    if isinstance(preds, (tuple, list)):
                        preds = preds[0]
                    loss = criterion(preds, masks)

                val_loss += loss.item()

                preds_bin = (torch.sigmoid(preds) > 0.5).cpu().numpy().astype(int)
                masks_int = masks.cpu().numpy().astype(int)
                evaluator.add_batch(masks_int, preds_bin)

                val_loader_tqdm.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_val_loss = val_loss / len(val_loader)

        _ = evaluator.Pixel_Precision()
        _ = evaluator.Pixel_Recall()
        val_f1 = evaluator.Pixel_F1()
        val_iou = evaluator.Intersection_over_Union()

        log_msg = (
            f"Epoch [{epoch}/{tcfg['epochs']}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
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
            "epochs_without_improvement": epochs_without_improvement,
            "config": config,
        }

        torch.save(checkpoint, os.path.join(exp_dir, "latest_model.pth"))

        if is_best:
            torch.save(checkpoint, os.path.join(exp_dir, "best_model.pth"))
            msg = f"      已保存最佳权重！Best Val IoU: {best_val_iou:.4f}\n"
        else:
            msg = f"      ⚠️ 连续 {epochs_without_improvement} 轮没有提升。\n"

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
        f"实验完成！前往 {exp_dir} 查看结果。\n"
        f"--------------------------------------------------\n"
    )

    print(end_msg, end="")
    log_file.write(end_msg)
    log_file.close()


if __name__ == "__main__":
    main()
