import os
import json
import glob
import argparse
import datetime
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.road_dataset import RoadDataset
from dataloaders.drive_dataset import DRIVEDataset
from models import get_model


class DatasetLevelEvaluator:
    """Dataset-level TP/FP/FN evaluator for road class."""
    def __init__(self, thresholds):
        self.thresholds = [float(t) for t in thresholds]
        self.stats = {
            t: {"TP": 0.0, "FP": 0.0, "FN": 0.0, "pred_fg": 0.0, "gt_fg": 0.0, "pixels": 0.0, "images": 0}
            for t in self.thresholds
        }

    @torch.no_grad()
    def update(self, probs, targets):
        targets = targets.float()
        gt_fg = targets.sum().item()
        pixels = targets.numel()
        bsz = targets.shape[0]

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
                "TP": tp,
                "FP": fp,
                "FN": fn
            }
        return out


def parse_thresholds(text):
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    new_state = OrderedDict()
    changed = False
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
            changed = True
        else:
            new_state[k] = v
    return new_state if changed else state_dict


def load_checkpoint_state(weight_path, prefer_ema=False):
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    state_type = "raw_state_dict"

    if prefer_ema and isinstance(ckpt, dict) and "ema_state_dict" in ckpt:
        state = ckpt["ema_state_dict"]
        state_type = "ema_state_dict"
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        state_type = "model_state_dict"
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
        state_type = "state_dict"
    else:
        state = ckpt

    info = {}
    if isinstance(ckpt, dict):
        for key in ["epoch", "metrics", "score"]:
            if key in ckpt:
                info[key] = ckpt[key]

    return strip_module_prefix(state), state_type, info


def build_dataset(config, mode="test"):
    name = config["dataset"]["name"]
    root = config["dataset"]["root_path"]
    img_size = config["dataset"].get("input_size", 1024)

    if name.lower() == "drive":
        return DRIVEDataset(root, name, mode=mode, img_size=img_size)
    return RoadDataset(root, name, mode=mode, img_size=img_size)


def sample_filter_by_mask(imgs, masks, names=None, mode="full"):
    if mode == "full":
        return imgs, masks, names, int(imgs.shape[0]), int(imgs.shape[0])

    flat_sum = masks.view(masks.shape[0], -1).sum(dim=1)
    if mode == "nonblack":
        keep = flat_sum > 0
    elif mode == "black":
        keep = flat_sum == 0
    else:
        raise ValueError(f"Unknown eval mode: {mode}")

    kept = int(keep.sum().item())
    total = int(keep.numel())

    if kept == 0:
        return None, None, None, kept, total

    imgs = imgs[keep]
    masks = masks[keep]
    if names is not None and isinstance(names, (list, tuple)):
        names = [n for n, k in zip(names, keep.cpu().tolist()) if k]
    else:
        names = None
    return imgs, masks, names, kept, total


def collect_candidate_weights(weight_dir):
    patterns = [
        "best_model.pth",
        "best_ema_balanced*.pth",
        "best_ema_iou*.pth",
        "top5_ema_avg.pth",

        "window_best_ema_041_060*.pth",
        "window_best_ema_021_040*.pth",
        "window_best_ema_061_080*.pth",
        "window_best_ema_001_020*.pth",
        "window_best_ema_081_100*.pth",

        "best_raw_balanced*.pth",
        "best_raw_iou*.pth",
        "top5_raw_avg.pth",

        "window_best_raw_041_060*.pth",
        "window_best_raw_021_040*.pth",
        "window_best_raw_061_080*.pth",
        "window_best_raw_001_020*.pth",
        "window_best_raw_081_100*.pth"
    ]

    found, seen = [], set()
    for pat in patterns:
        for p in sorted(glob.glob(os.path.join(weight_dir, pat))):
            ap = os.path.abspath(p)
            if ap not in seen and os.path.isfile(ap):
                found.append(ap)
                seen.add(ap)
    return found


def evaluate_one_weight(config, weight_path, args, log_file):
    img_size = config["dataset"].get("input_size", 1024)
    dataset = build_dataset(config, mode=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = get_model(config["model"], img_size=img_size).cuda()
    state, state_type, ckpt_info = load_checkpoint_state(weight_path, prefer_ema=args.prefer_ema_from_latest)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    thresholds = parse_thresholds(args.thresholds)
    evaluator = DatasetLevelEvaluator(thresholds)

    total_seen, total_used, total_skipped = 0, 0, 0

    with torch.no_grad():
        for batch_data in tqdm(loader, desc=f"Testing {os.path.basename(weight_path)} [{args.eval_mode}]", leave=False):
            if len(batch_data) == 3:
                imgs, masks, names = batch_data
            else:
                imgs, masks = batch_data
                names = None

            imgs = imgs.cuda(non_blocking=True)
            masks = masks.cuda(non_blocking=True)

            imgs, masks, names, kept, total = sample_filter_by_mask(imgs, masks, names, mode=args.eval_mode)
            total_seen += total
            total_used += kept
            total_skipped += total - kept

            if imgs is None:
                continue

            with torch.amp.autocast("cuda", enabled=args.amp):
                logits = model(imgs)
                if isinstance(logits, (tuple, list)):
                    logits = logits[0]
                elif isinstance(logits, dict):
                    logits = logits.get("final_logits", next(iter(logits.values())))

            probs = torch.sigmoid(logits)
            evaluator.update(probs, masks)

    metrics = evaluator.get_metrics()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg = (
        f"\n==================================================\n"
        f" ⏱测试时间: {now}\n"
        f" 权重路径: {weight_path}\n"
        f" 加载字段: {state_type}\n"
        f" 数据集:   {config['dataset']['name']}\n"
        f" Split:    {args.split}\n"
        f" EvalMode: {args.eval_mode}  (seen={total_seen}, used={total_used}, skipped={total_skipped})\n"
        f" 模型:     {config['model']['name']}\n"
        f" Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}\n"
    )
    if ckpt_info:
        msg += f" Checkpoint info: {ckpt_info}\n"
    msg += "--------------------------------------------------\n"

    best_t, best_iou = None, -1.0
    for t in thresholds:
        m = metrics[t]
        if m["iou"] > best_iou:
            best_iou = m["iou"]
            best_t = t

        msg += (
            f" threshold={t:.3f} | "
            f"P={m['precision']*100:.2f}% | "
            f"R={m['recall']*100:.2f}% | "
            f"F1={m['f1']*100:.2f}% | "
            f"IoU={m['iou']*100:.2f}% | "
            f"PredFG={m['pred_fg_ratio']*100:.3f}% | "
            f"GtFG={m['gt_fg_ratio']*100:.3f}% | "
            f"Images={m['images']}\n"
        )

    msg += (
        f"--------------------------------------------------\n"
        f" Best threshold by IoU in this run: {best_t:.3f}, IoU={best_iou*100:.2f}%\n"
        f"==================================================\n\n"
    )

    print(msg)
    log_file.write(msg)
    log_file.flush()

    return {"weight": weight_path, "best_threshold": best_t, "best_iou": best_iou, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config json")
    parser.add_argument("-w", "--weight", type=str, default=None, help="Single checkpoint path")
    parser.add_argument("--weight_dir", type=str, default=None, help="Directory containing checkpoints")
    parser.add_argument("--auto_candidates", action="store_true", help="Auto collect recommended HLD checkpoints from weight_dir")

    parser.add_argument("--split", type=str, default="test", choices=["test", "val"])
    parser.add_argument("--eval_mode", type=str, default="full", choices=["full", "nonblack", "black"])
    parser.add_argument("--thresholds", type=str, default="0.5",
                        help="Comma-separated thresholds, e.g. 0.35,0.4,0.45,0.5,0.55,0.6")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--prefer_ema_from_latest", action="store_true",
                        help="When evaluating latest_model.pth, load ema_state_dict if it exists")
    parser.add_argument("--log_name", type=str, default="test_HLD_results.log")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    if args.weight is None and args.weight_dir is None:
        raise ValueError("Please provide either --weight or --weight_dir")

    if args.weight is not None:
        weights = [os.path.abspath(args.weight)]
        save_dir = os.path.dirname(os.path.abspath(args.weight))
    else:
        save_dir = os.path.abspath(args.weight_dir)
        if args.auto_candidates:
            weights = collect_candidate_weights(save_dir)
        else:
            weights = sorted(glob.glob(os.path.join(save_dir, "*.pth")))

    if len(weights) == 0:
        raise FileNotFoundError(f"No checkpoint found. weight={args.weight}, weight_dir={args.weight_dir}")

    log_path = os.path.join(save_dir, args.log_name)
    summary = []

    with open(log_path, "a", encoding="utf-8") as log_file:
        header = (
            f"\n\n########## HLD Test Run ##########\n"
            f"Config: {args.config}\n"
            f"Weights: {len(weights)}\n"
            f"Split: {args.split}, EvalMode: {args.eval_mode}, Thresholds: {args.thresholds}\n"
            f"Log: {log_path}\n"
            f"##################################\n"
        )
        print(header)
        log_file.write(header)

        for w in weights:
            if not os.path.isfile(w):
                print(f"[Skip] Missing checkpoint: {w}")
                continue
            try:
                summary.append(evaluate_one_weight(config, w, args, log_file))
            except Exception as e:
                err = f"\n[ERROR] Failed evaluating {w}: {repr(e)}\n"
                print(err)
                log_file.write(err)
                log_file.flush()

        if summary:
            ranked = sorted(summary, key=lambda x: x["best_iou"], reverse=True)
            rank_msg = "\n================ HLD Test Summary Ranking ================\n"
            for i, r in enumerate(ranked, 1):
                rank_msg += f"{i:02d}. IoU={r['best_iou']*100:.2f}% @thr={r['best_threshold']:.3f} | {r['weight']}\n"
            rank_msg += "==========================================================\n"
            print(rank_msg)
            log_file.write(rank_msg)

    print(f"已保存测试结果到: {log_path}")


if __name__ == "__main__":
    main()
