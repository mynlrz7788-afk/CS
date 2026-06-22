import os
import json
import argparse
import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import get_model


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def extract_logits(outputs):
    if torch.is_tensor(outputs):
        return outputs

    if isinstance(outputs, dict):
        for k in ("final_logits", "logits", "out", "pred", "prediction", "fused_logits"):
            if k in outputs and torch.is_tensor(outputs[k]):
                return outputs[k]
        for v in outputs.values():
            if torch.is_tensor(v) and v.dim() == 4 and v.shape[1] == 1:
                return v

    if isinstance(outputs, (tuple, list)):
        return outputs[0]

    raise RuntimeError(f"无法从模型输出中提取 logits: {type(outputs)}")


def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"无法读取图像: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_binary(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取 mask: {path}")
    return (mask > 0).astype(np.uint8)


def normalize_image(image):
    image = image.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    image = (image - mean) / std
    image = image.transpose(2, 0, 1)

    return torch.from_numpy(image).float()


def get_2x2_center_positions(h, w, crop_size):
    max_y = max(0, h - crop_size)
    max_x = max(0, w - crop_size)

    positions = [
        (0, 0),
        (0, max_x),
        (max_y, 0),
        (max_y, max_x),
        (int(round(max_y / 2)), int(round(max_x / 2))),
    ]

    return sorted(set(positions))


def pad_to_crop(image, gt, fov, crop_size):
    h, w = fov.shape

    pad_h = max(0, crop_size - h)
    pad_w = max(0, crop_size - w)

    if pad_h == 0 and pad_w == 0:
        return image, gt, fov, h, w

    image = np.pad(
        image,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    gt = np.pad(
        gt,
        ((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=0,
    )
    fov = np.pad(
        fov,
        ((0, pad_h), (0, pad_w)),
        mode="constant",
        constant_values=0,
    )

    return image, gt, fov, h, w


@torch.no_grad()
def predict_full_image(model, image, crop_size, device):
    h, w = image.shape[:2]

    dummy_gt = np.zeros((h, w), dtype=np.uint8)
    dummy_fov = np.ones((h, w), dtype=np.uint8)

    image_pad, _, _, orig_h, orig_w = pad_to_crop(
        image, dummy_gt, dummy_fov, crop_size
    )

    hp, wp = image_pad.shape[:2]

    prob_sum = np.zeros((hp, wp), dtype=np.float32)
    count_map = np.zeros((hp, wp), dtype=np.float32)

    positions = get_2x2_center_positions(hp, wp, crop_size)

    for y, x in positions:
        crop = image_pad[y:y + crop_size, x:x + crop_size]

        tensor = normalize_image(crop).unsqueeze(0).to(device)

        outputs = model(tensor)
        logits = extract_logits(outputs)

        if logits.shape[-2:] != (crop_size, crop_size):
            logits = F.interpolate(
                logits,
                size=(crop_size, crop_size),
                mode="bilinear",
                align_corners=False,
            )

        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

        prob_sum[y:y + crop_size, x:x + crop_size] += prob
        count_map[y:y + crop_size, x:x + crop_size] += 1.0

    prob_full = prob_sum / np.maximum(count_map, 1e-6)
    prob_full = prob_full[:orig_h, :orig_w]

    return prob_full


class FOVMetric:
    def __init__(self):
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0

    def update(self, pred_bin, gt, fov):
        valid = fov > 0

        pred = pred_bin[valid].astype(np.uint8)
        target = gt[valid].astype(np.uint8)

        self.tp += np.logical_and(pred == 1, target == 1).sum()
        self.fp += np.logical_and(pred == 1, target == 0).sum()
        self.fn += np.logical_and(pred == 0, target == 1).sum()

    def get(self):
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        return precision, recall, f1, iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-w", "--weight", type=str, required=True)
    parser.add_argument("--split", type=str, default="test_full", choices=["val_full", "test_full"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save_pred", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    dataset_name = config["dataset"]["name"]
    root_path = config["dataset"]["root_path"]
    crop_size = config["dataset"].get("input_size", 512)

    full_root = os.path.join(root_path, dataset_name, args.split)

    image_dir = os.path.join(full_root, "images")
    mask_dir = os.path.join(full_root, "masks")
    fov_dir = os.path.join(full_root, "fovs")

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"找不到 image_dir: {image_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"找不到 mask_dir: {mask_dir}")
    if not os.path.isdir(fov_dir):
        raise FileNotFoundError(f"找不到 fov_dir: {fov_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(config["model"], img_size=crop_size).to(device)

    checkpoint = torch.load(args.weight, map_location=device, weights_only=False)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    if hasattr(model, "return_aux"):
        model.return_aux = False

    model.eval()

    image_names = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(IMAGE_EXTS)
    ])

    metric = FOVMetric()

    save_dir = os.path.dirname(args.weight)
    pred_dir = os.path.join(save_dir, f"pred_{args.split}_thr{args.threshold}")
    if args.save_pred:
        os.makedirs(pred_dir, exist_ok=True)

    print("=" * 80)
    print("DRIVE full-image FOV evaluation")
    print(f"dataset: {dataset_name}")
    print(f"split: {args.split}")
    print(f"crop_size: {crop_size}")
    print(f"threshold: {args.threshold}")
    print(f"weight: {args.weight}")
    print("=" * 80)

    for name in tqdm(image_names, desc=f"Evaluating {args.split}"):
        image_path = os.path.join(image_dir, name)
        mask_path = os.path.join(mask_dir, name)
        fov_path = os.path.join(fov_dir, name)

        image = read_rgb(image_path)
        gt = read_binary(mask_path)
        fov = read_binary(fov_path)

        image[fov == 0] = 0

        prob = predict_full_image(
            model=model,
            image=image,
            crop_size=crop_size,
            device=device,
        )

        pred_bin = (prob >= args.threshold).astype(np.uint8)

        metric.update(pred_bin, gt, fov)

        if args.save_pred:
            cv2.imwrite(os.path.join(pred_dir, name), (pred_bin * 255).astype(np.uint8))

    precision, recall, f1, iou = metric.get()

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    result_msg = (
        f"\n==================================================\n"
        f" ⏱测试时间: {now_str}\n"
        f" 权重路径: {args.weight}\n"
        f" 数据集:   {dataset_name}\n"
        f" Split:    {args.split}\n"
        f" 模型:     {config['model']['name']}\n"
        f" Crop:     {crop_size} full-image sliding inference\n"
        f" Threshold:{args.threshold}\n"
        f"--------------------------------------------------\n"
        f" Precision: {precision * 100:.2f}%\n"
        f" Recall:    {recall * 100:.2f}%\n"
        f" F1-Score:  {f1 * 100:.2f}%\n"
        f" IoU:       {iou * 100:.2f}%\n"
        f"==================================================\n"
    )

    print(result_msg)

    result_file = os.path.join(save_dir, f"drive_full_{args.split}_results.log")
    with open(result_file, "a", encoding="utf-8") as f:
        f.write(result_msg)

    print(f"结果已保存到: {result_file}")


if __name__ == "__main__":
    main()