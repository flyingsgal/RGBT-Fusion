#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只导出问题样本的四宫格对比脚本（HBB）   两种模态有各自的标签不使用这一版
------------------------------------------------
功能：
1) 读取 RGB / IR 测试图片、GT 标签
2) 支持两种预测来源：
   - 直接读取已有预测 txt（推荐）
   - 或加载权重分别对 RGB / IR 目录推理
3) 生成四宫格图：
   - 左上：RGB GT
   - 右上：RGB GT + Pred
   - 左下：IR  GT
   - 右下：IR  GT + Pred
4) 只导出“问题样本”：
   - RGB/IR 任一侧 GT 与 Pred 数量差超过阈值
   - RGB/IR 两侧 GT 数量差超过阈值
   - RGB/IR 两侧 Pred 数量差超过阈值
   - 指定类别在任一模态漏检
   - 也可额外按低置信度预测进行筛查
5) 输出问题清单 CSV，方便后续定量分析

标签格式（HBB, YOLO）：
GT:   cls cx cy w h
Pred: cls cx cy w h conf
"""

import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 配置区：改这里就行
# =========================
USE_EXISTING_PRED = True

# 只有 USE_EXISTING_PRED=False 时才会用到
WEIGHTS = "/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/train/ACMp5/weights/best.pt"

RGB_IMG_DIR = "/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/images/test"
IR_IMG_DIR  = "/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/image/test"

RGB_GT_DIR = "/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/labelrgb/test"
IR_GT_DIR  = "/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/labels/test"

SAVE_ROOT = "/storage/jyx4/projects/TwoStream_Yolov8-main/runs/val/exp/images_compare"
SAVE_VIS_DIR = os.path.join(SAVE_ROOT, "vis")
SAVE_CSV = os.path.join(SAVE_ROOT, "problem_summary.csv")

# 若读取已有预测，双流模型导出的同一份 labels 可同时填给 RGB / IR
RGB_PRED_SOURCE_DIR = "/storage/jyx4/projects/TwoStream_Yolov8-main/runs/val/exp/labels"
IR_PRED_SOURCE_DIR  = "/storage/jyx4/projects/TwoStream_Yolov8-main/runs/val/exp/labels"

# 若重新推理，预测 txt 保存到这里
RGB_PRED_DIR = os.path.join(SAVE_ROOT, "pred_rgb")
IR_PRED_DIR  = os.path.join(SAVE_ROOT, "pred_ir")

# 推理参数（仅 USE_EXISTING_PRED=False 时使用）
IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = 0
BATCH = 16

# 类别名字（按你的数据集改；若不改也不影响运行）
CLASS_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "van",
    4: "freight_car",
}

# 问题样本筛选策略
GT_PRED_DIFF_THR = 1         # 单模态下 |GT数-Pred数| >= 该值 就导出
CROSS_MODAL_GT_DIFF_THR = 1  # |RGB_GT-IR_GT| >= 该值 就导出
CROSS_MODAL_PD_DIFF_THR = 1  # |RGB_Pred-IR_Pred| >= 该值 就导出
LOW_CONF_MEAN_THR = 0.35     # 平均预测置信度低于此值可视为问题（仅对有预测的样本生效）

# 指定重点漏检类别。写类别id，如 [3, 4] 代表 van、freight_car
FOCUS_MISS_CLASSES = [3, 4]
ENABLE_FOCUS_CLASS_MISS = True
EXPORT_IF_MISSING_ANY_MODAL = True

IMG_SUFFIXES = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
TITLE_FS = 12
TEXT_FS = 10


# =========================
# 工具函数
# =========================
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def list_images(img_dir: str) -> Dict[str, str]:
    d = {}
    for p in Path(img_dir).iterdir():
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES:
            d[p.stem] = str(p)
    return d


def read_gt_yolo_hbb(txt_path: str, img_w: int, img_h: int) -> List[Dict]:
    """GT: cls cx cy w h -> 绝对坐标 xyxy"""
    items = []
    if not os.path.exists(txt_path):
        return items

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                cx = float(parts[1]) * img_w
                cy = float(parts[2]) * img_h
                bw = float(parts[3]) * img_w
                bh = float(parts[4]) * img_h
            except Exception:
                continue

            x1 = max(0, cx - bw / 2)
            y1 = max(0, cy - bh / 2)
            x2 = min(img_w - 1, cx + bw / 2)
            y2 = min(img_h - 1, cy + bh / 2)
            items.append({"cls": cls_id, "conf": None, "xyxy": [x1, y1, x2, y2]})
    return items


def read_pred_yolo_hbb(txt_path: str, img_w: int, img_h: int) -> List[Dict]:
    """Pred: cls cx cy w h conf -> 绝对坐标 xyxy"""
    items = []
    if not os.path.exists(txt_path):
        return items
    VIS_CONF_THR = 0.25  # 可视化时的置信度下限，低于这个值的预测框不画出来，但仍然会计数和参与问题样本筛选
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                cls_id = int(float(parts[0]))
                cx = float(parts[1]) * img_w
                cy = float(parts[2]) * img_h
                bw = float(parts[3]) * img_w
                bh = float(parts[4]) * img_h
                conf = float(parts[5])
            except Exception:
                continue
            
            if conf < VIS_CONF_THR:
                continue

            x1 = max(0, cx - bw / 2)
            y1 = max(0, cy - bh / 2)
            x2 = min(img_w - 1, cx + bw / 2)
            y2 = min(img_h - 1, cy + bh / 2)
            items.append({"cls": cls_id, "conf": conf, "xyxy": [x1, y1, x2, y2]})
    return items


def write_pred_yolo_hbb(txt_path: str, preds: List[Dict], img_w: int, img_h: int):
    """Pred: cls cx cy w h conf"""
    with open(txt_path, "w", encoding="utf-8") as f:
        for p in preds:
            x1, y1, x2, y2 = p["xyxy"]
            cls_id = int(p["cls"])
            conf = float(p["conf"]) if p["conf"] is not None else 0.0
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            bw = (x2 - x1) / img_w
            bh = (y2 - y1) / img_h
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")


def run_predict_on_images(model, img_paths: List[str], save_txt_dir: str) -> Dict[str, List[Dict]]:
    """对单目录图片推理并写出 txt；仅 USE_EXISTING_PRED=False 时使用。"""
    ensure_dir(save_txt_dir)
    results = model.predict(
        source=img_paths,
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=DEVICE,
        batch=BATCH,
        verbose=False,
        save=False,
        stream=False,
    )

    out = {}
    for r in results:
        path = r.path
        stem = Path(path).stem
        img = cv2.imread(path)
        if img is None:
            out[stem] = []
            continue

        h, w = img.shape[:2]
        preds = []
        if getattr(r, "boxes", None) is not None and r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_ = r.boxes.cls.cpu().numpy()
            conf_ = r.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i].tolist()
                preds.append({
                    "cls": int(cls_[i]),
                    "conf": float(conf_[i]),
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                })
        out[stem] = preds
        write_pred_yolo_hbb(os.path.join(save_txt_dir, f"{stem}.txt"), preds, w, h)
    return out


def draw_boxes(img_bgr: np.ndarray, boxes: List[Dict], color=(0, 255, 0), show_conf=False) -> np.ndarray:
    out = img_bgr.copy()
    for item in boxes:
        x1, y1, x2, y2 = map(int, item["xyxy"])
        cls_id = int(item["cls"])
        conf = item["conf"]
        name = CLASS_NAMES.get(cls_id, str(cls_id))
        label = name if (conf is None or not show_conf) else f"{name} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        ((tw, th), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_text = max(0, y1 - th - 4)
        cv2.rectangle(out, (x1, y_text), (x1 + tw + 4, y_text + th + 4), color, -1)
        cv2.putText(out, label, (x1 + 2, y_text + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def cls_hist(items: List[Dict]) -> Dict[int, int]:
    hist = {}
    for it in items:
        c = int(it["cls"])
        hist[c] = hist.get(c, 0) + 1
    return hist


def mean_conf(items: List[Dict]) -> Optional[float]:
    vals = [float(x["conf"]) for x in items if x["conf"] is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def check_focus_miss(gt_items: List[Dict], pred_items: List[Dict], focus_classes: List[int]) -> List[int]:
    gt_hist = cls_hist(gt_items)
    pd_hist = cls_hist(pred_items)
    misses = []
    for c in focus_classes:
        if gt_hist.get(c, 0) > 0 and pd_hist.get(c, 0) == 0:
            misses.append(c)
    return misses


def get_reasons(rgb_gt, rgb_pred, ir_gt, ir_pred, rgb_missing=False, ir_missing=False) -> List[str]:
    reasons = []

    if EXPORT_IF_MISSING_ANY_MODAL and (rgb_missing or ir_missing):
        if rgb_missing:
            reasons.append("missing_rgb_image")
        if ir_missing:
            reasons.append("missing_ir_image")

    if abs(len(rgb_gt) - len(rgb_pred)) >= GT_PRED_DIFF_THR:
        reasons.append(f"rgb_gt_pred_diff>={GT_PRED_DIFF_THR}")
    if abs(len(ir_gt) - len(ir_pred)) >= GT_PRED_DIFF_THR:
        reasons.append(f"ir_gt_pred_diff>={GT_PRED_DIFF_THR}")

    if abs(len(rgb_gt) - len(ir_gt)) >= CROSS_MODAL_GT_DIFF_THR:
        reasons.append(f"cross_modal_gt_diff>={CROSS_MODAL_GT_DIFF_THR}")
    if abs(len(rgb_pred) - len(ir_pred)) >= CROSS_MODAL_PD_DIFF_THR:
        reasons.append(f"cross_modal_pred_diff>={CROSS_MODAL_PD_DIFF_THR}")

    rgb_mc = mean_conf(rgb_pred)
    ir_mc = mean_conf(ir_pred)
    if rgb_mc is not None and rgb_mc < LOW_CONF_MEAN_THR:
        reasons.append(f"rgb_low_conf_mean<{LOW_CONF_MEAN_THR}")
    if ir_mc is not None and ir_mc < LOW_CONF_MEAN_THR:
        reasons.append(f"ir_low_conf_mean<{LOW_CONF_MEAN_THR}")

    if ENABLE_FOCUS_CLASS_MISS and FOCUS_MISS_CLASSES:
        rgb_miss = check_focus_miss(rgb_gt, rgb_pred, FOCUS_MISS_CLASSES)
        ir_miss = check_focus_miss(ir_gt, ir_pred, FOCUS_MISS_CLASSES)
        if rgb_miss:
            names = ",".join(CLASS_NAMES.get(c, str(c)) for c in rgb_miss)
            reasons.append(f"rgb_focus_class_miss:{names}")
        if ir_miss:
            names = ",".join(CLASS_NAMES.get(c, str(c)) for c in ir_miss)
            reasons.append(f"ir_focus_class_miss:{names}")

    return reasons


def safe_read_image(path: Optional[str]) -> Optional[np.ndarray]:
    if path is None or not os.path.exists(path):
        return None
    return cv2.imread(path)


def blank_panel(text="MISSING", size=(640, 640, 3)) -> np.ndarray:
    panel = np.full(size, 240, dtype=np.uint8)
    cv2.putText(panel, text, (40, size[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (20, 20, 20), 2, cv2.LINE_AA)
    return panel


def make_same_size(imgs: List[np.ndarray]) -> List[np.ndarray]:
    hs = [x.shape[0] for x in imgs]
    ws = [x.shape[1] for x in imgs]
    th, tw = max(hs), max(ws)
    outs = []
    for im in imgs:
        h, w = im.shape[:2]
        canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
        y0 = (th - h) // 2
        x0 = (tw - w) // 2
        canvas[y0:y0 + h, x0:x0 + w] = im
        outs.append(canvas)
    return outs


def save_quad_figure(
    save_path: str,
    stem: str,
    rgb_gt_img: np.ndarray,
    rgb_mix_img: np.ndarray,
    ir_gt_img: np.ndarray,
    ir_mix_img: np.ndarray,
    rgb_gt_n: int,
    rgb_pred_n: int,
    ir_gt_n: int,
    ir_pred_n: int,
    reasons: List[str],
    rgb_conf_mean: Optional[float],
    ir_conf_mean: Optional[float],
):
    panels = make_same_size([rgb_gt_img, rgb_mix_img, ir_gt_img, ir_mix_img])
    rgb_gt_img, rgb_mix_img, ir_gt_img, ir_mix_img = panels

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax = axes.ravel()

    ax[0].imshow(bgr_to_rgb(rgb_gt_img))
    ax[0].set_title(f"RGB GT (n={rgb_gt_n})", fontsize=TITLE_FS)
    ax[0].axis("off")

    ax[1].imshow(bgr_to_rgb(rgb_mix_img))
    conf_txt = "NA" if rgb_conf_mean is None else f"{rgb_conf_mean:.3f}"
    ax[1].set_title(f"RGB GT + Pred (GT={rgb_gt_n}, Pred={rgb_pred_n}, mean_conf={conf_txt})", fontsize=TITLE_FS)
    ax[1].axis("off")

    ax[2].imshow(bgr_to_rgb(ir_gt_img))
    ax[2].set_title(f"IR GT (n={ir_gt_n})", fontsize=TITLE_FS)
    ax[2].axis("off")

    ax[3].imshow(bgr_to_rgb(ir_mix_img))
    conf_txt = "NA" if ir_conf_mean is None else f"{ir_conf_mean:.3f}"
    ax[3].set_title(f"IR GT + Pred (GT={ir_gt_n}, Pred={ir_pred_n}, mean_conf={conf_txt})", fontsize=TITLE_FS)
    ax[3].axis("off")

    reason_text = " | ".join(reasons) if reasons else "none"
    fig.suptitle(f"{stem}\nProblem reasons: {reason_text}", fontsize=13)
    fig.text(0.5, 0.02, "Green = GT, Red = Prediction", ha="center", fontsize=TEXT_FS)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =========================
# 主流程
# =========================
def main():
    ensure_dir(SAVE_ROOT)
    ensure_dir(SAVE_VIS_DIR)

    rgb_imgs = list_images(RGB_IMG_DIR)
    ir_imgs = list_images(IR_IMG_DIR)

    all_stems = sorted(set(rgb_imgs.keys()) | set(ir_imgs.keys()))
    print(f"[INFO] RGB images: {len(rgb_imgs)}")
    print(f"[INFO] IR  images: {len(ir_imgs)}")
    print(f"[INFO] Pair stems : {len(all_stems)}")
    print(f"[INFO] USE_EXISTING_PRED = {USE_EXISTING_PRED}")

    rgb_preds = {}
    ir_preds = {}

    if not USE_EXISTING_PRED:
        from ultralytics import YOLO

        ensure_dir(RGB_PRED_DIR)
        ensure_dir(IR_PRED_DIR)

        print(f"[INFO] Loading weights: {WEIGHTS}")
        model = YOLO(WEIGHTS)

        rgb_img_paths = [rgb_imgs[s] for s in sorted(rgb_imgs.keys())]
        ir_img_paths = [ir_imgs[s] for s in sorted(ir_imgs.keys())]

        print("[INFO] Running predictions for RGB...")
        rgb_preds = run_predict_on_images(model, rgb_img_paths, RGB_PRED_DIR) if rgb_img_paths else {}

        print("[INFO] Running predictions for IR...")
        ir_preds = run_predict_on_images(model, ir_img_paths, IR_PRED_DIR) if ir_img_paths else {}
    else:
        print(f"[INFO] Read RGB pred txt from: {RGB_PRED_SOURCE_DIR}")
        print(f"[INFO] Read IR  pred txt from: {IR_PRED_SOURCE_DIR}")

    rows = []
    exported = 0

    for stem in all_stems:
        rgb_path = rgb_imgs.get(stem)
        ir_path = ir_imgs.get(stem)

        rgb_missing = rgb_path is None
        ir_missing = ir_path is None

        rgb_img = safe_read_image(rgb_path)
        ir_img = safe_read_image(ir_path)

        rgb_gt, ir_gt, rgb_pd, ir_pd = [], [], [], []

        if rgb_img is not None:
            h, w = rgb_img.shape[:2]
            rgb_gt = read_gt_yolo_hbb(os.path.join(RGB_GT_DIR, f"{stem}.txt"), w, h)
            rgb_pd = read_pred_yolo_hbb(os.path.join(RGB_PRED_SOURCE_DIR, f"{stem}.txt"), w, h) if USE_EXISTING_PRED else rgb_preds.get(stem, [])
        elif not USE_EXISTING_PRED:
            rgb_pd = rgb_preds.get(stem, [])

        if ir_img is not None:
            h, w = ir_img.shape[:2]
            ir_gt = read_gt_yolo_hbb(os.path.join(IR_GT_DIR, f"{stem}.txt"), w, h)
            ir_pd = read_pred_yolo_hbb(os.path.join(IR_PRED_SOURCE_DIR, f"{stem}.txt"), w, h) if USE_EXISTING_PRED else ir_preds.get(stem, [])
        elif not USE_EXISTING_PRED:
            ir_pd = ir_preds.get(stem, [])

        reasons = get_reasons(rgb_gt, rgb_pd, ir_gt, ir_pd, rgb_missing=rgb_missing, ir_missing=ir_missing)
        rgb_mc = mean_conf(rgb_pd)
        ir_mc = mean_conf(ir_pd)

        rows.append({
            "stem": stem,
            "rgb_img_exists": not rgb_missing,
            "ir_img_exists": not ir_missing,
            "rgb_gt_num": len(rgb_gt),
            "rgb_pred_num": len(rgb_pd),
            "ir_gt_num": len(ir_gt),
            "ir_pred_num": len(ir_pd),
            "rgb_mean_conf": None if rgb_mc is None else round(rgb_mc, 6),
            "ir_mean_conf": None if ir_mc is None else round(ir_mc, 6),
            "reasons": " | ".join(reasons),
        })

        if not reasons:
            continue

        if rgb_img is None:
            rgb_gt_img = blank_panel("RGB IMAGE MISSING")
            rgb_mix_img = blank_panel("RGB IMAGE MISSING")
        else:
            rgb_gt_img = draw_boxes(rgb_img, rgb_gt, color=(0, 255, 0), show_conf=False)
            rgb_mix_img = draw_boxes(rgb_gt_img, rgb_pd, color=(0, 0, 255), show_conf=True)

        if ir_img is None:
            ir_gt_img = blank_panel("IR IMAGE MISSING")
            ir_mix_img = blank_panel("IR IMAGE MISSING")
        else:
            ir_gt_img = draw_boxes(ir_img, ir_gt, color=(0, 255, 0), show_conf=False)
            ir_mix_img = draw_boxes(ir_gt_img, ir_pd, color=(0, 0, 255), show_conf=True)

        save_path = os.path.join(SAVE_VIS_DIR, f"{stem}.jpg")
        save_quad_figure(
            save_path=save_path,
            stem=stem,
            rgb_gt_img=rgb_gt_img,
            rgb_mix_img=rgb_mix_img,
            ir_gt_img=ir_gt_img,
            ir_mix_img=ir_mix_img,
            rgb_gt_n=len(rgb_gt),
            rgb_pred_n=len(rgb_pd),
            ir_gt_n=len(ir_gt),
            ir_pred_n=len(ir_pd),
            reasons=reasons,
            rgb_conf_mean=rgb_mc,
            ir_conf_mean=ir_mc,
        )
        exported += 1
        if exported % 50 == 0:
            print(f"[INFO] Exported {exported} problem samples...")

    df = pd.DataFrame(rows)
    df.to_csv(SAVE_CSV, index=False, encoding="utf-8-sig")

    print("\n[DONE] Finished.")
    print(f"[DONE] Problem vis dir      : {SAVE_VIS_DIR}")
    print(f"[DONE] Problem summary csv : {SAVE_CSV}")
    if USE_EXISTING_PRED:
        print(f"[DONE] RGB pred txt dir     : {RGB_PRED_SOURCE_DIR}")
        print(f"[DONE] IR  pred txt dir     : {IR_PRED_SOURCE_DIR}")
    else:
        print(f"[DONE] RGB pred txt dir     : {RGB_PRED_DIR}")
        print(f"[DONE] IR  pred txt dir     : {IR_PRED_DIR}")
        print(f"[DONE] Weights              : {WEIGHTS}")


if __name__ == "__main__":
    main()
