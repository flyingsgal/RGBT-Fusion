#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path
from typing import List

import cv2
from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO predictions to txt files")
    parser.add_argument("--weights", type=str, required=True, 
                        default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/train/LowLightLoss_refine2ctx_p5only/weights/best.pt", 
                        help="模型权重路径，如 best.pt")
    parser.add_argument("--source", type=str, required=True, default= "", help="图片目录")
    parser.add_argument("--save-dir", type=str, required=True, help="预测txt保存目录")
    parser.add_argument("--imgsz", type=int, default=640, help="推理尺寸")
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU阈值")
    parser.add_argument("--device", type=str, default="0", help="设备，如 0 / 0,1 / cpu")
    parser.add_argument("--max-det", type=int, default=300, help="每张图最大检测框数")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="可选：只保留这些类别")
    parser.add_argument("--suffix", type=str, default=".txt", help="输出标签后缀")
    return parser.parse_args()


def list_images(source_dir: str) -> List[Path]:
    source = Path(source_dir)
    if not source.exists():
        raise FileNotFoundError(f"source 不存在: {source_dir}")

    files = []
    for p in source.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    files.sort()
    return files


def xyxy_to_yolo(x1, y1, x2, y2, w, h):
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    return (
        cx / w,
        cy / h,
        bw / w,
        bh / h,
    )


def write_one_txt(txt_path: Path, result):
    """
    写出格式:
    cls cx cy w h conf
    """
    img = result.orig_img
    if img is None:
        txt_path.write_text("", encoding="utf-8")
        return

    h, w = img.shape[:2]
    boxes = result.boxes

    lines = []
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i]
            c = int(cls[i])
            s = float(conf[i])

            cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            line = f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {s:.6f}"
            lines.append(line)

    with open(txt_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines))


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    image_paths = list_images(args.source)
    print(f"[INFO] Found images: {len(image_paths)}")

    if len(image_paths) == 0:
        print("[WARN] 没有找到图片，退出")
        return

    print(f"[INFO] Loading model from: {args.weights}")
    model = YOLO(args.weights)

    results = model.predict(
        source=[str(p) for p in image_paths],
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_det=args.max_det,
        classes=args.classes,
        verbose=False,
        stream=False,
        save=False,
    )

    count = 0
    for img_path, result in zip(image_paths, results):
        stem = img_path.stem
        txt_path = Path(args.save_dir) / f"{stem}{args.suffix}"
        write_one_txt(txt_path, result)
        count += 1
        if count % 200 == 0:
            print(f"[INFO] Exported {count}/{len(image_paths)}")

    print(f"[INFO] Done. Txt files saved to: {args.save_dir}")


if __name__ == "__main__":
    main()