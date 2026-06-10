# visualize_tsci_debug.py
# 用途：
# 1. 加载训练好的 best.pt / last.pt
# 2. 输入一对 RGB / IR 图像，拼成 6 通道
# 3. 前向一次模型
# 4. 读取 TSCIv2RawCueFusion.debug_info
# 5. 保存窗口分数、top-k mask、残差响应、融合响应等 heatmap

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 确保优先使用当前项目中的 ultralytics，而不是 conda/site-packages 里的官方版本
import sys
PROJECT_ROOT = os.environ.get("TWOSTREAM_YOLO_ROOT", "/storage/jyx4/projects/TwoStream_Yolov8-main")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ultralytics
from ultralytics import YOLO


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_rgb_image(path, imgsz):
    """
    读取 RGB 图像，输出 torch tensor: [3, imgsz, imgsz], range 0~1
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return img


def read_ir_image(path, imgsz):
    """
    读取 IR 图像，输出 torch tensor: [3, imgsz, imgsz], range 0~1

    如果 IR 是灰度图，则复制成 3 通道。
    如果 IR 是三通道图，则直接作为三通道输入。
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read IR image: {path}")

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
    return img


def to_uint8_img(x):
    """
    x: torch tensor [3,H,W] or [1,H,W], range 0~1
    return: uint8 HWC
    """
    x = x.detach().cpu().float()

    if x.dim() == 3:
        x = x.numpy()
        x = np.transpose(x, (1, 2, 0))
    elif x.dim() == 2:
        x = x.numpy()
    else:
        raise ValueError(f"Unsupported image shape: {x.shape}")

    x = np.clip(x, 0.0, 1.0)

    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)

    return (x * 255).astype(np.uint8)


def norm01_np(x, eps=1e-6):
    x = x.astype(np.float32)
    mn = x.min()
    mx = x.max()
    return (x - mn) / (mx - mn + eps)


def save_heatmap_only(hmap, save_path, cmap="jet"):
    """
    hmap: torch tensor [1,H,W] or [H,W]
    """
    h = hmap.detach().cpu().float().squeeze().numpy()
    h = norm01_np(h)

    plt.figure(figsize=(5, 5))
    plt.imshow(h, cmap=cmap)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overlay(img_rgb, hmap, save_path, alpha=0.45, boxes=None):
    """
    img_rgb: torch tensor [3,H,W], range 0~1
    hmap: torch tensor [1,h,w] or [h,w]
    boxes:
        None
        or list of dict:
            {"type": "hbb", "box": [x1,y1,x2,y2]}
            {"type": "obb", "pts": [[x1,y1],...,[x4,y4]]}
    """
    img = to_uint8_img(img_rgb)
    H, W = img.shape[:2]

    h = hmap.detach().cpu().float().squeeze().numpy()
    h = norm01_np(h)
    h = cv2.resize(h, (W, H), interpolation=cv2.INTER_LINEAR)
    h = (h * 255).astype(np.uint8)

    h_color = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    h_color = cv2.cvtColor(h_color, cv2.COLOR_BGR2RGB)

    overlay = (1 - alpha) * img.astype(np.float32) + alpha * h_color.astype(np.float32)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if boxes is not None:
        overlay = draw_boxes_rgb(overlay, boxes)

    cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def class_color(cls_id, default=(0, 255, 0)):
    """返回 RGB 颜色。这里只用于可视化，不影响模型。"""
    palette = [
        (255, 0, 0),      # red
        (255, 160, 0),    # orange
        (0, 200, 255),    # cyan
        (255, 0, 255),    # magenta
        (220, 220, 0),    # yellow
        (0, 255, 0),      # green
    ]
    if cls_id is None:
        return default
    return palette[int(cls_id) % len(palette)]


def draw_label_rgb(out, text, x, y, color, font_scale=0.5, thickness=1):
    """在 RGB 图像上画类别文字。"""
    if text is None or len(text) == 0:
        return out

    H, W = out.shape[:2]
    x = int(max(0, min(W - 1, x)))
    y = int(max(0, min(H - 1, y)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 尽量让文字框不出界
    y_text = y - 4
    if y_text - th - baseline < 0:
        y_text = y + th + baseline + 4
    x2 = min(W - 1, x + tw + 4)
    y1 = max(0, y_text - th - baseline - 2)
    y2 = min(H - 1, y_text + baseline + 2)

    cv2.rectangle(out, (x, y1), (x2, y2), color, -1)
    cv2.putText(out, text, (x + 2, y_text), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def draw_boxes_rgb(img_rgb, boxes, draw_label=True, default_color=(0, 255, 0), thickness=2):
    """
    在 RGB 图像上画 HBB 或 OBB，并可标注类别/置信度。

    boxes item 支持：
      HBB: {"type":"hbb", "box":[x1,y1,x2,y2], "cls":0, "name":"car", "conf":0.8}
      OBB: {"type":"obb", "pts":[[x1,y1],...], "cls":0, "name":"car", "conf":0.8}
    """
    out = img_rgb.copy()

    for item in boxes:
        cls_id = item.get("cls", None)
        color = item.get("color", class_color(cls_id, default=default_color))

        name = item.get("name", None)
        conf = item.get("conf", None)
        if name is None and cls_id is not None:
            name = str(cls_id)
        if conf is not None and name is not None:
            text = f"{name} {float(conf):.2f}"
        else:
            text = name

        if item["type"] == "hbb":
            x1, y1, x2, y2 = map(int, item["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            if draw_label:
                out = draw_label_rgb(out, text, x1, y1, color)

        elif item["type"] == "obb":
            pts = np.array(item["pts"], dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
            if draw_label:
                x, y = pts[:, 0].min(), pts[:, 1].min()
                out = draw_label_rgb(out, text, x, y, color)

    return out


def parse_names_from_string(names_str):
    """--class-names 'car,truck,bus,van,freight_car' -> {0:'car',...}"""
    if names_str is None or str(names_str).strip() == "":
        return None
    names = [x.strip() for x in names_str.split(",") if x.strip()]
    return {i: n for i, n in enumerate(names)}


def get_model_names(model):
    """优先读取权重中的 model.names；没有则使用 DroneVehicle 默认类别名。"""
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)

    if isinstance(names, dict) and len(names) > 0:
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)) and len(names) > 0:
        return {i: str(v) for i, v in enumerate(names)}

    # DroneVehicle 常用顺序；如果你的 data.yaml 不是这个顺序，请用 --class-names 覆盖
    return {0: "car", 1: "truck", 2: "bus", 3: "van", 4: "freight_car"}


def load_yolo_label(label_path, imgsz, names=None):
    """
    支持两种 GT label 格式：
      HBB: cls cx cy w h
      OBB: cls x1 y1 x2 y2 x3 y3 x4 y4

    坐标默认是归一化坐标。
    返回 boxes 时会保留 cls 和 name，便于绘制类别文字。
    """
    boxes = []

    if label_path is None:
        return boxes

    label_path = Path(label_path)
    if not label_path.exists():
        print(f"[WARN] label file not found: {label_path}")
        return boxes

    names = names or {}

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0:
            continue

        vals = [float(x) for x in parts]
        cls_id = int(vals[0])
        cls_name = names.get(cls_id, str(cls_id))

        if len(vals) == 5:
            _, cx, cy, w, h = vals
            x1 = (cx - w / 2) * imgsz
            y1 = (cy - h / 2) * imgsz
            x2 = (cx + w / 2) * imgsz
            y2 = (cy + h / 2) * imgsz
            boxes.append({
                "type": "hbb",
                "box": [x1, y1, x2, y2],
                "cls": cls_id,
                "name": cls_name,
            })

        elif len(vals) == 9:
            coords = vals[1:]
            pts = []
            for i in range(0, 8, 2):
                x = coords[i] * imgsz
                y = coords[i + 1] * imgsz
                pts.append([x, y])
            boxes.append({
                "type": "obb",
                "pts": pts,
                "cls": cls_id,
                "name": cls_name,
            })

        else:
            print(f"[WARN] unsupported label line with {len(vals)} values: {line.strip()}")

    return boxes


def get_prediction_boxes(model, x, imgsz, device, names, conf=0.25, iou=0.7):
    """
    对 6 通道输入做一次预测，返回可用于 draw_boxes_rgb 的 boxes。
    支持 HBB boxes 和 OBB obb 输出。
    """
    boxes = []

    # Ultralytics 支持 torch Tensor source: [B,C,H,W]，这里 C=6
    results = model.predict(
        source=x,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=str(device),
        verbose=False,
    )

    r = results[0]

    # OBB 优先
    obb = getattr(r, "obb", None)
    if obb is not None and len(obb) > 0:
        pts_all = obb.xyxyxyxy.detach().cpu().numpy()  # [N,4,2]
        cls_all = obb.cls.detach().cpu().numpy().astype(int)
        conf_all = obb.conf.detach().cpu().numpy()

        for pts, cls_id, cf in zip(pts_all, cls_all, conf_all):
            boxes.append({
                "type": "obb",
                "pts": pts.tolist(),
                "cls": int(cls_id),
                "name": names.get(int(cls_id), str(int(cls_id))),
                "conf": float(cf),
            })
        return boxes

    # HBB
    b = getattr(r, "boxes", None)
    if b is not None and len(b) > 0:
        xyxy_all = b.xyxy.detach().cpu().numpy()
        cls_all = b.cls.detach().cpu().numpy().astype(int)
        conf_all = b.conf.detach().cpu().numpy()

        for xyxy, cls_id, cf in zip(xyxy_all, cls_all, conf_all):
            boxes.append({
                "type": "hbb",
                "box": xyxy.tolist(),
                "cls": int(cls_id),
                "name": names.get(int(cls_id), str(int(cls_id))),
                "conf": float(cf),
            })

    return boxes


def save_labeled_rgb(x, out_dir, boxes, filename, default_color=(0, 255, 0)):
    """保存一张带框和类别文字的 RGB 图。"""
    rgb = x[0, :3].detach().cpu()
    rgb_np = to_uint8_img(rgb)
    rgb_np = draw_boxes_rgb(rgb_np, boxes, draw_label=True, default_color=default_color)
    cv2.imwrite(str(Path(out_dir) / filename), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))


def save_input_images(x, out_dir, boxes=None):
    """
    x: [1,6,H,W]
    保存原始 RGB/IR；如果提供 GT boxes，额外保存 gt_rgb_with_labels.png。
    """
    img = x[0].detach().cpu()
    rgb = img[:3]
    ir = img[3:6]

    rgb_np = to_uint8_img(rgb)
    ir_np = to_uint8_img(ir)

    cv2.imwrite(str(Path(out_dir) / "rgb.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(Path(out_dir) / "ir.png"), cv2.cvtColor(ir_np, cv2.COLOR_RGB2BGR))

    if boxes is not None:
        gt_rgb = draw_boxes_rgb(rgb_np, boxes, draw_label=True, default_color=(0, 255, 0))
        gt_ir = draw_boxes_rgb(ir_np, boxes, draw_label=True, default_color=(0, 255, 0))
        cv2.imwrite(str(Path(out_dir) / "gt_rgb_with_labels.png"), cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(Path(out_dir) / "gt_ir_with_labels.png"), cv2.cvtColor(gt_ir, cv2.COLOR_RGB2BGR))


def find_tsci_modules(net):
    """
    不强依赖 import 路径，按类名查找 TSCI 调试模块。
    兼容：
      - TSCIv2RawCueFusion
      - TSCIv3RawCueSS2DFusion
    只要模块 forward 后写入 debug_info，本脚本都会把里面的 heatmap 全部保存。
    """
    target_names = {
        "TSCIv2RawCueFusion",
        "TSCIv3RawCueSS2DFusion",
    }

    modules = []
    for m in net.modules():
        if m.__class__.__name__ in target_names:
            modules.append(m)
    return modules


def run_forward(net, x):
    net.eval()
    with torch.no_grad():
        _ = net(x)


def save_debug_maps(tsci_modules, x, out_dir, boxes=None, overlay_alpha=0.45):
    """
    保存每个 TSCIv2RawCueFusion 模块的 debug heatmap。
    """
    rgb0 = x[0, :3].detach().cpu()

    for idx, m in enumerate(tsci_modules):
        module_dir = Path(out_dir) / f"tsci_{idx}"
        ensure_dir(module_dir)

        info = getattr(m, "debug_info", None)
        if not info:
            print(f"[WARN] module tsci_{idx} has empty debug_info. "
                  f"Check whether enable_debug=True and forward has run.")
            continue

        print(f"[INFO] saving debug maps for module {idx}: {module_dir}")

        for key, value in info.items():
            if value is None:
                continue

            # value 一般是 [B,1,H,W]
            if not torch.is_tensor(value):
                print(f"[WARN] debug key {key} is not tensor, skip.")
                continue

            if value.dim() == 4:
                heat = value[0]
            elif value.dim() == 3:
                heat = value
            else:
                print(f"[WARN] debug key {key} has unsupported shape {value.shape}, skip.")
                continue

            save_heatmap_only(heat, module_dir / f"{key}.png")
            save_overlay(
                rgb0,
                heat,
                module_dir / f"{key}_on_rgb.png",
                alpha=overlay_alpha,
                boxes=boxes,
            )


def build_input_from_pair(rgb_path, ir_path, imgsz, device):
    rgb = read_rgb_image(rgb_path, imgsz)
    ir = read_ir_image(ir_path, imgsz)
    x = torch.cat([rgb, ir], dim=0).unsqueeze(0)
    x = x.to(device).float()
    return x

#353
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/mamba/noBN_TSCIv3RawCueSS2DFusion2/weights/best.pt", help="训练好的 best.pt 或 last.pt")
    parser.add_argument("--rgb", type=str, default="/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/images/val/00001.jpg", help="RGB 图像路径")
    parser.add_argument("--ir", type=str, default="/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/image/val//00001.jpg", help="IR 图像路径")
    parser.add_argument("--label", type=str, default="/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/labels/val//00001.txt", help="可选：YOLO label txt 路径，用于画 GT 框")
    parser.add_argument("--out", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/mamba/noBN_TSCIv3RawCueSS2DFusion2/001", help="输出目录")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 或 cpu")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)
    parser.add_argument("--conf", type=float, default=0.25, help="预测可视化置信度阈值")
    parser.add_argument("--iou", type=float, default=0.7, help="预测可视化 NMS IoU 阈值")
    parser.add_argument("--class-names", type=str, default="",
                        help="可选：逗号分隔类别名，例如 car,truck,bus,van,freight_car；为空则读取 model.names")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ensure_dir(args.out)

    print(f"[INFO] using ultralytics from: {ultralytics.__file__}")
    print(f"[INFO] loading model: {args.weights}")
    model = YOLO(args.weights)
    net = model.model.to(device).float().eval()

    tsci_modules = find_tsci_modules(net)

    if len(tsci_modules) == 0:
        raise RuntimeError(
            "没有找到 TSCIv2RawCueFusion / TSCIv3RawCueSS2DFusion 模块。请确认：\n"
            "1. 当前权重对应的模型结构里确实用了 TSCIv2 或 TSCIv3 模块；\n"
            "2. 当前脚本导入的是项目内 ultralytics，而不是 site-packages；\n"
            "3. 类名是否已经改名。"
        )

    print(f"[INFO] found {len(tsci_modules)} TSCI debug modules:")
    for i, m in enumerate(tsci_modules):
        print(f"    [{i}] {m.__class__.__name__}")

    # for m in tsci_modules:
    #     if hasattr(m, "enable_debug"):
    #         m.enable_debug = True
    #     else:
    #         raise AttributeError(
    #             "TSCIv2RawCueFusion 模块没有 enable_debug 属性。\n"
    #             "请先在模块 __init__ 中加入：\n"
    #             "self.enable_debug = False\n"
    #             "self.debug_info = {}"
    #         )
    for m in tsci_modules:
        m.enable_debug = True
        if not hasattr(m, "debug_info"):
            m.debug_info = {}

    x = build_input_from_pair(args.rgb, args.ir, args.imgsz, device)

    names = parse_names_from_string(args.class_names) or get_model_names(model)
    print(f"[INFO] class names: {names}")

    boxes = load_yolo_label(args.label, args.imgsz, names=names) if args.label else None

    save_input_images(x, args.out, boxes=boxes)

    print("[INFO] running forward...")
    run_forward(net, x)

    print("[INFO] saving heatmaps...")
    save_debug_maps(
        tsci_modules=tsci_modules,
        x=x,
        out_dir=args.out,
        boxes=boxes,
        overlay_alpha=args.overlay_alpha,
    )

    print("[INFO] running prediction for labeled pred image...")
    pred_boxes = get_prediction_boxes(
        model=model,
        x=x,
        imgsz=args.imgsz,
        device=device,
        names=names,
        conf=args.conf,
        iou=args.iou,
    )
    print(f"[INFO] predicted boxes: {len(pred_boxes)}")
    save_labeled_rgb(
        x=x,
        out_dir=args.out,
        boxes=pred_boxes,
        filename="pred_rgb_with_labels.png",
        default_color=(255, 0, 0),
    )

    print(f"[DONE] debug visualization saved to: {args.out}")
    print(f"[DONE] GT image:   {Path(args.out) / 'gt_rgb_with_labels.png'}")
    print(f"[DONE] Pred image: {Path(args.out) / 'pred_rgb_with_labels.png'}")


if __name__ == "__main__":
    main()