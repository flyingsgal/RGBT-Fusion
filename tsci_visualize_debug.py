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


def draw_boxes_rgb(img_rgb, boxes):
    """
    在 RGB 图像上画 HBB 或 OBB。
    """
    out = img_rgb.copy()

    for item in boxes:
        if item["type"] == "hbb":
            x1, y1, x2, y2 = map(int, item["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        elif item["type"] == "obb":
            pts = np.array(item["pts"], dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    return out


def load_yolo_label(label_path, imgsz):
    """
    支持两种格式：
    HBB: cls cx cy w h
    OBB: cls x1 y1 x2 y2 x3 y3 x4 y4

    坐标默认是归一化坐标。
    """
    boxes = []

    if label_path is None:
        return boxes

    label_path = Path(label_path)
    if not label_path.exists():
        print(f"[WARN] label file not found: {label_path}")
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) == 0:
            continue

        vals = [float(x) for x in parts]

        if len(vals) == 5:
            _, cx, cy, w, h = vals
            x1 = (cx - w / 2) * imgsz
            y1 = (cy - h / 2) * imgsz
            x2 = (cx + w / 2) * imgsz
            y2 = (cy + h / 2) * imgsz
            boxes.append({"type": "hbb", "box": [x1, y1, x2, y2]})

        elif len(vals) == 9:
            coords = vals[1:]
            pts = []
            for i in range(0, 8, 2):
                x = coords[i] * imgsz
                y = coords[i + 1] * imgsz
                pts.append([x, y])
            boxes.append({"type": "obb", "pts": pts})

        else:
            print(f"[WARN] unsupported label line with {len(vals)} values: {line.strip()}")

    return boxes


def save_input_images(x, out_dir, boxes=None):
    """
    x: [1,6,H,W]
    """
    img = x[0].detach().cpu()
    rgb = img[:3]
    ir = img[3:6]

    rgb_np = to_uint8_img(rgb)
    ir_np = to_uint8_img(ir)

    if boxes is not None:
        rgb_np = draw_boxes_rgb(rgb_np, boxes)
        ir_np = draw_boxes_rgb(ir_np, boxes)

    cv2.imwrite(str(Path(out_dir) / "rgb.png"), cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(Path(out_dir) / "ir.png"), cv2.cvtColor(ir_np, cv2.COLOR_RGB2BGR))


def find_tsci_modules(net):
    """
    不强依赖 import 路径，只按类名找 TSCIv2RawCueFusion。
    """
    modules = []
    for m in net.modules():
        if m.__class__.__name__ == "TSCIv2RawCueFusion":
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

    parser.add_argument("--weights", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/TSCIv2RawCueFusion/with_r_dark_ir_max/weights/best.pt", help="训练好的 best.pt 或 last.pt")
    parser.add_argument("--rgb", type=str, default="/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/images/test/04514.jpg", help="RGB 图像路径")
    parser.add_argument("--ir", type=str, default="/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/image/test/04514.jpg", help="IR 图像路径")
    parser.add_argument("--label", type=str, default="/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/labels/test/04514.txt", help="可选：YOLO label txt 路径，用于画 GT 框")
    parser.add_argument("--out", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/TSCIv2RawCueFusion/with_r_dark_ir_max/test_hbb_4514", help="输出目录")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 或 cpu")
    parser.add_argument("--overlay-alpha", type=float, default=0.45)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ensure_dir(args.out)

    print(f"[INFO] loading model: {args.weights}")
    model = YOLO(args.weights)
    net = model.model.to(device).float().eval()

    tsci_modules = find_tsci_modules(net)

    if len(tsci_modules) == 0:
        raise RuntimeError(
            "没有找到 TSCIv2RawCueFusion 模块。请确认：\n"
            "1. 当前权重对应的模型结构里确实用了 TSCIv2RawCueFusion；\n"
            "2. 类名是否叫 TSCIv2RawCueFusion；\n"
            "3. 不是加载了 RawCueAddC 或 ADD baseline 的权重。"
        )

    print(f"[INFO] found {len(tsci_modules)} TSCIv2RawCueFusion modules.")

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

    boxes = load_yolo_label(args.label, args.imgsz) if args.label else None

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

    print(f"[DONE] debug visualization saved to: {args.out}")


if __name__ == "__main__":
    main()