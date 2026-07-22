#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualize internal maps of LASCIModule.

This script saves:
    D       : feature difference map
    U       : cosine inconsistency map
    L       : window-wise low-light cue map
    score   : score produced by score_mlp, usually after sigmoid in LASCIModule
    prob    : softmax(score / tau), i.e. soft budget allocation ratio
    gate    : final soft budget gate
    gate_i2r: IR-to-RGB directional gate
    gate_r2i: RGB-to-IR directional gate

Typical usage:
    python visualize_lasci_maps.py \
        --weights runs/detect/train/weights/best.pt \
        --rgb /path/to/rgb.jpg \
        --ir /path/to/ir.jpg \
        --out vis_lasci \
        --imgsz 640 \
        --device cuda:0 \
        --save-overlay

Notes:
    1. The script assumes the model input is 6-channel RGBT input.
    2. Default channel order is [RGB, IR].
    3. IR image is loaded as grayscale and repeated to 3 channels by default.
    4. This script does not modify block.py permanently.
"""

import argparse
import math
import types
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser("Visualize LASCIModule maps")

    parser.add_argument("--weights", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/obb/LASCI_add/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--rgb", type=str, default="/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/images/test/00108.jpg", help="Path to RGB image")
    parser.add_argument("--ir", type=str, default="/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/image/test/00108.jpg", help="Path to IR image")
    parser.add_argument("--out", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/obb/LASCI_add/test_108", help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--input-order", type=str, default="rgbir", choices=["rgbir", "irrgb"])
    parser.add_argument("--ir-mode", type=str, default="gray_repeat", choices=["gray_repeat", "rgb"])
    parser.add_argument("--half", action="store_true", help="Use FP16 inference on CUDA")
    parser.add_argument("--module-filter", type=str, default="", help="Only visualize modules whose name contains this string")
    parser.add_argument("--max-modules", type=int, default=999, help="Maximum number of LASCIModule modules to save")
    parser.add_argument("--save-overlay", action="store_true", help="Save heatmap overlays on RGB image")
    parser.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap, e.g. magma, viridis, jet")
    parser.add_argument("--overlay-alpha", type=float, default=0.45, help="Overlay heatmap alpha")

    return parser.parse_args()
# 看这几张
# score_mlp.png
# prob_softmax_allocation.png
# gate_soft_budget.png
# gate_i2r_IR_to_RGB.png
# gate_r2i_RGB_to_IR.png
# all_maps_contact_sheet.png



# -----------------------------
# Image preprocessing
# -----------------------------

def letterbox(img, new_shape=640, color=(114, 114, 114)):
    h0, w0 = img.shape[:2]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    new_h, new_w = new_shape
    r = min(new_h / h0, new_w / w0)

    resized_w = int(round(w0 * r))
    resized_h = int(round(h0 * r))

    dw = new_w - resized_w
    dh = new_h - resized_h
    dw /= 2.0
    dh /= 2.0

    if (w0, h0) != (resized_w, resized_h):
        img = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out


def read_rgb(path):
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def read_ir(path, mode="gray_repeat"):
    if mode == "gray_repeat":
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read IR image: {path}")
        return np.stack([img, img, img], axis=-1)

    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read IR image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def build_input_tensor(rgb_path, ir_path, imgsz, input_order="rgbir", ir_mode="gray_repeat"):
    rgb = read_rgb(rgb_path)
    ir = read_ir(ir_path, mode=ir_mode)

    rgb_lb = letterbox(rgb, imgsz)
    ir_lb = letterbox(ir, imgsz)

    rgb_f = rgb_lb.astype(np.float32) / 255.0
    ir_f = ir_lb.astype(np.float32) / 255.0

    if input_order == "rgbir":
        x = np.concatenate([rgb_f, ir_f], axis=-1)
    else:
        x = np.concatenate([ir_f, rgb_f], axis=-1)

    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).contiguous()

    return x, rgb_lb, ir_lb


# -----------------------------
# LASCIModule monkey patch
# -----------------------------

def infer_grid_hw(num_windows):
    root = int(round(math.sqrt(num_windows)))
    best_gh, best_gw = 1, num_windows
    best_err = 10 ** 9

    for gh in range(1, int(math.sqrt(num_windows)) + 1):
        if num_windows % gh == 0:
            gw = num_windows // gh
            err = abs(gh - gw)
            if err < best_err:
                best_err = err
                best_gh, best_gw = gh, gw

    if best_gh * best_gw == num_windows:
        return best_gh, best_gw

    gh = max(root, 1)
    gw = int(math.ceil(num_windows / gh))
    return gh, gw


def squeeze_window_tensor(x):
    if not torch.is_tensor(x):
        return x

    if x.ndim == 3 and x.shape[-1] == 1:
        x = x.squeeze(-1)

    if x.ndim > 2:
        x = x.reshape(x.shape[0], -1)

    return x


def compute_prob_from_score(module, score):
    score = squeeze_window_tensor(score)

    if not torch.is_tensor(score):
        return None

    if score.ndim != 2:
        return None

    tau = float(getattr(module, "softmax_tau", 1.0))
    tau = max(tau, 1e-6)

    prob = torch.softmax(score / tau, dim=1)
    return prob


def patch_lasci_module(module, name):
    module._lasci_vis_name = name
    module._lasci_vis_cache = {}
    module._lasci_vis_records = []

    original_compute = module._compute_window_score
    original_budget = module._soft_budget_gate

    def wrapped_compute(self, *args, **kwargs):
        out = original_compute(*args, **kwargs)

        score = None
        diff_score = None
        incons_score = None

        if isinstance(out, (tuple, list)):
            if len(out) >= 1:
                score = out[0]
            if len(out) >= 2:
                diff_score = out[1]
            if len(out) >= 3:
                incons_score = out[2]
        else:
            score = out

        low_win = None
        if len(args) >= 3:
            low_win = args[2]
        elif "low_win" in kwargs:
            low_win = kwargs["low_win"]

        if torch.is_tensor(score):
            self._lasci_vis_cache["score"] = squeeze_window_tensor(score.detach())
        if torch.is_tensor(diff_score):
            self._lasci_vis_cache["D"] = squeeze_window_tensor(diff_score.detach())
        if torch.is_tensor(incons_score):
            self._lasci_vis_cache["U"] = squeeze_window_tensor(incons_score.detach())
        if torch.is_tensor(low_win):
            self._lasci_vis_cache["L"] = squeeze_window_tensor(low_win.detach())

        return out

    def wrapped_budget(self, *args, **kwargs):
        out = original_budget(*args, **kwargs)

        if isinstance(out, (tuple, list)):
            gate = out[0]
        else:
            gate = out

        score = None
        if len(args) >= 1:
            score = args[0]
        elif "score" in kwargs:
            score = kwargs["score"]
        elif "score" in self._lasci_vis_cache:
            score = self._lasci_vis_cache["score"]

        if torch.is_tensor(gate):
            gate = squeeze_window_tensor(gate.detach())
            self._lasci_vis_cache["gate"] = gate

        prob = compute_prob_from_score(self, score)
        if torch.is_tensor(prob):
            self._lasci_vis_cache["prob"] = squeeze_window_tensor(prob.detach())

        low_win = self._lasci_vis_cache.get("L", None)

        if torch.is_tensor(gate) and torch.is_tensor(low_win):
            low_win = low_win.to(device=gate.device, dtype=gate.dtype)
            lambda_low = float(getattr(self, "lambda_low", 0.0))

            gate_i2r = gate * (1.0 + lambda_low * low_win)
            gate_r2i = gate * (1.0 - lambda_low * low_win).clamp(min=0.0)

            self._lasci_vis_cache["gate_i2r"] = gate_i2r.detach()
            self._lasci_vis_cache["gate_r2i"] = gate_r2i.detach()

        record = {}
        for k, v in self._lasci_vis_cache.items():
            if torch.is_tensor(v):
                record[k] = squeeze_window_tensor(v).detach().float().cpu()

        nwin = None
        for key in ["gate", "prob", "score", "D", "U", "L"]:
            if key in record:
                nwin = int(record[key].reshape(record[key].shape[0], -1).shape[1])
                break

        if nwin is not None:
            gh, gw = infer_grid_hw(nwin)
            record["gh"] = gh
            record["gw"] = gw
            record["num_windows"] = nwin

        self._lasci_vis_records.append(record)
        self._lasci_vis_cache = {}

        return out

    module._compute_window_score = types.MethodType(wrapped_compute, module)
    module._soft_budget_gate = types.MethodType(wrapped_budget, module)


def patch_all_lasci_modules(model, module_filter="", max_modules=999):
    patched = []

    for name, module in model.named_modules():
        if module.__class__.__name__ != "LASCIModule":
            continue

        if module_filter and module_filter not in name:
            continue

        if len(patched) >= max_modules:
            break

        patch_lasci_module(module, name)
        patched.append((name, module))

    return patched


# -----------------------------
# Save visualization
# -----------------------------

def normalize01(x, eps=1e-6):
    x = np.asarray(x, dtype=np.float32)
    x_min = float(np.nanmin(x))
    x_max = float(np.nanmax(x))

    if x_max - x_min < eps:
        return np.zeros_like(x, dtype=np.float32)

    return (x - x_min) / (x_max - x_min + eps)


def tensor_to_grid(x, gh, gw, batch_idx=0):
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")

    x = squeeze_window_tensor(x)
    x = x.detach().float().cpu()

    if x.ndim == 2:
        v = x[batch_idx]
    else:
        v = x.flatten()

    v = v.flatten()
    need = gh * gw

    if v.numel() < need:
        pad = torch.zeros(need - v.numel(), dtype=v.dtype)
        v = torch.cat([v, pad], dim=0)
    elif v.numel() > need:
        v = v[:need]

    return v.reshape(gh, gw).numpy()


def save_heatmap(grid, path, title, cmap="magma"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = normalize01(grid)

    plt.figure(figsize=(4, 4))
    plt.imshow(data, cmap=cmap, interpolation="nearest")
    plt.title(title, fontsize=10)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_overlay(grid, rgb_img, path, alpha=0.45):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    heat = normalize01(grid)
    heat = cv2.resize(
        heat,
        (rgb_img.shape[1], rgb_img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    heat_u8 = np.uint8(255 * heat)
    heat_color_bgr = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color_bgr, cv2.COLOR_BGR2RGB)

    overlay = np.uint8((1.0 - alpha) * rgb_img + alpha * heat_color)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(path), overlay_bgr)


def save_contact_sheet(items, path, cmap="magma"):
    if not items:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n = len(items)
    cols = min(4, n)
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(4 * cols, 3.5 * rows))

    for i, (name, grid) in enumerate(items, start=1):
        plt.subplot(rows, cols, i)
        plt.imshow(normalize01(grid), cmap=cmap, interpolation="nearest")
        plt.title(name, fontsize=10)
        plt.axis("off")
        plt.colorbar(fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(str(path), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()


def save_one_module(record, module_dir, rgb_img, cmap="magma", save_overlay_flag=False, overlay_alpha=0.45):
    module_dir = Path(module_dir)
    module_dir.mkdir(parents=True, exist_ok=True)

    if "gh" not in record or "gw" not in record:
        raise RuntimeError("Cannot find gh/gw in LASCIModule debug record.")

    gh = int(record["gh"])
    gw = int(record["gw"])

    key_name_pairs = [
        ("D", "D_feature_difference"),
        ("U", "U_cosine_inconsistency"),
        ("L", "L_low_light_cue"),
        ("score", "score_mlp"),
        ("prob", "prob_softmax_allocation"),
        ("gate", "gate_soft_budget"),
        ("gate_i2r", "gate_i2r_IR_to_RGB"),
        ("gate_r2i", "gate_r2i_RGB_to_IR"),
    ]

    contact_items = []

    for key, name in key_name_pairs:
        if key not in record:
            continue

        grid = tensor_to_grid(record[key], gh, gw)
        np.save(str(module_dir / f"{name}.npy"), grid)

        save_heatmap(
            grid=grid,
            path=module_dir / f"{name}.png",
            title=name,
            cmap=cmap,
        )

        contact_items.append((name, grid))

        if save_overlay_flag:
            save_overlay(
                grid=grid,
                rgb_img=rgb_img,
                path=module_dir / f"{name}_overlay.png",
                alpha=overlay_alpha,
            )

    save_contact_sheet(
        items=contact_items,
        path=module_dir / "all_maps_contact_sheet.png",
        cmap=cmap,
    )

    with open(module_dir / "meta.txt", "w", encoding="utf-8") as f:
        f.write(f"gh={gh}\n")
        f.write(f"gw={gw}\n")
        f.write(f"num_windows={record.get('num_windows', gh * gw)}\n")
        f.write("keys:\n")
        for key, name in key_name_pairs:
            if key in record:
                f.write(f"  {key}: {name}\n")


# -----------------------------
# Model loading and forward
# -----------------------------

def load_model(weights, device, half=False):
    from ultralytics import YOLO

    yolo = YOLO(weights)
    model = yolo.model

    model.to(device)
    model.eval()

    if half and device.startswith("cuda"):
        model.half()
    else:
        model.float()

    return yolo, model


def run_forward(model, x, device, half=False):
    x = x.to(device)

    if half and device.startswith("cuda"):
        x = x.half()
    else:
        x = x.float()

    with torch.no_grad():
        _ = model(x)


def main():
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Use CPU instead.")
        device = "cpu"

    x, rgb_lb, ir_lb = build_input_tensor(
        rgb_path=args.rgb,
        ir_path=args.ir,
        imgsz=args.imgsz,
        input_order=args.input_order,
        ir_mode=args.ir_mode,
    )

    cv2.imwrite(
        str(out_dir / "input_rgb_letterbox.png"),
        cv2.cvtColor(rgb_lb, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        str(out_dir / "input_ir_letterbox.png"),
        cv2.cvtColor(ir_lb, cv2.COLOR_RGB2BGR),
    )

    _, model = load_model(
        weights=args.weights,
        device=device,
        half=args.half,
    )

    patched = patch_all_lasci_modules(
        model=model,
        module_filter=args.module_filter,
        max_modules=args.max_modules,
    )

    if not patched:
        raise RuntimeError(
            "No LASCIModule found in this model. "
            "Check whether your best.pt contains LASCIModule and whether the class name is exactly 'LASCIModule'."
        )

    print("[INFO] Patched LASCIModule modules:")
    for i, (name, _) in enumerate(patched):
        print(f"  [{i}] {name}")

    run_forward(
        model=model,
        x=x,
        device=device,
        half=args.half,
    )

    saved = 0

    for idx, (name, module) in enumerate(patched):
        records = getattr(module, "_lasci_vis_records", [])

        if not records:
            print(f"[WARN] No record captured for module: {name}")
            continue

        record = records[-1]
        safe_name = name.replace(".", "_").replace("/", "_")
        module_dir = out_dir / f"{idx:02d}_{safe_name}"

        save_one_module(
            record=record,
            module_dir=module_dir,
            rgb_img=rgb_lb,
            cmap=args.cmap,
            save_overlay_flag=args.save_overlay,
            overlay_alpha=args.overlay_alpha,
        )

        print(f"[INFO] Saved maps for {name} -> {module_dir}")
        saved += 1

    print(f"[DONE] Saved visualization for {saved} LASCIModule module(s).")
    print(f"[DONE] Output directory: {out_dir}")


if __name__ == "__main__":
    main()