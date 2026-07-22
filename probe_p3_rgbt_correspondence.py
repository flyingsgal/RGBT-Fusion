#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frozen-feature P3 correspondence probe for two-stream Ultralytics YOLOv8.

Purpose
-------
This is a diagnostic experiment, not a detector-training script. It freezes an
Add-baseline checkpoint, captures the pre-fusion RGB/IR P3 feature maps, builds
a local 5*5 correlation surface for every strictly matched IR object, and asks:

    Can the IR query recover the annotated RGB sampling position?

The script reports argmax/softargmax endpoint error (EPE), response margin,
entropy, spatial variance, forward-backward cycle consistency, image-bootstrap
confidence intervals, and confidence risk-coverage curves.

Expected model layout (overridable from CLI)
---------------------------------------------
    RGB P3: model.model[11]
    IR  P3: model.model[12]
    P3 stride: 8

Expected state CSV columns
--------------------------
Required:
    image_id, rgb_cx, rgb_cy, ir_cx, ir_cy

Recommended:
    split, cls, rgb_source_index, ir_source_index, dx, dy, dist,
    geometry_state, mean_area, rgb_visible_ratio, ir_visible_ratio,
    rgb_was_clipped, ir_was_clipped

Important sign convention
-------------------------
The target vector is computed directly as:

    RGB sample position - IR query position

so no manual sign reversal is needed. With stride=8:

    gt_dx_cell = (rgb_cx - ir_cx) * resize_gain_x / 8
    gt_dy_cell = (rgb_cy - ir_cy) * resize_gain_y / 8

Typical usage
-------------
1) Syntax/import-independent self-test in the server environment:

    python probe_p3_rgbt_correspondence.py --self-test

2) First 100 validation images:

    python probe_p3_rgbt_correspondence.py \
      --weights /path/to/add/best.pt \
      --data /path/to/dronevehicle_obb.yaml \
      --state-csv /path/to/val_matched_state_index.csv \
      --split val --states near_center,local_shift \
      --output runs/p3_correspondence/val_100 \
      --device 0 --batch 8 --imgsz 640 --max-images 100 --tau 0.10

3) Full validation after the 100-image output has been checked:

    python probe_p3_rgbt_correspondence.py \
      --weights /path/to/add/best.pt \
      --data /path/to/dronevehicle_obb.yaml \
      --state-csv /path/to/val_matched_state_index.csv \
      --split val --states near_center,local_shift \
      --output runs/p3_correspondence/val_full \
      --device 0 --batch 8 --imgsz 640 --tau 0.10

4) Optional temperature calibration on a deterministic train subset. The CSV
   must have the same schema but contain train objects. Do not report train
   calibration numbers as final evidence:

    python probe_p3_rgbt_correspondence.py \
      --weights /path/to/add/best.pt \
      --data /path/to/dronevehicle_obb.yaml \
      --state-csv /path/to/train_matched_state_index.csv \
      --split train \
      --output runs/p3_correspondence/train_calibration \
      --device 0 --batch 8 --imgsz 640 --max-images 2000 \
      --calibrate-tau --tau-grid 0.03,0.05,0.10,0.20,0.30,0.50

Notes
-----
* Run this file from the root of the user's modified TwoStream YOLO repository
  so that its local `ultralytics` package is imported instead of a pip version.
* Validation must be single-process and non-shuffled. DDP is intentionally not
  supported because object/image bookkeeping must remain exact.
* RGB boxes are used only as an offline measuring instrument. No RGB-box loss
  is introduced into detector training.
"""

import argparse
import json
import math
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class StopProbeEarly(RuntimeError):
    """Internal control-flow exception used by --max-images."""


def require_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is unavailable. Run this script inside the same conda "
            "environment used to train the TwoStream YOLO model."
        ) from exc
    return torch, F


def signed_tag(value: int) -> str:
    if value < 0:
        return "m{}".format(abs(value))
    if value > 0:
        return "p{}".format(value)
    return "z0"


def candidate_column(dx: int, dy: int) -> str:
    return "score_dy_{}_dx_{}".format(signed_tag(dy), signed_tag(dx))


def parse_tau_grid(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0:
            raise ValueError("All temperature values must be positive.")
        values.append(value)
    if not values:
        raise ValueError("Empty --tau-grid.")
    return values


def first_tensor(value: Any):
    torch, _ = require_torch()
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


class TensorCapture:
    def __init__(self, module: Any, name: str):
        self.name = name
        self.tensor = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        tensor = first_tensor(output)
        if tensor is None:
            raise RuntimeError("{} hook did not receive a Tensor output.".format(self.name))
        self.tensor = tensor

    def close(self) -> None:
        self.handle.remove()


def read_state_csv(
    path: str, split: Optional[str], states: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype={"image_id": str, "split": str},
        low_memory=False,
    )
    required = {"image_id", "rgb_cx", "rgb_cy", "ir_cx", "ir_cy"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError("State CSV is missing required columns: {}".format(missing))

    df["image_id"] = df["image_id"].astype(str).str.strip()
    if split and "split" in df.columns:
        df = df[df["split"].astype(str).str.lower() == split.lower()].copy()
    if states:
        if "geometry_state" not in df.columns:
            raise ValueError("--states was provided but geometry_state is absent from the CSV.")
        normalized_states = {str(state).strip() for state in states if str(state).strip()}
        df = df[df["geometry_state"].astype(str).isin(normalized_states)].copy()
    if len(df) == 0:
        raise ValueError("No state rows remain after split filtering.")

    numeric_columns = [
        "rgb_cx",
        "rgb_cy",
        "ir_cx",
        "ir_cy",
        "mean_area",
        "rgb_area",
        "ir_area",
        "rgb_visible_ratio",
        "ir_visible_ratio",
        "rgb_was_clipped",
        "ir_was_clipped",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.reset_index(drop=True)


def image_size(path: str) -> Tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        return int(height), int(width)
    except Exception:
        try:
            import cv2

            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError("cv2.imread returned None")
            return int(image.shape[0]), int(image.shape[1])
        except Exception as exc:
            raise RuntimeError("Cannot read image shape for {}".format(path)) from exc


def letterbox_geometry(
    orig_h: int, orig_w: int, input_h: int, input_w: int
) -> Dict[str, float]:
    gain = min(float(input_h) / float(orig_h), float(input_w) / float(orig_w))
    resized_w = int(round(orig_w * gain))
    resized_h = int(round(orig_h * gain))
    gain_x = float(resized_w) / float(orig_w)
    gain_y = float(resized_h) / float(orig_h)
    pad_x = (float(input_w) - float(resized_w)) / 2.0
    pad_y = (float(input_h) - float(resized_h)) / 2.0
    return {
        "gain_x": gain_x,
        "gain_y": gain_y,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "resized_w": float(resized_w),
        "resized_h": float(resized_h),
    }


def sample_feature(feature: Any, points_xy: Any):
    """Bilinearly sample CxHxW feature at arbitrary (..., 2) feature coordinates."""
    torch, F = require_torch()
    if feature.ndim != 3:
        raise ValueError("Expected CxHxW feature, got {}".format(tuple(feature.shape)))
    c, h, w = feature.shape
    original_shape = tuple(points_xy.shape[:-1])
    points = points_xy.reshape(-1, 2)
    grid_x = 2.0 * (points[:, 0] + 0.5) / float(w) - 1.0
    grid_y = 2.0 * (points[:, 1] + 0.5) / float(h) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 1, 2)
    sampled = F.grid_sample(
        feature.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    sampled = sampled.squeeze(0).squeeze(-1).transpose(0, 1)
    return sampled.reshape(*original_shape, c)


def points_inside(points_xy: Any, bounds: Tuple[float, float, float, float]):
    xmin, xmax, ymin, ymax = bounds
    x = points_xy[..., 0]
    y = points_xy[..., 1]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def infer_mean_area(row: pd.Series, gain_x: float, gain_y: float) -> float:
    if "mean_area" in row and pd.notna(row["mean_area"]):
        area = float(row["mean_area"])
    elif (
        "rgb_area" in row
        and "ir_area" in row
        and pd.notna(row["rgb_area"])
        and pd.notna(row["ir_area"])
    ):
        area = 0.5 * (float(row["rgb_area"]) + float(row["ir_area"]))
    else:
        return float("nan")
    return area * gain_x * gain_y


def size_group(area: float) -> str:
    if not np.isfinite(area):
        return "unknown"
    if area < 32.0 * 32.0:
        return "small"
    if area < 96.0 * 96.0:
        return "medium"
    return "large"


def offset_bin(distance_px: float) -> str:
    if not np.isfinite(distance_px):
        return "unknown"
    if distance_px <= 2.0:
        return "0-2"
    if distance_px < 4.0:
        return "2-4"
    if distance_px < 6.0:
        return "4-6"
    if distance_px < 8.0:
        return "6-8"
    if distance_px < 12.0:
        return "8-12"
    if distance_px <= 16.0:
        return "12-16"
    return ">16"


def row_value(row: pd.Series, key: str, default: Any = None) -> Any:
    if key not in row or pd.isna(row[key]):
        return default
    value = row[key]
    if isinstance(value, np.generic):
        return value.item()
    return value


def analyze_image_features(
    rgb_feature: Any,
    ir_feature: Any,
    rows: pd.DataFrame,
    image_id: str,
    image_path: str,
    orig_h: int,
    orig_w: int,
    stride: int,
    search_radius: int,
    patch_radius: int,
    keep_scores: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Analyze all matched objects for one image with vectorized sampling."""
    torch, F = require_torch()
    stats = {"requested": int(len(rows)), "invalid_query": 0, "no_candidate": 0}
    if len(rows) == 0:
        return [], stats
    if rgb_feature.shape != ir_feature.shape:
        raise RuntimeError(
            "RGB/IR P3 feature mismatch: {} vs {}".format(
                tuple(rgb_feature.shape), tuple(ir_feature.shape)
            )
        )

    rgb_feature = rgb_feature.detach().float()
    ir_feature = ir_feature.detach().float()
    _c, feature_h, feature_w = rgb_feature.shape
    input_h = int(feature_h * stride)
    input_w = int(feature_w * stride)
    geom = letterbox_geometry(orig_h, orig_w, input_h, input_w)
    gain_x = geom["gain_x"]
    gain_y = geom["gain_y"]
    pad_x = geom["pad_x"]
    pad_y = geom["pad_y"]

    device = rgb_feature.device
    dtype = rgb_feature.dtype

    # Feature coordinate x corresponds to input coordinate (x + 0.5) * stride.
    ir_cx = torch.as_tensor(rows["ir_cx"].to_numpy(float), device=device, dtype=dtype)
    ir_cy = torch.as_tensor(rows["ir_cy"].to_numpy(float), device=device, dtype=dtype)
    rgb_cx = torch.as_tensor(rows["rgb_cx"].to_numpy(float), device=device, dtype=dtype)
    rgb_cy = torch.as_tensor(rows["rgb_cy"].to_numpy(float), device=device, dtype=dtype)
    center_x = (ir_cx * gain_x + pad_x) / float(stride) - 0.5
    center_y = (ir_cy * gain_y + pad_y) / float(stride) - 0.5
    centers = torch.stack([center_x, center_y], dim=-1)

    candidate_list = [
        (dx, dy)
        for dy in range(-search_radius, search_radius + 1)
        for dx in range(-search_radius, search_radius + 1)
    ]
    candidates = torch.as_tensor(candidate_list, device=device, dtype=dtype)
    patch_list = [
        (dx, dy)
        for dy in range(-patch_radius, patch_radius + 1)
        for dx in range(-patch_radius, patch_radius + 1)
    ]
    patch_offsets = torch.as_tensor(patch_list, device=device, dtype=dtype)
    n_objects = centers.shape[0]
    n_candidates = candidates.shape[0]
    n_patch = patch_offsets.shape[0]

    query_points = centers[:, None, :] + patch_offsets[None, :, :]
    candidate_points = (
        centers[:, None, None, :]
        + candidates[None, :, None, :]
        + patch_offsets[None, None, :, :]
    )

    # Exclude samples that enter letterbox padding, not merely feature-map bounds.
    content_xmin = pad_x / float(stride) - 0.5
    content_ymin = pad_y / float(stride) - 0.5
    content_xmax = (pad_x + geom["resized_w"] - 1.0) / float(stride) - 0.5
    content_ymax = (pad_y + geom["resized_h"] - 1.0) / float(stride) - 0.5
    bounds = (content_xmin, content_xmax, content_ymin, content_ymax)
    query_valid = points_inside(query_points, bounds).all(dim=-1)
    candidate_valid = points_inside(candidate_points, bounds).all(dim=-1)

    ir_patch = sample_feature(ir_feature, query_points)  # N,P,C
    rgb_patches = sample_feature(rgb_feature, candidate_points)  # N,K,P,C
    ir_patch = F.normalize(ir_patch, dim=-1, eps=1e-6)
    rgb_patches = F.normalize(rgb_patches, dim=-1, eps=1e-6)
    scores = (ir_patch[:, None, :, :] * rgb_patches).sum(dim=-1).mean(dim=-1)
    scores = scores.masked_fill(~candidate_valid, float("-inf"))

    valid_counts = candidate_valid.sum(dim=-1)
    usable = query_valid & (valid_counts > 0)
    safe_scores = scores.clone()
    safe_scores[~usable] = 0.0
    argmax_idx = safe_scores.argmax(dim=-1)
    pred_argmax = candidates[argmax_idx]
    top_values, _ = safe_scores.topk(k=min(2, n_candidates), dim=-1)

    # Reverse search from the forward argmax RGB position back to IR.
    rgb_query_centers = centers + pred_argmax
    reverse_query_points = rgb_query_centers[:, None, :] + patch_offsets[None, :, :]
    reverse_candidate_points = (
        rgb_query_centers[:, None, None, :]
        + candidates[None, :, None, :]
        + patch_offsets[None, None, :, :]
    )
    reverse_query_valid = points_inside(reverse_query_points, bounds).all(dim=-1)
    reverse_candidate_valid = points_inside(reverse_candidate_points, bounds).all(dim=-1)
    rgb_reverse_query = sample_feature(rgb_feature, reverse_query_points)
    ir_reverse_patches = sample_feature(ir_feature, reverse_candidate_points)
    rgb_reverse_query = F.normalize(rgb_reverse_query, dim=-1, eps=1e-6)
    ir_reverse_patches = F.normalize(ir_reverse_patches, dim=-1, eps=1e-6)
    reverse_scores = (
        rgb_reverse_query[:, None, :, :] * ir_reverse_patches
    ).sum(dim=-1).mean(dim=-1)
    reverse_scores = reverse_scores.masked_fill(~reverse_candidate_valid, float("-inf"))
    reverse_usable = reverse_query_valid & (reverse_candidate_valid.sum(dim=-1) > 0)
    reverse_safe = reverse_scores.clone()
    reverse_safe[~reverse_usable] = 0.0
    reverse_idx = reverse_safe.argmax(dim=-1)
    reverse_pred = candidates[reverse_idx]
    cycle_error = torch.linalg.vector_norm(pred_argmax + reverse_pred, dim=-1)
    mutual = (pred_argmax + reverse_pred).abs().max(dim=-1).values < 1e-6

    gt_dx = (rgb_cx - ir_cx) * gain_x / float(stride)
    gt_dy = (rgb_cy - ir_cy) * gain_y / float(stride)
    gt_delta = torch.stack([gt_dx, gt_dy], dim=-1)
    zero_epe = torch.linalg.vector_norm(gt_delta, dim=-1) * float(stride)
    argmax_epe = torch.linalg.vector_norm(pred_argmax - gt_delta, dim=-1) * float(stride)

    scores_cpu = scores.detach().cpu().numpy()
    valid_cpu = usable.detach().cpu().numpy().astype(bool)
    query_valid_cpu = query_valid.detach().cpu().numpy().astype(bool)
    valid_counts_cpu = valid_counts.detach().cpu().numpy()
    pred_cpu = pred_argmax.detach().cpu().numpy()
    gt_cpu = gt_delta.detach().cpu().numpy()
    zero_epe_cpu = zero_epe.detach().cpu().numpy()
    argmax_epe_cpu = argmax_epe.detach().cpu().numpy()
    top_cpu = top_values.detach().cpu().numpy()
    cycle_cpu = cycle_error.detach().cpu().numpy()
    mutual_cpu = mutual.detach().cpu().numpy().astype(bool)
    reverse_usable_cpu = reverse_usable.detach().cpu().numpy().astype(bool)

    output: List[Dict[str, Any]] = []
    for index in range(n_objects):
        if not query_valid_cpu[index]:
            stats["invalid_query"] += 1
            continue
        if not valid_cpu[index]:
            stats["no_candidate"] += 1
            continue
        row = rows.iloc[index]
        gt_dist_px = float(zero_epe_cpu[index])
        area = infer_mean_area(row, gain_x, gain_y)
        record: Dict[str, Any] = {
            "image_id": image_id,
            "image_path": image_path,
            "class_label": row_value(row, "cls", row_value(row, "class_id", "unknown")),
            "rgb_source_index": row_value(row, "rgb_source_index", -1),
            "ir_source_index": row_value(row, "ir_source_index", -1),
            "geometry_state": row_value(row, "geometry_state", "unknown"),
            "size_group": size_group(area),
            "mean_area_input_px2": area,
            "offset_bin": offset_bin(gt_dist_px),
            "offset_distance_input_px": gt_dist_px,
            "orig_h": orig_h,
            "orig_w": orig_w,
            "input_h": input_h,
            "input_w": input_w,
            "feature_h": feature_h,
            "feature_w": feature_w,
            "gain_x": gain_x,
            "gain_y": gain_y,
            "pad_x": pad_x,
            "pad_y": pad_y,
            "ir_center_x_feature": float(center_x[index].item()),
            "ir_center_y_feature": float(center_y[index].item()),
            "gt_dx_cell": float(gt_cpu[index, 0]),
            "gt_dy_cell": float(gt_cpu[index, 1]),
            "pred_argmax_dx_cell": float(pred_cpu[index, 0]),
            "pred_argmax_dy_cell": float(pred_cpu[index, 1]),
            "zero_epe_px": gt_dist_px,
            "argmax_epe_px": float(argmax_epe_cpu[index]),
            "argmax_gain_vs_zero_px": gt_dist_px - float(argmax_epe_cpu[index]),
            "top1_score": float(top_cpu[index, 0]),
            "top2_score": float(top_cpu[index, 1]) if top_cpu.shape[1] > 1 else float("nan"),
            "score_margin": (
                float(top_cpu[index, 0] - top_cpu[index, 1])
                if top_cpu.shape[1] > 1
                else float("nan")
            ),
            "valid_candidate_count": int(valid_counts_cpu[index]),
            "cycle_error_cell": (
                float(cycle_cpu[index]) if reverse_usable_cpu[index] else float("nan")
            ),
            "mutual_match": bool(mutual_cpu[index]) if reverse_usable_cpu[index] else False,
            "reverse_search_valid": bool(reverse_usable_cpu[index]),
            "peak_on_boundary": bool(
                abs(float(pred_cpu[index, 0])) == search_radius
                or abs(float(pred_cpu[index, 1])) == search_radius
            ),
            "rgb_visible_ratio": row_value(row, "rgb_visible_ratio", float("nan")),
            "ir_visible_ratio": row_value(row, "ir_visible_ratio", float("nan")),
            "rgb_was_clipped": row_value(row, "rgb_was_clipped", float("nan")),
            "ir_was_clipped": row_value(row, "ir_was_clipped", float("nan")),
        }
        if keep_scores:
            for candidate_index, (dx, dy) in enumerate(candidate_list):
                score = scores_cpu[index, candidate_index]
                record[candidate_column(dx, dy)] = float(score) if np.isfinite(score) else np.nan
        output.append(record)
    return output, stats


def softmax_from_scores(scores: np.ndarray, tau: float) -> np.ndarray:
    valid = np.isfinite(scores)
    safe = np.where(valid, scores, -np.inf)
    max_score = np.max(safe, axis=1, keepdims=True)
    max_score[~np.isfinite(max_score)] = 0.0
    exp_scores = np.where(valid, np.exp((safe - max_score) / tau), 0.0)
    denominator = exp_scores.sum(axis=1, keepdims=True)
    denominator[denominator <= 0.0] = 1.0
    return exp_scores / denominator


def attach_soft_metrics(
    df: pd.DataFrame, search_radius: int, stride: int, tau: float
) -> pd.DataFrame:
    candidate_list = [
        (dx, dy)
        for dy in range(-search_radius, search_radius + 1)
        for dx in range(-search_radius, search_radius + 1)
    ]
    score_columns = [candidate_column(dx, dy) for dx, dy in candidate_list]
    missing = [column for column in score_columns if column not in df.columns]
    if missing:
        raise ValueError("Missing candidate score columns: {}".format(missing[:5]))
    scores = df[score_columns].to_numpy(float)
    probabilities = softmax_from_scores(scores, tau)
    candidates = np.asarray(candidate_list, dtype=float)
    pred = probabilities @ candidates
    gt = df[["gt_dx_cell", "gt_dy_cell"]].to_numpy(float)
    epe = np.linalg.norm(pred - gt, axis=1) * float(stride)
    valid_count = np.isfinite(scores).sum(axis=1)
    entropy_raw = -np.sum(
        np.where(probabilities > 0, probabilities * np.log(probabilities + 1e-12), 0.0),
        axis=1,
    )
    entropy_den = np.log(np.maximum(valid_count, 2))
    entropy = entropy_raw / entropy_den
    delta = candidates[None, :, :] - pred[:, None, :]
    variance = np.sum(probabilities * np.sum(delta * delta, axis=-1), axis=1)

    result = df.copy()
    result["softmax_tau"] = float(tau)
    result["pred_soft_dx_cell"] = pred[:, 0]
    result["pred_soft_dy_cell"] = pred[:, 1]
    result["soft_epe_px"] = epe
    result["soft_gain_vs_zero_px"] = result["zero_epe_px"].to_numpy(float) - epe
    result["normalized_entropy"] = entropy
    result["entropy_certainty"] = 1.0 - entropy
    result["spatial_variance_cell2"] = variance
    result["strict_correct_4px"] = epe <= 4.0
    result["usable_correct_8px"] = epe <= 8.0
    result["argmax_strict_correct_4px"] = result["argmax_epe_px"].to_numpy(float) <= 4.0
    result["argmax_usable_correct_8px"] = result["argmax_epe_px"].to_numpy(float) <= 8.0
    return result


def balanced_tau_objective(df: pd.DataFrame) -> Tuple[float, float, float]:
    near = df.loc[df["geometry_state"] == "near_center", "soft_epe_px"].mean()
    shift = df.loc[df["geometry_state"] == "local_shift", "soft_epe_px"].mean()
    available = [value for value in [near, shift] if np.isfinite(value)]
    objective = float(np.mean(available)) if available else float("nan")
    return objective, float(near), float(shift)


def calibrate_tau(
    raw_df: pd.DataFrame,
    search_radius: int,
    stride: int,
    tau_grid: Sequence[float],
) -> Tuple[float, pd.DataFrame]:
    records = []
    for tau in tau_grid:
        evaluated = attach_soft_metrics(raw_df, search_radius, stride, tau)
        objective, near_epe, shift_epe = balanced_tau_objective(evaluated)
        records.append(
            {
                "tau": tau,
                "balanced_state_mean_epe_px": objective,
                "near_center_mean_epe_px": near_epe,
                "local_shift_mean_epe_px": shift_epe,
                "all_mean_epe_px": evaluated["soft_epe_px"].mean(),
                "all_strict_correct_4px": evaluated["strict_correct_4px"].mean(),
                "all_usable_correct_8px": evaluated["usable_correct_8px"].mean(),
            }
        )
    table = pd.DataFrame(records).sort_values("tau").reset_index(drop=True)
    finite = table[np.isfinite(table["balanced_state_mean_epe_px"])].copy()
    if len(finite) == 0:
        raise RuntimeError("Temperature calibration produced no finite objective.")
    best_row = finite.sort_values(["balanced_state_mean_epe_px", "tau"]).iloc[0]
    return float(best_row["tau"]), table


def average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask]
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / (np.arange(len(sorted_labels)) + 1.0)
    return float(precision[sorted_labels].sum() / positives)


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    rank_sum_pos = ranks[labels].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman_correlation(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rank_x = pd.Series(x[mask]).rank(method="average").to_numpy(float)
    rank_y = pd.Series(y[mask]).rank(method="average").to_numpy(float)
    if np.std(rank_x) <= 0 or np.std(rank_y) <= 0:
        return float("nan")
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def confidence_signals(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        "top1_score": df["top1_score"].to_numpy(float),
        "score_margin": df["score_margin"].to_numpy(float),
        "entropy_certainty": df["entropy_certainty"].to_numpy(float),
        "negative_spatial_variance": -df["spatial_variance_cell2"].to_numpy(float),
        "negative_cycle_error": -df["cycle_error_cell"].to_numpy(float),
        "mutual_match": df["mutual_match"].astype(float).to_numpy(float),
    }


def iter_named_subsets(df: pd.DataFrame) -> Iterable[Tuple[str, pd.DataFrame]]:
    yield "all", df
    for state in ["near_center", "local_shift"]:
        subset = df[df["geometry_state"] == state]
        if len(subset):
            yield "state:{}".format(state), subset
    for size in ["small", "medium", "large"]:
        subset = df[df["size_group"] == size]
        if len(subset):
            yield "size:{}".format(size), subset
    small_shift = df[
        (df["geometry_state"] == "local_shift") & (df["size_group"] == "small")
    ]
    if len(small_shift):
        yield "state_size:local_shift_small", small_shift


def build_confidence_tables(
    df: pd.DataFrame, coverage_values: Sequence[float]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    quality_records: List[Dict[str, Any]] = []
    risk_records: List[Dict[str, Any]] = []
    for subset_name, subset in iter_named_subsets(df):
        labels4 = subset["strict_correct_4px"].astype(bool).to_numpy()
        labels8 = subset["usable_correct_8px"].astype(bool).to_numpy()
        epe = subset["soft_epe_px"].to_numpy(float)
        for signal_name, signal in confidence_signals(subset).items():
            quality_records.append(
                {
                    "subset": subset_name,
                    "signal": signal_name,
                    "object_count": len(subset),
                    "finite_score_count": int(np.isfinite(signal).sum()),
                    "auroc_correct_4px": roc_auc_binary(labels4, signal),
                    "auprc_correct_4px": average_precision_binary(labels4, signal),
                    "auroc_correct_8px": roc_auc_binary(labels8, signal),
                    "auprc_correct_8px": average_precision_binary(labels8, signal),
                    "spearman_confidence_vs_negative_epe": spearman_correlation(signal, -epe),
                }
            )
            finite_indices = np.where(np.isfinite(signal) & np.isfinite(epe))[0]
            if len(finite_indices) == 0:
                continue
            ordered = finite_indices[np.argsort(-signal[finite_indices], kind="mergesort")]
            for coverage in coverage_values:
                keep = max(1, int(math.ceil(len(ordered) * coverage)))
                selected = ordered[:keep]
                risk_records.append(
                    {
                        "subset": subset_name,
                        "signal": signal_name,
                        "coverage": coverage,
                        "selected_count": keep,
                        "mean_soft_epe_px": float(np.mean(epe[selected])),
                        "median_soft_epe_px": float(np.median(epe[selected])),
                        "strict_correct_4px": float(np.mean(labels4[selected])),
                        "usable_correct_8px": float(np.mean(labels8[selected])),
                    }
                )
    return pd.DataFrame(quality_records), pd.DataFrame(risk_records)


def bootstrap_image_mean_ci(
    df: pd.DataFrame,
    value_column: str,
    repeats: int,
    seed: int,
) -> Tuple[float, float]:
    image_means = df.groupby("image_id")[value_column].mean().dropna().to_numpy(float)
    if len(image_means) == 0 or repeats <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled = rng.choice(image_means, size=len(image_means), replace=True)
        samples[index] = sampled.mean()
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def build_correspondence_summary(
    df: pd.DataFrame, bootstrap_repeats: int, seed: int
) -> pd.DataFrame:
    group_specs: List[Tuple[str, str, pd.DataFrame]] = [("all", "all", df)]
    for column in ["geometry_state", "size_group", "offset_bin", "class_label"]:
        for value, subset in df.groupby(column, dropna=False):
            group_specs.append((column, str(value), subset))
    for (state, size), subset in df.groupby(["geometry_state", "size_group"], dropna=False):
        group_specs.append(("state_size", "{}_{}".format(state, size), subset))

    records = []
    for group_type, group_value, subset in group_specs:
        ci_low, ci_high = bootstrap_image_mean_ci(
            subset, "soft_gain_vs_zero_px", bootstrap_repeats, seed
        )
        records.append(
            {
                "group_type": group_type,
                "group_value": group_value,
                "object_count": len(subset),
                "image_count": subset["image_id"].nunique(),
                "mean_zero_epe_px": subset["zero_epe_px"].mean(),
                "mean_argmax_epe_px": subset["argmax_epe_px"].mean(),
                "mean_soft_epe_px": subset["soft_epe_px"].mean(),
                "median_soft_epe_px": subset["soft_epe_px"].median(),
                "mean_argmax_gain_vs_zero_px": subset["argmax_gain_vs_zero_px"].mean(),
                "mean_soft_gain_vs_zero_px": subset["soft_gain_vs_zero_px"].mean(),
                "image_bootstrap_soft_gain_ci95_low": ci_low,
                "image_bootstrap_soft_gain_ci95_high": ci_high,
                "argmax_correct_4px": subset["argmax_strict_correct_4px"].mean(),
                "soft_correct_4px": subset["strict_correct_4px"].mean(),
                "argmax_correct_8px": subset["argmax_usable_correct_8px"].mean(),
                "soft_correct_8px": subset["usable_correct_8px"].mean(),
                "mean_margin": subset["score_margin"].mean(),
                "mean_entropy": subset["normalized_entropy"].mean(),
                "mean_spatial_variance": subset["spatial_variance_cell2"].mean(),
                "mean_cycle_error_cell": subset["cycle_error_cell"].mean(),
                "mutual_match_rate": subset["mutual_match"].mean(),
                "boundary_peak_rate": subset["peak_on_boundary"].mean(),
            }
        )
    return pd.DataFrame(records)


def extract_paths_from_batch(validator: Any, batch_size: int) -> Optional[List[str]]:
    batch = getattr(validator, "batch", None)
    if isinstance(batch, dict):
        paths = batch.get("im_file")
        if paths is None:
            paths = batch.get("im_files")
        if paths is not None:
            if isinstance(paths, (str, Path)):
                paths = [str(paths)]
            else:
                paths = [str(path) for path in paths]
            if len(paths) == batch_size:
                return paths
    return None


def dataset_image_paths(validator: Any) -> List[str]:
    dataloader = getattr(validator, "dataloader", None)
    dataset = getattr(dataloader, "dataset", None)
    for attribute in ["im_files", "img_files", "files"]:
        paths = getattr(dataset, attribute, None)
        if paths is not None:
            return [str(path) for path in paths]
    raise RuntimeError(
        "Could not resolve validation image paths. Expected validator.batch['im_file'] "
        "or validator.dataloader.dataset.im_files."
    )


class P3ProbeCallback:
    def __init__(
        self,
        rgb_capture: TensorCapture,
        ir_capture: TensorCapture,
        state_df: pd.DataFrame,
        stride: int,
        search_radius: int,
        patch_radius: int,
        max_images: Optional[int],
    ):
        self.rgb_capture = rgb_capture
        self.ir_capture = ir_capture
        self.state_groups = {
            str(image_id): group.reset_index(drop=True)
            for image_id, group in state_df.groupby("image_id", sort=False)
        }
        self.stride = stride
        self.search_radius = search_radius
        self.patch_radius = patch_radius
        self.max_images = max_images
        self.cursor = 0
        self.processed_images = 0
        self.seen_images = 0
        self.rows_requested = 0
        self.invalid_query = 0
        self.no_candidate = 0
        self.missing_state_images = 0
        self.feature_shapes: List[Tuple[int, ...]] = []
        self.records: List[Dict[str, Any]] = []

    def __call__(self, validator: Any) -> None:
        torch, _ = require_torch()
        rgb = self.rgb_capture.tensor
        ir = self.ir_capture.tensor
        if rgb is None or ir is None:
            raise RuntimeError("P3 hooks did not capture both RGB and IR features.")
        if rgb.ndim != 4 or ir.ndim != 4:
            raise RuntimeError(
                "Expected BxCxHxW hook outputs, got {} and {}".format(
                    tuple(rgb.shape), tuple(ir.shape)
                )
            )
        if rgb.shape != ir.shape:
            raise RuntimeError(
                "RGB/IR batch feature mismatch: {} vs {}".format(
                    tuple(rgb.shape), tuple(ir.shape)
                )
            )
        batch_size = int(rgb.shape[0])
        if not self.feature_shapes:
            self.feature_shapes.append(tuple(int(value) for value in rgb.shape))

        paths = extract_paths_from_batch(validator, batch_size)
        if paths is None:
            all_paths = dataset_image_paths(validator)
            paths = all_paths[self.cursor : self.cursor + batch_size]
            if len(paths) != batch_size:
                raise RuntimeError(
                    "Dataset path cursor mismatch: expected {}, got {}.".format(
                        batch_size, len(paths)
                    )
                )
        self.cursor += batch_size

        remaining = batch_size
        if self.max_images is not None:
            remaining = max(0, min(batch_size, self.max_images - self.seen_images))
        for batch_index in range(remaining):
            path = paths[batch_index]
            image_id = Path(path).stem
            self.seen_images += 1
            rows = self.state_groups.get(image_id)
            if rows is None or len(rows) == 0:
                self.missing_state_images += 1
                continue
            orig_h, orig_w = image_size(path)
            image_records, image_stats = analyze_image_features(
                rgb[batch_index],
                ir[batch_index],
                rows,
                image_id=image_id,
                image_path=path,
                orig_h=orig_h,
                orig_w=orig_w,
                stride=self.stride,
                search_radius=self.search_radius,
                patch_radius=self.patch_radius,
                keep_scores=True,
            )
            self.records.extend(image_records)
            self.processed_images += 1
            self.rows_requested += image_stats["requested"]
            self.invalid_query += image_stats["invalid_query"]
            self.no_candidate += image_stats["no_candidate"]

        if self.max_images is not None and self.seen_images >= self.max_images:
            raise StopProbeEarly("Reached --max-images={}.".format(self.max_images))


def resolve_model_layers(yolo: Any) -> Any:
    core = getattr(yolo, "model", None)
    layers = getattr(core, "model", None)
    if layers is None:
        raise RuntimeError("Could not resolve YOLO.model.model layer sequence.")
    return layers


def print_layer_check(layers: Any, rgb_layer: int, ir_layer: int) -> None:
    if rgb_layer >= len(layers) or ir_layer >= len(layers):
        raise IndexError(
            "Requested layer indices rgb={}, ir={}, but model has {} layers.".format(
                rgb_layer, ir_layer, len(layers)
            )
        )
    print("[Layer check]")
    print("  RGB layer {}: {}".format(rgb_layer, layers[rgb_layer].__class__.__name__))
    print("  IR  layer {}: {}".format(ir_layer, layers[ir_layer].__class__.__name__))


def run_self_test() -> None:
    torch, _ = require_torch()
    torch.manual_seed(7)
    channels, height, width = 32, 32, 32
    stride = 8
    true_dx, true_dy = 2, -1
    ir = torch.randn(channels, height, width)
    rgb = torch.roll(ir, shifts=(true_dy, true_dx), dims=(1, 2))
    feature_x, feature_y = 16.0, 16.0
    ir_cx = (feature_x + 0.5) * stride
    ir_cy = (feature_y + 0.5) * stride
    rows = pd.DataFrame(
        [
            {
                "image_id": "synthetic",
                "cls": 0,
                "rgb_cx": ir_cx + true_dx * stride,
                "rgb_cy": ir_cy + true_dy * stride,
                "ir_cx": ir_cx,
                "ir_cy": ir_cy,
                "geometry_state": "local_shift",
                "mean_area": 900.0,
            }
        ]
    )
    records, stats = analyze_image_features(
        rgb,
        ir,
        rows,
        image_id="synthetic",
        image_path="synthetic",
        orig_h=height * stride,
        orig_w=width * stride,
        stride=stride,
        search_radius=2,
        patch_radius=1,
        keep_scores=True,
    )
    if stats["requested"] != 1 or len(records) != 1:
        raise AssertionError("Self-test failed to produce one diagnostic record.")
    record = records[0]
    predicted = (record["pred_argmax_dx_cell"], record["pred_argmax_dy_cell"])
    expected = (float(true_dx), float(true_dy))
    if predicted != expected:
        raise AssertionError("Expected {}, got {}.".format(expected, predicted))
    evaluated = attach_soft_metrics(pd.DataFrame(records), 2, stride, tau=0.03)
    if float(evaluated.iloc[0]["argmax_epe_px"]) > 1e-5:
        raise AssertionError("Synthetic argmax EPE is not zero.")
    print("[PASS] P3 candidate direction and argmax recovery")
    print("[PASS] token-wise 3*3 cosine correlation")
    print("[PASS] softmax diagnostics and EPE computation")
    print("All P3 correspondence self-tests passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe frozen RGB/IR P3 correspondence in a TwoStream YOLOv8 model."
    )
    parser.add_argument("--weights", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/obb/base_add/weights/best.pt",
                        help="Add-baseline best.pt")
    parser.add_argument("--data", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/data/dronevehicle_obb.yaml",
                        help="TwoStream dataset YAML")
    parser.add_argument("--state-csv", type=str, default="/storage/jyx4/projects/dataset/DroneVehicle/tool/val_matched_state_index.csv",
                        help="Strict matched offset state CSV")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument(
        "--states",
        type=str,
        default="near_center,local_shift",
        help="Comma-separated geometry states to include; use 'all' to disable filtering.",
    )
    parser.add_argument("--output", type=str, default="runs/p3_correspondence/val_100")
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rgb-layer", type=int, default=11)
    parser.add_argument("--ir-layer", type=int, default=12)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--search-radius", type=int, default=2)
    parser.add_argument("--patch-radius", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument(
        "--tau-grid", type=str, default="0.03,0.05,0.10,0.20,0.30,0.50"
    )
    parser.add_argument("--calibrate-tau", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    for name in ["weights", "data", "state_csv"]:
        value = getattr(args, name.replace("-", "_"), None)
        if not value:
            raise ValueError("--{} is required unless --self-test is used.".format(name))
        if not Path(value).exists():
            raise FileNotFoundError("--{} does not exist: {}".format(name, value))
    if args.tau <= 0:
        raise ValueError("--tau must be positive.")
    if args.stride <= 0 or args.search_radius < 1 or args.patch_radius < 0:
        raise ValueError("Invalid stride/search-radius/patch-radius.")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("--max-images must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.self_test:
        run_self_test()
        return

    torch, _ = require_torch()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Cannot import the local modified ultralytics package. Run this script "
            "from the TwoStream YOLO project root in the training environment."
        ) from exc

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_states = None
    if args.states.strip().lower() != "all":
        requested_states = [item.strip() for item in args.states.split(",") if item.strip()]
    state_df = read_state_csv(args.state_csv, args.split, requested_states)
    print("Loaded {} state rows from {} images.".format(
        len(state_df), state_df["image_id"].nunique()
    ))

    yolo = YOLO(args.weights)
    layers = resolve_model_layers(yolo)
    print_layer_check(layers, args.rgb_layer, args.ir_layer)
    rgb_capture = TensorCapture(layers[args.rgb_layer], "RGB P3")
    ir_capture = TensorCapture(layers[args.ir_layer], "IR P3")
    callback = P3ProbeCallback(
        rgb_capture=rgb_capture,
        ir_capture=ir_capture,
        state_df=state_df,
        stride=args.stride,
        search_radius=args.search_radius,
        patch_radius=args.patch_radius,
        max_images=args.max_images,
    )
    yolo.add_callback("on_val_batch_end", callback)

    stopped_early = False
    try:
        with torch.inference_mode():
            yolo.val(
                data=args.data,
                split=args.split,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                half=False,
                augment=False,
                save=False,
                save_txt=False,
                plots=False,
                verbose=False,
            )
    except StopProbeEarly as exc:
        stopped_early = True
        print("[INFO] {}".format(exc))
    finally:
        rgb_capture.close()
        ir_capture.close()

    if len(callback.records) == 0:
        raise RuntimeError(
            "No diagnostic object was produced. Check image_id mapping, hook layer "
            "indices, and whether validator paths refer to the RGB validation images."
        )

    raw_df = pd.DataFrame(callback.records)
    raw_path = output_dir / "p3_correspondence_raw.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    tau_calibration_path = None
    selected_tau = float(args.tau)
    if args.calibrate_tau:
        selected_tau, tau_table = calibrate_tau(
            raw_df,
            args.search_radius,
            args.stride,
            parse_tau_grid(args.tau_grid),
        )
        tau_calibration_path = output_dir / "tau_calibration.csv"
        tau_table.to_csv(tau_calibration_path, index=False, encoding="utf-8-sig")
        print("Selected tau={:.6g} on the current calibration split.".format(selected_tau))

    diagnostics = attach_soft_metrics(
        raw_df, args.search_radius, args.stride, selected_tau
    )
    diagnostics_path = output_dir / "p3_correspondence_diagnostics.csv"
    diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")

    summary = build_correspondence_summary(diagnostics, args.bootstrap, args.seed)
    summary_path = output_dir / "p3_correspondence_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    confidence_quality, risk_coverage = build_confidence_tables(
        diagnostics, coverage_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    confidence_path = output_dir / "confidence_quality_summary.csv"
    risk_path = output_dir / "risk_coverage.csv"
    confidence_quality.to_csv(confidence_path, index=False, encoding="utf-8-sig")
    risk_coverage.to_csv(risk_path, index=False, encoding="utf-8-sig")

    run_summary = {
        "weights": str(Path(args.weights).resolve()),
        "data": str(Path(args.data).resolve()),
        "state_csv": str(Path(args.state_csv).resolve()),
        "split": args.split,
        "included_geometry_states": requested_states if requested_states is not None else "all",
        "stopped_early": stopped_early,
        "max_images": args.max_images,
        "rgb_layer": args.rgb_layer,
        "ir_layer": args.ir_layer,
        "stride": args.stride,
        "search_radius": args.search_radius,
        "candidate_count": (2 * args.search_radius + 1) ** 2,
        "patch_radius": args.patch_radius,
        "patch_token_count": (2 * args.patch_radius + 1) ** 2,
        "selected_tau": selected_tau,
        "tau_calibrated_on_this_run": bool(args.calibrate_tau),
        "state_csv_rows": int(len(state_df)),
        "state_csv_images": int(state_df["image_id"].nunique()),
        "seen_images": callback.seen_images,
        "processed_images_with_state_rows": callback.processed_images,
        "images_without_state_rows": callback.missing_state_images,
        "rows_requested_in_seen_images": callback.rows_requested,
        "diagnostic_objects": int(len(diagnostics)),
        "invalid_query_objects": callback.invalid_query,
        "objects_without_valid_candidate": callback.no_candidate,
        "first_feature_batch_shape": callback.feature_shapes[0] if callback.feature_shapes else None,
        "warning": (
            "This is an offline correspondence diagnostic using RGB boxes as measuring "
            "instruments. It is not detector mAP and does not authorize RGB-box supervision "
            "in the final method. A tau selected on val must not be reported as held-out evidence."
        ),
        "outputs": {
            "raw": str(raw_path),
            "diagnostics": str(diagnostics_path),
            "summary": str(summary_path),
            "confidence_quality": str(confidence_path),
            "risk_coverage": str(risk_path),
            "tau_calibration": str(tau_calibration_path) if tau_calibration_path else None,
        },
    }
    summary_json_path = output_dir / "run_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)

    print("\nCompleted P3 correspondence probe.")
    print("  Diagnostic objects: {}".format(len(diagnostics)))
    print("  Selected tau: {}".format(selected_tau))
    print("  Output directory: {}".format(output_dir))
    all_summary = summary[(summary["group_type"] == "all") & (summary["group_value"] == "all")]
    if len(all_summary):
        row = all_summary.iloc[0]
        print("  Mean zero-offset EPE: {:.4f}px".format(row["mean_zero_epe_px"]))
        print("  Mean argmax EPE: {:.4f}px".format(row["mean_argmax_epe_px"]))
        print("  Mean softargmax EPE: {:.4f}px".format(row["mean_soft_epe_px"]))
        print(
            "  Soft gain vs zero: {:.4f}px, image-bootstrap 95% CI [{:.4f}, {:.4f}]".format(
                row["mean_soft_gain_vs_zero_px"],
                row["image_bootstrap_soft_gain_ci95_low"],
                row["image_bootstrap_soft_gain_ci95_high"],
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[FATAL] {}".format(exc), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
