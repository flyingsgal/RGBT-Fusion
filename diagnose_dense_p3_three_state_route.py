#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dense P3 three-state routing diagnostic for two-stream YOLOv8.

This is a read-only representation diagnostic. It loads:

  1) the frozen Add-baseline detector checkpoint, and
  2) the learned modality-specific projection checkpoint produced by
     train_p3_common_space_probe.py.

For every valid P3 location it evaluates a 5*5 RGB candidate neighborhood for
an IR-reference 3*3 query patch. The dense score volume is implemented with
shifted feature correlations and average pooling; it does not materialize an
H*W*25*9*C tensor.

Three route states
------------------
  aligned:
      best_noncenter_score - center_score <= margin_threshold

  reliable_shift:
      margin > margin_threshold and normalized_entropy <= entropy_threshold

  uncertain:
      margin > margin_threshold and normalized_entropy > entropy_threshold

The default thresholds (margin=0.04, entropy=0.68) are diagnostic starting
points derived from the prior image-grouped validation analysis. They are not
final paper hyperparameters.

The script uses IR OBB labels only to divide the dense P3 map into IR
foreground, a one-cell context ring, and far background. Matched RGB boxes from
the state CSV are used only to measure target offset EPE. No parameter is
trained and no detector checkpoint is modified.

Required companion files
------------------------
Place these three scripts in the modified TwoStream YOLO repository root:

  probe_p3_rgbt_correspondence.py
  train_p3_common_space_probe.py
  diagnose_dense_p3_three_state_route.py

Typical pilot
-------------
  python diagnose_dense_p3_three_state_route.py \
    --weights /path/to/add/best.pt \
    --projection-weights /path/to/projection_probe.pt \
    --data /path/to/dronevehicle_obb.yaml \
    --state-csv /path/to/val_matched_state_index.csv \
    --ir-label-dir /path/to/IR/labels/val \
    --output runs/dense_p3_route/val100 \
    --device 0 --batch 8 --imgsz 640 --max-images 100

Main outputs
------------
  dense_route_region_summary.csv
  dense_route_image_summary.csv
  dense_route_target_diagnostics.csv
  dense_route_target_summary.csv
  dense_route_threshold_sweep.csv
  dense_route_decision_summary.json
  visualizations/*_route_overlay.png
"""

import argparse
import json
import math
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import probe_p3_rgbt_correspondence as base
    import train_p3_common_space_probe as learned
except ImportError as exc:
    raise RuntimeError(
        "Put diagnose_dense_p3_three_state_route.py, "
        "probe_p3_rgbt_correspondence.py, and train_p3_common_space_probe.py "
        "in the modified TwoStream YOLO repository root."
    ) from exc


ROUTE_INVALID = -1
ROUTE_ALIGNED = 0
ROUTE_RELIABLE = 1
ROUTE_UNCERTAIN = 2
ROUTE_NAMES = {
    ROUTE_INVALID: "invalid",
    ROUTE_ALIGNED: "aligned",
    ROUTE_RELIABLE: "reliable_shift",
    ROUTE_UNCERTAIN: "uncertain",
}


def parse_float_grid(text: str, name: str) -> List[float]:
    values: List[float] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    values = sorted(set(values))
    if not values:
        raise ValueError("{} cannot be empty.".format(name))
    return values


def require_torch():
    return base.require_torch()


def shift_source_to_destination(tensor: Any, dx: int, dy: int):
    """Return out[..., y, x] = tensor[..., y + dy, x + dx], zero elsewhere."""
    torch, _ = require_torch()
    if tensor.ndim < 2:
        raise ValueError("shift tensor must have at least two dimensions.")
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    output = torch.zeros_like(tensor)

    dst_x0 = max(0, -dx)
    dst_x1 = min(width, width - dx)
    dst_y0 = max(0, -dy)
    dst_y1 = min(height, height - dy)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return output
    src_x0, src_x1 = dst_x0 + dx, dst_x1 + dx
    src_y0, src_y1 = dst_y0 + dy, dst_y1 + dy
    output[..., dst_y0:dst_y1, dst_x0:dst_x1] = tensor[
        ..., src_y0:src_y1, src_x0:src_x1
    ]
    return output


def content_mask_from_geometry(
    feature_h: int,
    feature_w: int,
    geom: Dict[str, float],
    stride: int,
    device: Any,
):
    torch, _ = require_torch()
    ys = torch.arange(feature_h, device=device, dtype=torch.float32)
    xs = torch.arange(feature_w, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xmin = geom["pad_x"] / float(stride) - 0.5
    ymin = geom["pad_y"] / float(stride) - 0.5
    xmax = (geom["pad_x"] + geom["resized_w"] - 1.0) / float(stride) - 0.5
    ymax = (geom["pad_y"] + geom["resized_h"] - 1.0) / float(stride) - 0.5
    return (xx >= xmin) & (xx <= xmax) & (yy >= ymin) & (yy <= ymax)


def dense_projected_scores(
    matcher: Any,
    rgb_feature: Any,
    ir_feature: Any,
    content_mask: Any,
    search_radius: int,
    patch_radius: int,
) -> Dict[str, Any]:
    """Compute dense IR-reference candidate scores without patch unfolding."""
    torch, F = require_torch()
    if rgb_feature.ndim != 3 or ir_feature.ndim != 3:
        raise ValueError("Expected CxHxW RGB/IR features.")
    if rgb_feature.shape != ir_feature.shape:
        raise ValueError(
            "RGB/IR feature mismatch: {} vs {}".format(
                tuple(rgb_feature.shape), tuple(ir_feature.shape)
            )
        )
    channels, height, width = [int(value) for value in rgb_feature.shape]
    if channels != int(matcher.input_dim):
        raise ValueError(
            "P3 channel mismatch: feature={} projection={}".format(
                channels, matcher.input_dim
            )
        )

    rgb_hwc = rgb_feature.detach().float().permute(1, 2, 0)
    ir_hwc = ir_feature.detach().float().permute(1, 2, 0)
    rgb_projected = F.normalize(matcher.rgb_proj(rgb_hwc), dim=-1, eps=1e-6)
    ir_projected = F.normalize(matcher.ir_proj(ir_hwc), dim=-1, eps=1e-6)
    rgb_projected = rgb_projected.permute(2, 0, 1).contiguous()
    ir_projected = ir_projected.permute(2, 0, 1).contiguous()

    candidate_list = [
        (dx, dy)
        for dy in range(-search_radius, search_radius + 1)
        for dx in range(-search_radius, search_radius + 1)
    ]
    kernel = 2 * patch_radius + 1
    content_float = content_mask.to(dtype=torch.float32)
    score_maps: List[Any] = []
    valid_maps: List[Any] = []
    for dx, dy in candidate_list:
        shifted_rgb = shift_source_to_destination(rgb_projected, dx, dy)
        shifted_content = shift_source_to_destination(content_float, dx, dy)
        pair_valid = content_float * shifted_content
        point_score = (ir_projected * shifted_rgb).sum(dim=0) * pair_valid
        patch_score = F.avg_pool2d(
            point_score[None, None],
            kernel_size=kernel,
            stride=1,
            padding=patch_radius,
            count_include_pad=True,
        )[0, 0]
        patch_valid_fraction = F.avg_pool2d(
            pair_valid[None, None],
            kernel_size=kernel,
            stride=1,
            padding=patch_radius,
            count_include_pad=True,
        )[0, 0]
        candidate_valid = patch_valid_fraction >= (1.0 - 1e-6)
        score_maps.append(patch_score)
        valid_maps.append(candidate_valid)

    scores = torch.stack(score_maps, dim=0)
    valid = torch.stack(valid_maps, dim=0)
    scores = scores.masked_fill(~valid, float("-inf"))
    candidates = torch.as_tensor(
        candidate_list, device=scores.device, dtype=scores.dtype
    )
    center_index = candidate_list.index((0, 0))
    center_valid = valid[center_index]
    full_search_valid = valid.all(dim=0)
    usable = center_valid & valid.any(dim=0)
    return {
        "scores": scores,
        "valid": valid,
        "candidates": candidates,
        "candidate_list": candidate_list,
        "center_index": center_index,
        "usable": usable,
        "full_search_valid": full_search_valid,
        "height": height,
        "width": width,
    }


def route_from_dense_scores(
    dense: Dict[str, Any],
    tau: float,
    margin_threshold: float,
    entropy_threshold: float,
) -> Dict[str, Any]:
    torch, _ = require_torch()
    scores = dense["scores"]
    valid = dense["valid"]
    candidates = dense["candidates"]
    center_index = int(dense["center_index"])
    usable = dense["usable"]
    candidate_count = int(scores.shape[0])

    safe_scores = scores.clone()
    safe_scores[:, ~usable] = float("-inf")
    maximum = safe_scores.max(dim=0).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exp_scores = torch.where(
        valid,
        torch.exp((safe_scores - maximum[None]) / float(tau)),
        torch.zeros_like(safe_scores),
    )
    denominator = exp_scores.sum(dim=0).clamp_min(1e-12)
    probabilities = exp_scores / denominator[None]

    pred_dx = (probabilities * candidates[:, 0, None, None]).sum(dim=0)
    pred_dy = (probabilities * candidates[:, 1, None, None]).sum(dim=0)
    entropy_raw = -torch.where(
        probabilities > 0,
        probabilities * torch.log(probabilities.clamp_min(1e-12)),
        torch.zeros_like(probabilities),
    ).sum(dim=0)
    valid_count = valid.sum(dim=0)
    entropy_denominator = torch.log(valid_count.clamp_min(2).to(scores.dtype))
    normalized_entropy = entropy_raw / entropy_denominator

    noncenter_indices = [index for index in range(candidate_count) if index != center_index]
    best_noncenter = scores[noncenter_indices].max(dim=0).values
    center_score = scores[center_index]
    route_margin = best_noncenter - center_score

    route = torch.full_like(valid_count, ROUTE_INVALID, dtype=torch.int8)
    aligned = usable & (route_margin <= margin_threshold)
    reliable = (
        usable
        & (route_margin > margin_threshold)
        & (normalized_entropy <= entropy_threshold)
    )
    uncertain = (
        usable
        & (route_margin > margin_threshold)
        & (normalized_entropy > entropy_threshold)
    )
    route[aligned] = ROUTE_ALIGNED
    route[reliable] = ROUTE_RELIABLE
    route[uncertain] = ROUTE_UNCERTAIN

    return {
        "probabilities": probabilities,
        "pred_dx": pred_dx,
        "pred_dy": pred_dy,
        "entropy": normalized_entropy,
        "margin": route_margin,
        "route": route,
        "valid_count": valid_count,
    }


def read_yolo_obb_labels(path: Path, orig_h: int, orig_w: int) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError("Missing IR label file: {}".format(path))
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            pieces = stripped.split()
            try:
                values = [float(value) for value in pieces]
            except ValueError as exc:
                raise ValueError(
                    "Non-numeric IR label at {}:{}".format(path, line_number)
                ) from exc
            if len(values) >= 9:
                class_id = int(values[0])
                polygon = np.asarray(values[1:9], dtype=float).reshape(4, 2)
            elif len(values) >= 5:
                class_id = int(values[0])
                cx, cy, width, height = values[1:5]
                polygon = np.asarray(
                    [
                        [cx - width / 2.0, cy - height / 2.0],
                        [cx + width / 2.0, cy - height / 2.0],
                        [cx + width / 2.0, cy + height / 2.0],
                        [cx - width / 2.0, cy + height / 2.0],
                    ],
                    dtype=float,
                )
            else:
                raise ValueError(
                    "IR label needs class+8 OBB coordinates or class+xywh at {}:{}".format(
                        path, line_number
                    )
                )
            if np.nanmax(np.abs(polygon)) <= 1.5:
                polygon[:, 0] *= float(orig_w)
                polygon[:, 1] *= float(orig_h)
            records.append(
                {
                    "class_id": class_id,
                    "polygon": polygon,
                    "line_number": line_number,
                }
            )
    return records


def points_in_polygon(grid_x: np.ndarray, grid_y: np.ndarray, polygon: np.ndarray):
    inside = np.zeros(grid_x.shape, dtype=bool)
    x_vertices = polygon[:, 0]
    y_vertices = polygon[:, 1]
    previous = len(polygon) - 1
    for current in range(len(polygon)):
        xi, yi = x_vertices[current], y_vertices[current]
        xj, yj = x_vertices[previous], y_vertices[previous]
        crosses = (yi > grid_y) != (yj > grid_y)
        x_intersection = (xj - xi) * (grid_y - yi) / (yj - yi + 1e-12) + xi
        inside ^= crosses & (grid_x < x_intersection)
        previous = current
    return inside


def rasterize_ir_regions(
    labels: Sequence[Dict[str, Any]],
    feature_h: int,
    feature_w: int,
    geom: Dict[str, float],
    stride: int,
    dilation: int,
    device: Any,
) -> Dict[str, Any]:
    torch, F = require_torch()
    ys = np.arange(feature_h, dtype=float)
    xs = np.arange(feature_w, dtype=float)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    original_x = ((xx + 0.5) * stride - geom["pad_x"]) / geom["gain_x"]
    original_y = ((yy + 0.5) * stride - geom["pad_y"]) / geom["gain_y"]
    core = np.zeros((feature_h, feature_w), dtype=bool)
    polygons: List[np.ndarray] = []
    for record in labels:
        polygon = np.asarray(record["polygon"], dtype=float)
        polygons.append(polygon)
        object_mask = points_in_polygon(original_x, original_y, polygon)
        center = polygon.mean(axis=0)
        center_x = int(
            math.floor(
                (center[0] * geom["gain_x"] + geom["pad_x"]) / stride - 0.5 + 0.5
            )
        )
        center_y = int(
            math.floor(
                (center[1] * geom["gain_y"] + geom["pad_y"]) / stride - 0.5 + 0.5
            )
        )
        if 0 <= center_x < feature_w and 0 <= center_y < feature_h:
            object_mask[center_y, center_x] = True
        core |= object_mask

    core_tensor = torch.as_tensor(core, device=device, dtype=torch.bool)
    if dilation > 0:
        kernel = 2 * dilation + 1
        context = F.max_pool2d(
            core_tensor.float()[None, None],
            kernel_size=kernel,
            stride=1,
            padding=dilation,
        )[0, 0].bool()
    else:
        context = core_tensor.clone()
    return {
        "core": core_tensor,
        "context": context,
        "context_ring": context & ~core_tensor,
        "polygons": polygons,
    }


def nearest_feature_cell(
    x: float,
    y: float,
    geom: Dict[str, float],
    stride: int,
) -> Tuple[int, int, float, float]:
    feature_x = (x * geom["gain_x"] + geom["pad_x"]) / stride - 0.5
    feature_y = (y * geom["gain_y"] + geom["pad_y"]) / stride - 0.5
    cell_x = int(math.floor(feature_x + 0.5))
    cell_y = int(math.floor(feature_y + 0.5))
    return cell_x, cell_y, feature_x, feature_y


def summarize_route_mask(route: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    total = int(mask.sum())
    counts = {
        name: int((mask & (route == code)).sum())
        for code, name in [
            (ROUTE_ALIGNED, "aligned"),
            (ROUTE_RELIABLE, "reliable_shift"),
            (ROUTE_UNCERTAIN, "uncertain"),
        ]
    }
    output: Dict[str, Any] = {"valid_cells": total}
    for name, count in counts.items():
        output["{}_count".format(name)] = count
        output["{}_rate".format(name)] = count / total if total else float("nan")
    return output


def target_records_for_image(
    rows: pd.DataFrame,
    route_data: Dict[str, Any],
    dense: Dict[str, Any],
    geom: Dict[str, float],
    stride: int,
    image_id: str,
    image_path: str,
) -> List[Dict[str, Any]]:
    route = route_data["route"].detach().cpu().numpy()
    margin = route_data["margin"].detach().cpu().numpy()
    entropy = route_data["entropy"].detach().cpu().numpy()
    pred_dx = route_data["pred_dx"].detach().cpu().numpy()
    pred_dy = route_data["pred_dy"].detach().cpu().numpy()
    valid_count = route_data["valid_count"].detach().cpu().numpy()
    full_search = dense["full_search_valid"].detach().cpu().numpy()
    height, width = route.shape
    records: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        ir_cx = float(row["ir_cx"])
        ir_cy = float(row["ir_cy"])
        rgb_cx = float(row["rgb_cx"])
        rgb_cy = float(row["rgb_cy"])
        cell_x, cell_y, feature_x, feature_y = nearest_feature_cell(
            ir_cx, ir_cy, geom, stride
        )
        in_feature = 0 <= cell_x < width and 0 <= cell_y < height
        if in_feature:
            route_code = int(route[cell_y, cell_x])
            dx_pred = float(pred_dx[cell_y, cell_x])
            dy_pred = float(pred_dy[cell_y, cell_x])
            route_margin = float(margin[cell_y, cell_x])
            route_entropy = float(entropy[cell_y, cell_x])
            candidate_count = int(valid_count[cell_y, cell_x])
            full = bool(full_search[cell_y, cell_x])
        else:
            route_code = ROUTE_INVALID
            dx_pred = dy_pred = route_margin = route_entropy = float("nan")
            candidate_count = 0
            full = False

        gt_dx = (rgb_cx - ir_cx) * geom["gain_x"] / float(stride)
        gt_dy = (rgb_cy - ir_cy) * geom["gain_y"] / float(stride)
        zero_epe = math.hypot(gt_dx, gt_dy) * stride
        if route_code != ROUTE_INVALID:
            soft_epe = math.hypot(dx_pred - gt_dx, dy_pred - gt_dy) * stride
            movement = math.hypot(dx_pred, dy_pred) * stride
        else:
            soft_epe = movement = float("nan")
        gated_epe = soft_epe if route_code == ROUTE_RELIABLE else zero_epe
        area = base.infer_mean_area(row, geom["gain_x"], geom["gain_y"])
        records.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "class_label": base.row_value(
                    row, "cls", base.row_value(row, "class_id", "unknown")
                ),
                "rgb_source_index": base.row_value(row, "rgb_source_index", -1),
                "ir_source_index": base.row_value(row, "ir_source_index", -1),
                "geometry_state": base.row_value(row, "geometry_state", "unknown"),
                "size_group": base.size_group(area),
                "ir_cx": ir_cx,
                "ir_cy": ir_cy,
                "rgb_cx": rgb_cx,
                "rgb_cy": rgb_cy,
                "feature_x": feature_x,
                "feature_y": feature_y,
                "cell_x": cell_x,
                "cell_y": cell_y,
                "cell_quantization_distance": math.hypot(
                    feature_x - cell_x, feature_y - cell_y
                ),
                "route_state": ROUTE_NAMES[route_code],
                "route_margin": route_margin,
                "normalized_entropy": route_entropy,
                "valid_candidate_count": candidate_count,
                "full_search_valid": full,
                "gt_dx_cell": gt_dx,
                "gt_dy_cell": gt_dy,
                "pred_soft_dx_cell": dx_pred,
                "pred_soft_dy_cell": dy_pred,
                "zero_epe_px": zero_epe,
                "soft_epe_px": soft_epe,
                "gated_epe_px": gated_epe,
                "predicted_movement_px": movement,
            }
        )
    return records


def save_route_overlay(
    image_path: str,
    route: np.ndarray,
    analysis_mask: np.ndarray,
    polygons: Sequence[np.ndarray],
    output_path: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    height, width = route.shape
    resampling = getattr(Image, "Resampling", Image)
    colors = np.zeros((height, width, 3), dtype=np.uint8)
    colors[route == ROUTE_ALIGNED] = (55, 115, 210)
    colors[route == ROUTE_RELIABLE] = (225, 60, 55)
    colors[route == ROUTE_UNCERTAIN] = (240, 170, 45)
    colors[~analysis_mask] = (0, 0, 0)
    route_image = Image.fromarray(colors, mode="RGB").resize(
        image.size, resample=resampling.NEAREST
    )
    alpha = Image.new("L", image.size, color=110)
    invalid = Image.fromarray((~analysis_mask).astype(np.uint8) * 255, mode="L").resize(
        image.size, resample=resampling.NEAREST
    )
    alpha_array = np.asarray(alpha).copy()
    alpha_array[np.asarray(invalid) > 0] = 0
    route_image.putalpha(Image.fromarray(alpha_array, mode="L"))
    overlay = Image.alpha_composite(image.convert("RGBA"), route_image)
    draw = ImageDraw.Draw(overlay)
    for polygon in polygons:
        points = [(float(x), float(y)) for x, y in polygon]
        if points:
            draw.line(points + [points[0]], fill=(40, 230, 90, 255), width=2)
    legend = [
        ("aligned", (55, 115, 210, 255)),
        ("reliable", (225, 60, 55, 255)),
        ("uncertain", (240, 170, 45, 255)),
        ("IR OBB", (40, 230, 90, 255)),
    ]
    x0, y0 = 8, 8
    for index, (label, color) in enumerate(legend):
        y = y0 + 18 * index
        draw.rectangle([x0, y, x0 + 12, y + 12], fill=color)
        draw.text((x0 + 17, y - 1), label, fill=(255, 255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(output_path)


def build_target_summary(targets: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    groupings = [
        ("all", ["geometry_state"]),
        ("state_size", ["geometry_state", "size_group"]),
    ]
    valid = targets[targets["route_state"] != "invalid"].copy()
    for group_type, columns in groupings:
        grouper: Any = columns[0] if len(columns) == 1 else columns
        for keys, group in valid.groupby(grouper, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            label = "/".join(str(value) for value in keys)
            count = len(group)
            reliable = group["route_state"].eq("reliable_shift")
            uncertain = group["route_state"].eq("uncertain")
            aligned = group["route_state"].eq("aligned")
            records.append(
                {
                    "group_type": group_type,
                    "group_value": label,
                    "object_count": count,
                    "image_count": int(group["image_id"].nunique()),
                    "aligned_rate": float(aligned.mean()),
                    "reliable_shift_rate": float(reliable.mean()),
                    "uncertain_rate": float(uncertain.mean()),
                    "mean_zero_epe_px": float(group["zero_epe_px"].mean()),
                    "mean_soft_epe_px": float(group["soft_epe_px"].mean()),
                    "mean_gated_epe_px": float(group["gated_epe_px"].mean()),
                    "mean_gated_gain_vs_zero_px": float(
                        (group["zero_epe_px"] - group["gated_epe_px"]).mean()
                    ),
                    "reliable_object_count": int(reliable.sum()),
                    "reliable_mean_soft_epe_px": float(
                        group.loc[reliable, "soft_epe_px"].mean()
                    ),
                    "reliable_mean_zero_epe_px": float(
                        group.loc[reliable, "zero_epe_px"].mean()
                    ),
                }
            )
    return pd.DataFrame(records)


class DenseRouteCallback:
    def __init__(
        self,
        rgb_capture: Any,
        ir_capture: Any,
        matcher: Any,
        state_df: pd.DataFrame,
        args: argparse.Namespace,
    ):
        self.rgb_capture = rgb_capture
        self.ir_capture = ir_capture
        self.matcher = matcher
        self.args = args
        self.groups = {
            str(image_id): group.reset_index(drop=True)
            for image_id, group in state_df.groupby("image_id", sort=False)
        }
        self.cursor = 0
        self.seen_images = 0
        self.processed_images = 0
        self.images_with_targets = 0
        self.missing_label_files = 0
        self.label_objects = 0
        self.target_records: List[Dict[str, Any]] = []
        self.image_records: List[Dict[str, Any]] = []
        self.margin_grid = parse_float_grid(args.margin_grid, "--margin-grid")
        self.entropy_grid = parse_float_grid(args.entropy_grid, "--entropy-grid")
        self.sweep_dense: Dict[Tuple[float, float, str], List[int]] = defaultdict(
            lambda: [0, 0]
        )
        self.matcher_ready = False

    def __call__(self, validator: Any) -> None:
        torch, _ = require_torch()
        rgb = self.rgb_capture.tensor
        ir = self.ir_capture.tensor
        if rgb is None or ir is None or rgb.ndim != 4 or ir.ndim != 4:
            raise RuntimeError("Dense callback did not receive BxCxHxW P3 features.")
        if rgb.shape != ir.shape:
            raise RuntimeError("RGB/IR P3 feature shapes differ.")
        if not self.matcher_ready:
            if int(rgb.shape[1]) != int(self.matcher.input_dim):
                raise RuntimeError(
                    "P3 channels {} do not match projection input {}.".format(
                        rgb.shape[1], self.matcher.input_dim
                    )
                )
            self.matcher.to(rgb.device).eval()
            self.matcher_ready = True

        batch_size = int(rgb.shape[0])
        paths = base.extract_paths_from_batch(validator, batch_size)
        if paths is None:
            all_paths = base.dataset_image_paths(validator)
            paths = all_paths[self.cursor : self.cursor + batch_size]
        if len(paths) != batch_size:
            raise RuntimeError("Dense image path cursor mismatch.")
        self.cursor += batch_size
        remaining = batch_size
        if self.args.max_images is not None:
            remaining = max(
                0, min(batch_size, self.args.max_images - self.seen_images)
            )

        with torch.no_grad():
            for batch_index in range(remaining):
                image_path = paths[batch_index]
                image_id = Path(image_path).stem
                self.seen_images += 1
                orig_h, orig_w = base.image_size(image_path)
                feature_h, feature_w = int(rgb.shape[-2]), int(rgb.shape[-1])
                input_h = feature_h * self.args.stride
                input_w = feature_w * self.args.stride
                geom = base.letterbox_geometry(orig_h, orig_w, input_h, input_w)
                content = content_mask_from_geometry(
                    feature_h,
                    feature_w,
                    geom,
                    self.args.stride,
                    rgb.device,
                )
                dense = dense_projected_scores(
                    self.matcher,
                    rgb[batch_index],
                    ir[batch_index],
                    content,
                    self.args.search_radius,
                    self.args.patch_radius,
                )
                routed = route_from_dense_scores(
                    dense,
                    self.args.tau,
                    self.args.margin_threshold,
                    self.args.entropy_threshold,
                )

                label_path = Path(self.args.ir_label_dir) / (image_id + ".txt")
                try:
                    labels = read_yolo_obb_labels(label_path, orig_h, orig_w)
                except FileNotFoundError:
                    self.missing_label_files += 1
                    if not self.args.allow_missing_labels:
                        raise
                    labels = []
                self.label_objects += len(labels)
                regions = rasterize_ir_regions(
                    labels,
                    feature_h,
                    feature_w,
                    geom,
                    self.args.stride,
                    self.args.foreground_dilation,
                    rgb.device,
                )
                main_valid = (
                    dense["usable"]
                    if self.args.include_partial_search
                    else dense["full_search_valid"]
                )
                region_masks = {
                    "ir_foreground_core": main_valid & regions["core"],
                    "ir_context_ring": main_valid & regions["context_ring"],
                    "background_far": main_valid & ~regions["context"],
                    "all_valid": main_valid,
                }
                route_np = routed["route"].detach().cpu().numpy()
                for region_name, mask in region_masks.items():
                    row = {
                        "image_id": image_id,
                        "image_path": image_path,
                        "region": region_name,
                        "ir_label_objects": len(labels),
                        "feature_h": feature_h,
                        "feature_w": feature_w,
                    }
                    row.update(
                        summarize_route_mask(
                            route_np, mask.detach().cpu().numpy()
                        )
                    )
                    self.image_records.append(row)

                margin_grid_tensor = torch.as_tensor(
                    self.margin_grid,
                    device=rgb.device,
                    dtype=routed["margin"].dtype,
                )[:, None, None, None]
                entropy_grid_tensor = torch.as_tensor(
                    self.entropy_grid,
                    device=rgb.device,
                    dtype=routed["entropy"].dtype,
                )[None, :, None, None]
                reliable_grid = (
                    (routed["margin"][None, None] > margin_grid_tensor)
                    & (routed["entropy"][None, None] <= entropy_grid_tensor)
                )
                for region_name, mask in region_masks.items():
                    total = int(mask.sum().item())
                    selected_grid = (
                        reliable_grid & mask[None, None]
                    ).sum(dim=(-2, -1)).detach().cpu().numpy()
                    for margin_index, margin_value in enumerate(self.margin_grid):
                        for entropy_index, entropy_value in enumerate(
                            self.entropy_grid
                        ):
                            accumulator = self.sweep_dense[
                                (margin_value, entropy_value, region_name)
                            ]
                            accumulator[0] += total
                            accumulator[1] += int(
                                selected_grid[margin_index, entropy_index]
                            )

                rows = self.groups.get(image_id)
                if rows is not None and len(rows):
                    self.images_with_targets += 1
                    self.target_records.extend(
                        target_records_for_image(
                            rows,
                            routed,
                            dense,
                            geom,
                            self.args.stride,
                            image_id,
                            image_path,
                        )
                    )

                if self.processed_images < self.args.visualize_images:
                    save_route_overlay(
                        image_path,
                        route_np,
                        main_valid.detach().cpu().numpy(),
                        regions["polygons"],
                        Path(self.args.output)
                        / "visualizations"
                        / (image_id + "_route_overlay.png"),
                    )
                self.processed_images += 1

        if self.args.max_images is not None and self.seen_images >= self.args.max_images:
            raise base.StopProbeEarly(
                "Reached --max-images={}.".format(self.args.max_images)
            )


def aggregate_region_summary(images: pd.DataFrame) -> pd.DataFrame:
    records = []
    for region, group in images.groupby("region", sort=False):
        total = int(group["valid_cells"].sum())
        row: Dict[str, Any] = {
            "region": region,
            "image_count": int(group["image_id"].nunique()),
            "valid_cells": total,
        }
        for state in ["aligned", "reliable_shift", "uncertain"]:
            count = int(group["{}_count".format(state)].sum())
            row["{}_count".format(state)] = count
            row["{}_rate".format(state)] = count / total if total else float("nan")
        records.append(row)
    return pd.DataFrame(records)


def build_threshold_sweep(
    callback: DenseRouteCallback,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    valid_targets = targets[targets["route_state"] != "invalid"].copy()
    records: List[Dict[str, Any]] = []
    for margin_value in callback.margin_grid:
        for entropy_value in callback.entropy_grid:
            reliable = (
                (valid_targets["route_margin"] > margin_value)
                & (valid_targets["normalized_entropy"] <= entropy_value)
            )
            gated = np.where(
                reliable,
                valid_targets["soft_epe_px"].to_numpy(float),
                valid_targets["zero_epe_px"].to_numpy(float),
            )
            row: Dict[str, Any] = {
                "margin_threshold": margin_value,
                "entropy_threshold": entropy_value,
                "target_count": len(valid_targets),
                "target_reliable_shift_rate": float(reliable.mean()),
                "target_mean_zero_epe_px": float(
                    valid_targets["zero_epe_px"].mean()
                ),
                "target_mean_gated_epe_px": float(np.mean(gated)),
                "target_mean_gated_gain_vs_zero_px": float(
                    valid_targets["zero_epe_px"].mean() - np.mean(gated)
                ),
            }
            for state in ["near_center", "local_shift"]:
                state_mask = valid_targets["geometry_state"].eq(state).to_numpy()
                state_reliable = reliable.to_numpy() & state_mask
                state_count = int(state_mask.sum())
                row["{}_count".format(state)] = state_count
                row["{}_reliable_shift_rate".format(state)] = (
                    float(state_reliable.sum() / state_count)
                    if state_count
                    else float("nan")
                )
                if state_count:
                    state_gated = np.where(
                        state_reliable[state_mask],
                        valid_targets.loc[state_mask, "soft_epe_px"].to_numpy(float),
                        valid_targets.loc[state_mask, "zero_epe_px"].to_numpy(float),
                    )
                    row["{}_mean_gated_epe_px".format(state)] = float(
                        state_gated.mean()
                    )
                else:
                    row["{}_mean_gated_epe_px".format(state)] = float("nan")

            dense_rates: Dict[str, float] = {}
            for region in [
                "ir_foreground_core",
                "ir_context_ring",
                "background_far",
                "all_valid",
            ]:
                total, selected = callback.sweep_dense.get(
                    (margin_value, entropy_value, region), [0, 0]
                )
                rate = selected / total if total else float("nan")
                dense_rates[region] = rate
                row["{}_cells".format(region)] = total
                row["{}_reliable_shift_rate".format(region)] = rate
            background_rate = dense_rates.get("background_far", float("nan"))
            foreground_rate = dense_rates.get("ir_foreground_core", float("nan"))
            row["foreground_to_background_reliable_enrichment"] = (
                foreground_rate / background_rate
                if np.isfinite(foreground_rate)
                and np.isfinite(background_rate)
                and background_rate > 0
                else float("nan")
            )
            records.append(row)
    return pd.DataFrame(records)


def row_for_group(summary: pd.DataFrame, value: str) -> Optional[pd.Series]:
    subset = summary[
        (summary["group_type"] == "all") & (summary["group_value"] == value)
    ]
    return subset.iloc[0] if len(subset) else None


def row_for_region(summary: pd.DataFrame, region: str) -> Optional[pd.Series]:
    subset = summary[summary["region"] == region]
    return subset.iloc[0] if len(subset) else None


def finite_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def decision_summary(
    callback: DenseRouteCallback,
    regions: pd.DataFrame,
    targets: pd.DataFrame,
    target_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    near = row_for_group(target_summary, "near_center")
    local = row_for_group(target_summary, "local_shift")
    foreground = row_for_region(regions, "ir_foreground_core")
    background = row_for_region(regions, "background_far")
    foreground_rate = (
        finite_or_none(foreground["reliable_shift_rate"])
        if foreground is not None
        else None
    )
    background_rate = (
        finite_or_none(background["reliable_shift_rate"])
        if background is not None
        else None
    )
    enrichment = None
    if (
        foreground_rate is not None
        and background_rate is not None
        and background_rate > 0
    ):
        enrichment = foreground_rate / background_rate
    return {
        "experiment": "dense_p3_three_state_route_diagnostic",
        "seen_images": callback.seen_images,
        "processed_images": callback.processed_images,
        "images_with_matched_targets": callback.images_with_targets,
        "ir_label_objects": callback.label_objects,
        "missing_ir_label_files": callback.missing_label_files,
        "matched_target_records": len(targets),
        "thresholds": {
            "margin": args.margin_threshold,
            "normalized_entropy": args.entropy_threshold,
            "softmax_tau": args.tau,
        },
        "dense_background_question": {
            "ir_foreground_reliable_shift_rate": foreground_rate,
            "background_far_reliable_shift_rate": background_rate,
            "foreground_to_background_reliable_enrichment": enrichment,
        },
        "matched_target_question": {
            "near_center_reliable_shift_rate": (
                finite_or_none(near["reliable_shift_rate"]) if near is not None else None
            ),
            "near_center_mean_gated_epe_px": (
                finite_or_none(near["mean_gated_epe_px"]) if near is not None else None
            ),
            "local_shift_reliable_shift_rate": (
                finite_or_none(local["reliable_shift_rate"])
                if local is not None
                else None
            ),
            "local_shift_reliable_mean_soft_epe_px": (
                finite_or_none(local["reliable_mean_soft_epe_px"])
                if local is not None
                else None
            ),
            "local_shift_mean_zero_epe_px": (
                finite_or_none(local["mean_zero_epe_px"])
                if local is not None
                else None
            ),
            "local_shift_mean_gated_epe_px": (
                finite_or_none(local["mean_gated_epe_px"])
                if local is not None
                else None
            ),
            "local_shift_mean_gated_gain_vs_zero_px": (
                finite_or_none(local["mean_gated_gain_vs_zero_px"])
                if local is not None
                else None
            ),
        },
        "configuration": {
            "weights": str(Path(args.weights).resolve()),
            "projection_weights": str(Path(args.projection_weights).resolve()),
            "data": str(Path(args.data).resolve()),
            "state_csv": str(Path(args.state_csv).resolve()),
            "ir_label_dir": str(Path(args.ir_label_dir).resolve()),
            "stride": args.stride,
            "search_radius": args.search_radius,
            "patch_radius": args.patch_radius,
            "foreground_dilation": args.foreground_dilation,
            "main_dense_mask": (
                "usable_partial_search"
                if args.include_partial_search
                else "full_5x5_search_valid"
            ),
            "max_images": args.max_images,
        },
        "interpretation": {
            "reliable_shift": "Apply explicit RGB-to-IR sampling only here.",
            "aligned": "Keep the center sample; zero-offset protection.",
            "uncertain": (
                "Do not force an offset. A later ablation must compare center, "
                "bounded neighborhood aggregation, and suppressed RGB injection."
            ),
            "warning": (
                "The projection was trained with RGB-box correspondence supervision. "
                "This remains an analysis upper bound, not a deployable final method."
            ),
        },
    }


def run_self_test() -> None:
    torch, _ = require_torch()
    torch.manual_seed(7)
    channels = 16
    height = width = 20
    true_dx, true_dy = 2, -1
    matcher = learned.build_matcher(channels, channels)
    with torch.no_grad():
        matcher.rgb_proj.weight.copy_(torch.eye(channels))
        matcher.ir_proj.weight.copy_(torch.eye(channels))
        matcher.rgb_proj.bias.zero_()
        matcher.ir_proj.bias.zero_()
    ir = torch.randn(channels, height, width)
    rgb = torch.roll(ir, shifts=(true_dy, true_dx), dims=(1, 2))
    content = torch.ones(height, width, dtype=torch.bool)
    dense = dense_projected_scores(matcher, rgb, ir, content, 2, 1)
    routed = route_from_dense_scores(dense, 0.1, 0.04, 0.68)
    y, x = 10, 10
    rows = pd.DataFrame(
        [
            {
                "ir_cx": (x + 0.5) * 8,
                "ir_cy": (y + 0.5) * 8,
                "rgb_cx": (x + 0.5) * 8,
                "rgb_cy": (y + 0.5) * 8,
                "geometry_state": "near_center",
            }
        ]
    )
    patch_item = learned.extract_patch_batch(rgb, ir, rows, 160, 160, 8, 2, 1)
    if patch_item is None:
        raise AssertionError("Integer-center object patch extraction failed.")
    patch_scores = matcher.forward_scores(
        patch_item["ir_patch"], patch_item["rgb_patches"]
    )[0]
    dense_scores = dense["scores"][:, y, x]
    if not torch.allclose(patch_scores, dense_scores, atol=2e-5, rtol=2e-5):
        difference = float((patch_scores - dense_scores).abs().max().item())
        raise AssertionError(
            "Dense scores do not match object-patch scores; max diff={}".format(
                difference
            )
        )
    predicted = np.asarray(
        [
            float(routed["pred_dx"][y, x].item()),
            float(routed["pred_dy"][y, x].item()),
        ]
    )
    if np.linalg.norm(predicted - np.asarray([true_dx, true_dy])) > 0.35:
        raise AssertionError(
            "Dense shift sign/recovery failed: predicted={} expected={}".format(
                predicted.tolist(), [true_dx, true_dy]
            )
        )
    if int(routed["route"][y, x].item()) != ROUTE_RELIABLE:
        raise AssertionError("Synthetic exact shift was not routed as reliable_shift.")

    polygon = np.asarray([[40, 40], [80, 40], [80, 80], [40, 80]], dtype=float)
    geom = {
        "gain_x": 1.0,
        "gain_y": 1.0,
        "pad_x": 0.0,
        "pad_y": 0.0,
        "resized_w": 160.0,
        "resized_h": 160.0,
    }
    regions = rasterize_ir_regions(
        [{"class_id": 0, "polygon": polygon, "line_number": 1}],
        20,
        20,
        geom,
        8,
        1,
        torch.device("cpu"),
    )
    if not bool(regions["core"][7, 7].item()):
        raise AssertionError("IR OBB foreground rasterization failed.")
    if int(regions["context"].sum()) <= int(regions["core"].sum()):
        raise AssertionError("Foreground dilation did not add a context ring.")
    print("[PASS] dense scores equal the original integer-center 3x3/5x5 probe")
    print("[PASS] dense projected correlation sign and soft offset")
    print("[PASS] reliable-shift three-state route")
    print("[PASS] IR OBB foreground/context rasterization")
    print("All dense three-state diagnostic self-tests passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense P3 aligned/reliable/uncertain routing diagnostic."
    )
    parser.add_argument("--weights", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/obb/base_add/weights/best.pt",
                        help="Add-baseline best.pt")
    parser.add_argument(
        "--projection-weights",
        type=str,
        default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/p3_common_space_probe/first_run/projection_probe.pt",
        help="Saved projection_probe.pt; required with --eval-only.",
    )
    parser.add_argument("--data", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/data/dronevehicle_obb.yaml",
                        help="TwoStream dataset YAML")
    parser.add_argument("--train-state-csv", type=str)
    parser.add_argument("-state-csv", type=str, default="/storage/jyx4/projects/dataset/DroneVehicle/tool/val_matched_state_index.csv")
    parser.add_argument("--ir-label-dir", type=str,default="/storage/jyx4/projects/dataset/DroneVehicle/OBBCrop/labels/val")
    parser.add_argument("--output", type=str, default="runs/dense_p3_route_full")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rgb-layer", type=int, default=11)
    parser.add_argument("--ir-layer", type=int, default=12)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--search-radius", type=int, default=2)
    parser.add_argument("--patch-radius", type=int, default=1)
    parser.add_argument("--tau", type=float, default=0.10)
    parser.add_argument("--margin-threshold", type=float, default=0.04)
    parser.add_argument("--entropy-threshold", type=float, default=0.68)
    parser.add_argument(
        "--margin-grid",
        type=str,
        default="0.10",#0.02,0.03,0.04,0.05,0.06,0.08,0.10
        help="Threshold sweep; does not change the primary route overlay.",
    )
    parser.add_argument(
        "--entropy-grid",
        type=str,
        default="0.68",#0.55,0.60,0.65,0.68,0.70,0.75
        help="Threshold sweep; does not change the primary route overlay.",
    )
    parser.add_argument("--foreground-dilation", type=int, default=1)
    parser.add_argument("--include-partial-search", action="store_true")
    parser.add_argument("--allow-missing-labels", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--visualize-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    required = [
        "weights",
        "projection_weights",
        "data",
        "state_csv",
        "ir_label_dir",
    ]
    for name in required:
        value = getattr(args, name)
        if not value:
            raise ValueError("--{} is required.".format(name.replace("_", "-")))
        if not Path(value).exists():
            raise FileNotFoundError("Missing {}: {}".format(name, value))
    if args.tau <= 0:
        raise ValueError("--tau must be positive.")
    if args.stride <= 0 or args.search_radius < 1 or args.patch_radius < 0:
        raise ValueError("Invalid stride/search-radius/patch-radius.")
    if not 0.0 <= args.entropy_threshold <= 1.0:
        raise ValueError("--entropy-threshold must be between 0 and 1.")
    parse_float_grid(args.margin_grid, "--margin-grid")
    entropy_grid = parse_float_grid(args.entropy_grid, "--entropy-grid")
    if any(value < 0.0 or value > 1.0 for value in entropy_grid):
        raise ValueError("All --entropy-grid values must be between 0 and 1.")
    if args.foreground_dilation < 0:
        raise ValueError("--foreground-dilation cannot be negative.")
    if args.max_images is not None and args.max_images <= 0:
        args.max_images = None
    if args.visualize_images < 0:
        raise ValueError("--visualize-images cannot be negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.self_test:
        run_self_test()
        return
    torch, _ = require_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Run this script from the modified TwoStream YOLO repository root."
        ) from exc

    checkpoint = torch.load(args.projection_weights, map_location="cpu")
    required_checkpoint = {"matcher", "input_dim", "embed_dim"}
    missing = sorted(required_checkpoint.difference(checkpoint.keys()))
    if missing:
        raise ValueError("Projection checkpoint is missing keys: {}".format(missing))
    checkpoint_args = checkpoint.get("args", {})
    for name in [
        "rgb_layer",
        "ir_layer",
        "stride",
        "search_radius",
        "patch_radius",
        "imgsz",
    ]:
        if name in checkpoint_args and getattr(args, name) != checkpoint_args[name]:
            raise ValueError(
                "{} differs from projection training: command={} checkpoint={}".format(
                    name, getattr(args, name), checkpoint_args[name]
                )
            )

    state_df = base.read_state_csv(
        args.state_csv, args.split, ["near_center", "local_shift"]
    )
    print("State rows: {}".format(state_df["geometry_state"].value_counts().to_dict()))
    print("IR label directory: {}".format(Path(args.ir_label_dir).resolve()))
    print(
        "Route thresholds: margin>{:.4f}, entropy<={:.4f}".format(
            args.margin_threshold, args.entropy_threshold
        )
    )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    matcher = learned.build_matcher(
        int(checkpoint["input_dim"]), int(checkpoint["embed_dim"])
    )
    matcher.load_state_dict(checkpoint["matcher"], strict=True)
    matcher.eval()

    yolo = YOLO(args.weights)
    layers = base.resolve_model_layers(yolo)
    base.print_layer_check(layers, args.rgb_layer, args.ir_layer)
    rgb_capture = base.TensorCapture(layers[args.rgb_layer], "RGB P3")
    ir_capture = base.TensorCapture(layers[args.ir_layer], "IR P3")
    callback = DenseRouteCallback(
        rgb_capture, ir_capture, matcher, state_df, args
    )
    try:
        stopped = learned.run_validator(yolo, callback, args, args.split)
    finally:
        rgb_capture.close()
        ir_capture.close()

    if not callback.image_records:
        raise RuntimeError("No dense route record was produced.")
    image_summary = pd.DataFrame(callback.image_records)
    region_summary = aggregate_region_summary(image_summary)
    targets = pd.DataFrame(callback.target_records)
    if targets.empty:
        raise RuntimeError("No matched target route record was produced.")
    target_summary = build_target_summary(targets)
    threshold_sweep = build_threshold_sweep(callback, targets)
    decision = decision_summary(
        callback, region_summary, targets, target_summary, args
    )
    decision["stopped_early"] = bool(stopped)

    image_summary.to_csv(
        output / "dense_route_image_summary.csv", index=False, encoding="utf-8-sig"
    )
    region_summary.to_csv(
        output / "dense_route_region_summary.csv", index=False, encoding="utf-8-sig"
    )
    targets.to_csv(
        output / "dense_route_target_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    target_summary.to_csv(
        output / "dense_route_target_summary.csv", index=False, encoding="utf-8-sig"
    )
    threshold_sweep.to_csv(
        output / "dense_route_threshold_sweep.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with (output / "dense_route_decision_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)

    print("\nDense three-state route diagnostic completed.")
    print("  Output: {}".format(output))
    print("  Seen images: {}".format(callback.seen_images))
    print("  Matched target records: {}".format(len(targets)))
    print("  Region summary:")
    print(region_summary.to_string(index=False))
    print("  Target summary:")
    print(target_summary[target_summary["group_type"] == "all"].to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[FATAL] {}".format(exc), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
