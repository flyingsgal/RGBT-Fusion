#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Learned P3 common-space upper-bound probe for two-stream YOLOv8.

This script answers one diagnostic question:

    Does the frozen Add-baseline P3 contain recoverable object-level RGB/IR
    correspondence information that raw cosine cannot expose?

It freezes the detector and trains only two modality-specific linear 1*1
projections. RGB boxes are used to construct a Gaussian target over the local
5*5 displacement candidates. Therefore this is an annotation-assisted analysis
upper bound, NOT a final deployable method and NOT a paper contribution by
itself.

Dependency
----------
Place this file and `probe_p3_rgbt_correspondence.py` in the root of the
modified TwoStream YOLO repository.

Expected layer layout
---------------------
    RGB P3: model.model[11]
    IR  P3: model.model[12]
    P3 stride: 8

Required CSVs
-------------
`--val-state-csv` can use the prior `val_matched_state_index.csv`.
`--train-state-csv` can directly use the large `matched_offsets_postcrop.csv`;
geometry states are then derived from `dist`. Both files need at least:

    image_id, split, rgb_cx, rgb_cy, ir_cx, ir_cy

Only near_center and local_shift are used by default.

Suggested workflow
------------------
1) Self-test:

    python train_p3_common_space_probe.py --self-test

2) Short upper-bound experiment:

    python train_p3_common_space_probe.py \
      --weights /path/to/add/best.pt \
      --data /path/to/dronevehicle_obb.yaml \
      --train-state-csv /path/to/matched_offsets_postcrop.csv \
      --val-state-csv /path/to/val_matched_state_index.csv \
      --output runs/p3_common_space_probe/first_run \
      --device 0 --batch 8 --imgsz 640 \
      --epochs 5 --max-train-images 1000 --max-val-images 100

Main decision outputs
---------------------
    learned_p3_diagnostics.csv
    learned_p3_summary.csv
    learned_confidence_quality.csv
    learned_risk_coverage.csv
    learned_probe_decision_summary.json
    projection_probe.pt
"""

import argparse
import hashlib
import json
import math
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import probe_p3_rgbt_correspondence as base
except ImportError as exc:
    raise RuntimeError(
        "Missing probe_p3_rgbt_correspondence.py. Put both scripts in the "
        "TwoStream YOLO project root."
    ) from exc


def require_torch():
    return base.require_torch()


def stable_rng(seed: int, epoch: int, image_id: str) -> np.random.Generator:
    text = "{}:{}:{}".format(seed, epoch, image_id).encode("utf-8")
    number = int.from_bytes(hashlib.sha1(text).digest()[:8], byteorder="little")
    return np.random.default_rng(number)


def read_probe_state_csv(path: str, split: str) -> pd.DataFrame:
    """
    Read a prepared state index or the large matched_offsets_postcrop.csv.

    If geometry_state is absent, derive it from post-crop center distance:
      near_center: dist <= 2 px
      local_shift: 4 <= dist <= 16 px
    Boundary 2-4 px and shifts above 16 px are excluded.
    """
    header = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
    available = set(header.columns)
    required = {"image_id", "rgb_cx", "rgb_cy", "ir_cx", "ir_cy"}
    missing = sorted(required.difference(available))
    if missing:
        raise ValueError("Offset CSV is missing required columns: {}".format(missing))
    desired = [
        "split",
        "image_id",
        "cls",
        "class_id",
        "rgb_source_index",
        "ir_source_index",
        "rgb_cx",
        "rgb_cy",
        "ir_cx",
        "ir_cy",
        "dist",
        "rgb_area",
        "ir_area",
        "mean_area",
        "rgb_visible_ratio",
        "ir_visible_ratio",
        "rgb_was_clipped",
        "ir_was_clipped",
        "rgb_valid_candidate_count",
        "ir_valid_candidate_count",
        "assignment_cost",
        "geometry_state",
    ]
    usecols = [column for column in desired if column in available]
    chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=usecols,
        dtype={"image_id": str, "split": str},
        chunksize=200000,
        low_memory=False,
    ):
        chunk["image_id"] = chunk["image_id"].astype(str).str.strip()
        if "split" in chunk.columns:
            chunk = chunk[
                chunk["split"].astype(str).str.lower().eq(split.lower())
            ].copy()
        if len(chunk) == 0:
            continue
        if "geometry_state" not in chunk.columns:
            if "dist" not in chunk.columns:
                raise ValueError(
                    "geometry_state is absent and cannot be derived because dist is absent."
                )
            distance = pd.to_numeric(chunk["dist"], errors="coerce").to_numpy(float)
            state = np.full(len(chunk), "excluded", dtype=object)
            state[distance <= 2.0] = "near_center"
            local = (distance >= 4.0) & (distance <= 16.0)
            state[local] = "local_shift"
            chunk["geometry_state"] = state
        # Match the strict val index: retain only fully visible, unclipped,
        # one-to-one candidates. On the supplied val table this reproduces
        # exactly 13,292 rows (10,191 near + 3,101 local).
        if "rgb_visible_ratio" in chunk.columns:
            chunk = chunk[pd.to_numeric(chunk["rgb_visible_ratio"], errors="coerce") >= 0.999999]
        if "ir_visible_ratio" in chunk.columns:
            chunk = chunk[pd.to_numeric(chunk["ir_visible_ratio"], errors="coerce") >= 0.999999]
        if "rgb_was_clipped" in chunk.columns:
            chunk = chunk[pd.to_numeric(chunk["rgb_was_clipped"], errors="coerce").eq(0)]
        if "ir_was_clipped" in chunk.columns:
            chunk = chunk[pd.to_numeric(chunk["ir_was_clipped"], errors="coerce").eq(0)]
        if "rgb_valid_candidate_count" in chunk.columns:
            chunk = chunk[
                pd.to_numeric(chunk["rgb_valid_candidate_count"], errors="coerce").eq(1)
            ]
        if "ir_valid_candidate_count" in chunk.columns:
            chunk = chunk[
                pd.to_numeric(chunk["ir_valid_candidate_count"], errors="coerce").eq(1)
            ]
        chunk = chunk[
            chunk["geometry_state"].astype(str).isin(["near_center", "local_shift"])
        ].copy()
        if len(chunk):
            chunks.append(chunk)
    if not chunks:
        raise ValueError(
            "No near_center/local_shift rows found for split={}.".format(split)
        )
    return pd.concat(chunks, ignore_index=True)


def balanced_object_sample(
    rows: pd.DataFrame,
    limit: Optional[int],
    seed: int,
    epoch: int,
    image_id: str,
) -> pd.DataFrame:
    if limit is None or len(rows) <= limit:
        return rows.reset_index(drop=True)
    rng = stable_rng(seed, epoch, image_id)
    groups = []
    states = ["near_center", "local_shift"]
    quota = max(1, limit // len(states))
    used = set()
    for state in states:
        indices = rows.index[rows["geometry_state"].eq(state)].to_numpy()
        if len(indices):
            chosen = rng.choice(indices, size=min(quota, len(indices)), replace=False)
            groups.extend(chosen.tolist())
            used.update(chosen.tolist())
    remaining = limit - len(groups)
    if remaining > 0:
        pool = np.asarray([index for index in rows.index if index not in used])
        if len(pool):
            chosen = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            groups.extend(chosen.tolist())
    return rows.loc[groups].reset_index(drop=True)


def build_matcher(input_dim: int, embed_dim: int):
    torch, F = require_torch()

    class ProjectionMatcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_dim = int(input_dim)
            self.embed_dim = int(embed_dim)
            self.rgb_proj = torch.nn.Linear(input_dim, embed_dim, bias=True)
            self.ir_proj = torch.nn.Linear(input_dim, embed_dim, bias=True)
            torch.nn.init.orthogonal_(self.rgb_proj.weight)
            torch.nn.init.orthogonal_(self.ir_proj.weight)
            torch.nn.init.zeros_(self.rgb_proj.bias)
            torch.nn.init.zeros_(self.ir_proj.bias)

        def forward_scores(self, ir_patch: Any, rgb_patches: Any):
            # ir_patch: N,P,C; rgb_patches: N,K,P,C
            q = F.normalize(self.ir_proj(ir_patch), dim=-1, eps=1e-6)
            k = F.normalize(self.rgb_proj(rgb_patches), dim=-1, eps=1e-6)
            return (q[:, None, :, :] * k).sum(dim=-1).mean(dim=-1)

        def reverse_scores(self, rgb_patch: Any, ir_patches: Any):
            # rgb_patch: N,P,C; ir_patches: N,K,P,C
            q = F.normalize(self.rgb_proj(rgb_patch), dim=-1, eps=1e-6)
            k = F.normalize(self.ir_proj(ir_patches), dim=-1, eps=1e-6)
            return (q[:, None, :, :] * k).sum(dim=-1).mean(dim=-1)

    return ProjectionMatcher()


def candidate_tensors(
    device: Any, dtype: Any, search_radius: int, patch_radius: int
) -> Tuple[Any, Any, List[Tuple[int, int]]]:
    torch, _ = require_torch()
    candidate_list = [
        (dx, dy)
        for dy in range(-search_radius, search_radius + 1)
        for dx in range(-search_radius, search_radius + 1)
    ]
    patch_list = [
        (dx, dy)
        for dy in range(-patch_radius, patch_radius + 1)
        for dx in range(-patch_radius, patch_radius + 1)
    ]
    candidates = torch.as_tensor(candidate_list, device=device, dtype=dtype)
    patch_offsets = torch.as_tensor(patch_list, device=device, dtype=dtype)
    return candidates, patch_offsets, candidate_list


def extract_patch_batch(
    rgb_feature: Any,
    ir_feature: Any,
    rows: pd.DataFrame,
    orig_h: int,
    orig_w: int,
    stride: int,
    search_radius: int,
    patch_radius: int,
) -> Optional[Dict[str, Any]]:
    """Extract query/candidate patches without computing a similarity metric."""
    torch, _ = require_torch()
    if len(rows) == 0:
        return None
    if rgb_feature.shape != ir_feature.shape:
        raise RuntimeError(
            "RGB/IR P3 feature mismatch: {} vs {}".format(
                tuple(rgb_feature.shape), tuple(ir_feature.shape)
            )
        )
    rgb_feature = rgb_feature.float()
    ir_feature = ir_feature.float()
    _c, feature_h, feature_w = rgb_feature.shape
    input_h = int(feature_h * stride)
    input_w = int(feature_w * stride)
    geom = base.letterbox_geometry(orig_h, orig_w, input_h, input_w)
    device = rgb_feature.device
    dtype = rgb_feature.dtype

    ir_cx = torch.as_tensor(rows["ir_cx"].to_numpy(float), device=device, dtype=dtype)
    ir_cy = torch.as_tensor(rows["ir_cy"].to_numpy(float), device=device, dtype=dtype)
    rgb_cx = torch.as_tensor(rows["rgb_cx"].to_numpy(float), device=device, dtype=dtype)
    rgb_cy = torch.as_tensor(rows["rgb_cy"].to_numpy(float), device=device, dtype=dtype)
    center_x = (ir_cx * geom["gain_x"] + geom["pad_x"]) / float(stride) - 0.5
    center_y = (ir_cy * geom["gain_y"] + geom["pad_y"]) / float(stride) - 0.5
    centers = torch.stack([center_x, center_y], dim=-1)
    gt_dx = (rgb_cx - ir_cx) * geom["gain_x"] / float(stride)
    gt_dy = (rgb_cy - ir_cy) * geom["gain_y"] / float(stride)
    gt_delta = torch.stack([gt_dx, gt_dy], dim=-1)

    candidates, patch_offsets, candidate_list = candidate_tensors(
        device, dtype, search_radius, patch_radius
    )
    query_points = centers[:, None, :] + patch_offsets[None, :, :]
    candidate_points = (
        centers[:, None, None, :]
        + candidates[None, :, None, :]
        + patch_offsets[None, None, :, :]
    )
    xmin = geom["pad_x"] / float(stride) - 0.5
    ymin = geom["pad_y"] / float(stride) - 0.5
    xmax = (geom["pad_x"] + geom["resized_w"] - 1.0) / float(stride) - 0.5
    ymax = (geom["pad_y"] + geom["resized_h"] - 1.0) / float(stride) - 0.5
    bounds = (xmin, xmax, ymin, ymax)
    query_valid = base.points_inside(query_points, bounds).all(dim=-1)
    candidate_valid = base.points_inside(candidate_points, bounds).all(dim=-1)
    usable = query_valid & (candidate_valid.sum(dim=-1) > 0)
    if not bool(usable.any().item()):
        return None

    usable_indices = torch.nonzero(usable, as_tuple=False).flatten()
    query_points = query_points[usable_indices]
    candidate_points = candidate_points[usable_indices]
    candidate_valid = candidate_valid[usable_indices]
    centers = centers[usable_indices]
    gt_delta = gt_delta[usable_indices]
    selected_rows = rows.iloc[usable_indices.detach().cpu().numpy()].reset_index(drop=True)
    ir_patch = base.sample_feature(ir_feature, query_points)
    rgb_patches = base.sample_feature(rgb_feature, candidate_points)
    return {
        "ir_patch": ir_patch,
        "rgb_patches": rgb_patches,
        "candidate_valid": candidate_valid,
        "centers": centers,
        "gt_delta": gt_delta,
        "rows": selected_rows,
        "candidates": candidates,
        "patch_offsets": patch_offsets,
        "candidate_list": candidate_list,
        "bounds": bounds,
        "geom": geom,
        "feature_h": feature_h,
        "feature_w": feature_w,
        "input_h": input_h,
        "input_w": input_w,
    }


def gaussian_candidate_target(
    gt_delta: Any,
    candidates: Any,
    candidate_valid: Any,
    sigma: float,
):
    torch, _ = require_torch()
    squared_distance = ((candidates[None, :, :] - gt_delta[:, None, :]) ** 2).sum(dim=-1)
    target = torch.exp(-0.5 * squared_distance / (sigma * sigma))
    target = target * candidate_valid.to(target.dtype)
    denominator = target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return target / denominator


def add_callback(yolo: Any, event: str, callback: Any) -> None:
    callbacks = getattr(yolo, "callbacks", None)
    if not isinstance(callbacks, dict) or event not in callbacks:
        raise RuntimeError("YOLO callback registry does not expose {}.".format(event))
    callbacks[event].append(callback)


def remove_callback(yolo: Any, event: str, callback: Any) -> None:
    callbacks = getattr(yolo, "callbacks", None)
    if isinstance(callbacks, dict) and event in callbacks and callback in callbacks[event]:
        callbacks[event].remove(callback)


def run_validator(
    yolo: Any,
    callback: Any,
    args: argparse.Namespace,
    split: str,
) -> bool:
    torch, _ = require_torch()
    stopped_early = False
    add_callback(yolo, "on_val_batch_end", callback)
    try:
        with torch.inference_mode():
            yolo.val(
                data=args.data,
                split=split,
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
    except base.StopProbeEarly as exc:
        stopped_early = True
        print("[INFO] {}".format(exc))
    finally:
        remove_callback(yolo, "on_val_batch_end", callback)
    return stopped_early


class TrainingCallback:
    def __init__(
        self,
        rgb_capture: Any,
        ir_capture: Any,
        matcher: Any,
        optimizer: Any,
        state_df: pd.DataFrame,
        args: argparse.Namespace,
        epoch: int,
    ):
        self.rgb_capture = rgb_capture
        self.ir_capture = ir_capture
        self.matcher = matcher
        self.optimizer = optimizer
        self.args = args
        self.epoch = epoch
        self.groups = {
            str(image_id): group.reset_index(drop=True)
            for image_id, group in state_df.groupby("image_id", sort=False)
        }
        self.cursor = 0
        self.seen_images = 0
        self.images_with_rows = 0
        self.object_count = 0
        self.step_count = 0
        self.loss_sum = 0.0
        self.near_loss_sum = 0.0
        self.near_count = 0
        self.shift_loss_sum = 0.0
        self.shift_count = 0
        self.gt_vectors: List[np.ndarray] = []

    def __call__(self, validator: Any) -> None:
        torch, F = require_torch()
        rgb = self.rgb_capture.tensor
        ir = self.ir_capture.tensor
        if rgb is None or ir is None or rgb.ndim != 4 or ir.ndim != 4:
            raise RuntimeError("Training callback did not receive BxCxHxW P3 features.")
        batch_size = int(rgb.shape[0])
        paths = base.extract_paths_from_batch(validator, batch_size)
        if paths is None:
            all_paths = base.dataset_image_paths(validator)
            paths = all_paths[self.cursor : self.cursor + batch_size]
        if len(paths) != batch_size:
            raise RuntimeError("Training image path cursor mismatch.")
        self.cursor += batch_size
        remaining = batch_size
        if self.args.max_train_images is not None:
            remaining = max(
                0,
                min(batch_size, self.args.max_train_images - self.seen_images),
            )

        batch_items = []
        # YOLO validator runs under inference_mode. Disable it before cloning;
        # otherwise projection backward can fail on inference tensors.
        with torch.inference_mode(False), torch.enable_grad():
            for batch_index in range(remaining):
                path = paths[batch_index]
                image_id = Path(path).stem
                self.seen_images += 1
                rows = self.groups.get(image_id)
                if rows is None or len(rows) == 0:
                    continue
                rows = balanced_object_sample(
                    rows,
                    self.args.max_objects_per_image,
                    self.args.seed,
                    self.epoch,
                    image_id,
                )
                orig_h, orig_w = base.image_size(path)
                rgb_normal = rgb[batch_index].detach().clone().float()
                ir_normal = ir[batch_index].detach().clone().float()
                item = extract_patch_batch(
                    rgb_normal,
                    ir_normal,
                    rows,
                    orig_h,
                    orig_w,
                    self.args.stride,
                    self.args.search_radius,
                    self.args.patch_radius,
                )
                if item is not None:
                    batch_items.append(item)
                    self.images_with_rows += 1
            if batch_items:
                ir_patch = torch.cat([item["ir_patch"] for item in batch_items], dim=0)
                rgb_patches = torch.cat([item["rgb_patches"] for item in batch_items], dim=0)
                valid = torch.cat([item["candidate_valid"] for item in batch_items], dim=0)
                gt = torch.cat([item["gt_delta"] for item in batch_items], dim=0)
                state_names: List[str] = []
                for item in batch_items:
                    state_names.extend(item["rows"]["geometry_state"].astype(str).tolist())
                states = np.asarray(state_names)
                candidates = batch_items[0]["candidates"]
                target = gaussian_candidate_target(
                    gt, candidates, valid, self.args.target_sigma
                )
                scores = self.matcher.forward_scores(ir_patch, rgb_patches)
                if not scores.requires_grad:
                    raise RuntimeError(
                        "Projection scores have no gradient. The validator inference-mode "
                        "escape did not take effect in this PyTorch version."
                    )
                scores = scores.masked_fill(~valid, -1e4)
                per_object_loss = -(
                    target * F.log_softmax(scores / self.args.train_tau, dim=-1)
                ).sum(dim=-1)
                batch_state_counts = {
                    state: max(int((states == state).sum()), 1)
                    for state in ["near_center", "local_shift"]
                }
                batch_total = float(len(state_names))
                batch_state_weights = {
                    state: batch_total / (2.0 * count)
                    for state, count in batch_state_counts.items()
                }
                weights = torch.as_tensor(
                    [batch_state_weights.get(state, 1.0) for state in state_names],
                    device=per_object_loss.device,
                    dtype=per_object_loss.dtype,
                )
                loss = (per_object_loss * weights).sum() / weights.sum().clamp_min(1e-12)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.matcher.parameters(), self.args.grad_clip)
                self.optimizer.step()

                count = int(len(state_names))
                self.object_count += count
                self.step_count += 1
                self.loss_sum += float(loss.item())
                loss_cpu = per_object_loss.detach().cpu().numpy()
                near_mask = states == "near_center"
                shift_mask = states == "local_shift"
                if near_mask.any():
                    self.near_loss_sum += float(loss_cpu[near_mask].sum())
                    self.near_count += int(near_mask.sum())
                if shift_mask.any():
                    self.shift_loss_sum += float(loss_cpu[shift_mask].sum())
                    self.shift_count += int(shift_mask.sum())
                if self.epoch == 0 and shift_mask.any():
                    self.gt_vectors.append(gt[shift_mask].detach().cpu().numpy())

        if (
            self.args.max_train_images is not None
            and self.seen_images >= self.args.max_train_images
        ):
            raise base.StopProbeEarly(
                "Reached --max-train-images={}.".format(self.args.max_train_images)
            )

    def summary(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch + 1,
            "seen_images": self.seen_images,
            "images_with_rows": self.images_with_rows,
            "objects": self.object_count,
            "optimizer_steps": self.step_count,
            "mean_batch_loss": self.loss_sum / max(self.step_count, 1),
            "near_mean_object_loss": self.near_loss_sum / max(self.near_count, 1),
            "shift_mean_object_loss": self.shift_loss_sum / max(self.shift_count, 1),
        }


def learned_image_records(
    matcher: Any,
    rgb_feature: Any,
    ir_feature: Any,
    rows: pd.DataFrame,
    image_id: str,
    image_path: str,
    orig_h: int,
    orig_w: int,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    torch, _ = require_torch()
    stats = {"requested": len(rows), "usable": 0}
    with torch.inference_mode(False), torch.no_grad():
        rgb_normal = rgb_feature.detach().clone().float()
        ir_normal = ir_feature.detach().clone().float()
        item = extract_patch_batch(
            rgb_normal,
            ir_normal,
            rows,
            orig_h,
            orig_w,
            args.stride,
            args.search_radius,
            args.patch_radius,
        )
        if item is None:
            return [], stats
        scores = matcher.forward_scores(item["ir_patch"], item["rgb_patches"])
        scores = scores.masked_fill(~item["candidate_valid"], float("-inf"))
        argmax_idx = scores.argmax(dim=-1)
        pred = item["candidates"][argmax_idx]
        top = scores.topk(k=2, dim=-1).values

        # Learned forward-backward consistency.
        rgb_centers = item["centers"] + pred
        reverse_query_points = (
            rgb_centers[:, None, :] + item["patch_offsets"][None, :, :]
        )
        reverse_candidate_points = (
            rgb_centers[:, None, None, :]
            + item["candidates"][None, :, None, :]
            + item["patch_offsets"][None, None, :, :]
        )
        reverse_valid = base.points_inside(
            reverse_candidate_points, item["bounds"]
        ).all(dim=-1)
        rgb_query = base.sample_feature(rgb_normal, reverse_query_points)
        ir_candidates = base.sample_feature(ir_normal, reverse_candidate_points)
        reverse_scores = matcher.reverse_scores(rgb_query, ir_candidates)
        reverse_scores = reverse_scores.masked_fill(~reverse_valid, float("-inf"))
        reverse_idx = reverse_scores.argmax(dim=-1)
        reverse_pred = item["candidates"][reverse_idx]
        cycle = torch.linalg.vector_norm(pred + reverse_pred, dim=-1)
        mutual = (pred + reverse_pred).abs().max(dim=-1).values < 1e-6

        zero_epe = torch.linalg.vector_norm(item["gt_delta"], dim=-1) * args.stride
        argmax_epe = (
            torch.linalg.vector_norm(pred - item["gt_delta"], dim=-1) * args.stride
        )
        scores_np = scores.cpu().numpy()
        pred_np = pred.cpu().numpy()
        gt_np = item["gt_delta"].cpu().numpy()
        top_np = top.cpu().numpy()
        cycle_np = cycle.cpu().numpy()
        mutual_np = mutual.cpu().numpy()
        zero_np = zero_epe.cpu().numpy()
        argmax_np = argmax_epe.cpu().numpy()

    geom = item["geom"]
    output: List[Dict[str, Any]] = []
    for index, row in item["rows"].iterrows():
        area = base.infer_mean_area(row, geom["gain_x"], geom["gain_y"])
        record: Dict[str, Any] = {
            "image_id": image_id,
            "image_path": image_path,
            "class_label": base.row_value(
                row, "cls", base.row_value(row, "class_id", "unknown")
            ),
            "rgb_source_index": base.row_value(row, "rgb_source_index", -1),
            "ir_source_index": base.row_value(row, "ir_source_index", -1),
            "geometry_state": base.row_value(row, "geometry_state", "unknown"),
            "size_group": base.size_group(area),
            "mean_area_input_px2": area,
            "offset_distance_input_px": float(zero_np[index]),
            "offset_bin": base.offset_bin(float(zero_np[index])),
            "orig_h": orig_h,
            "orig_w": orig_w,
            "input_h": item["input_h"],
            "input_w": item["input_w"],
            "feature_h": item["feature_h"],
            "feature_w": item["feature_w"],
            "gain_x": geom["gain_x"],
            "gain_y": geom["gain_y"],
            "pad_x": geom["pad_x"],
            "pad_y": geom["pad_y"],
            "gt_dx_cell": float(gt_np[index, 0]),
            "gt_dy_cell": float(gt_np[index, 1]),
            "pred_argmax_dx_cell": float(pred_np[index, 0]),
            "pred_argmax_dy_cell": float(pred_np[index, 1]),
            "zero_epe_px": float(zero_np[index]),
            "argmax_epe_px": float(argmax_np[index]),
            "argmax_gain_vs_zero_px": float(zero_np[index] - argmax_np[index]),
            "top1_score": float(top_np[index, 0]),
            "top2_score": float(top_np[index, 1]),
            "score_margin": float(top_np[index, 0] - top_np[index, 1]),
            "valid_candidate_count": int(
                item["candidate_valid"][index].sum().item()
            ),
            "cycle_error_cell": float(cycle_np[index]),
            "mutual_match": bool(mutual_np[index]),
            "reverse_search_valid": True,
            "peak_on_boundary": bool(
                abs(float(pred_np[index, 0])) == args.search_radius
                or abs(float(pred_np[index, 1])) == args.search_radius
            ),
        }
        for candidate_index, (dx, dy) in enumerate(item["candidate_list"]):
            value = scores_np[index, candidate_index]
            record[base.candidate_column(dx, dy)] = (
                float(value) if np.isfinite(value) else np.nan
            )
        output.append(record)
    stats["usable"] = len(output)
    return output, stats


class EvaluationCallback:
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
        self.requested = 0
        self.records: List[Dict[str, Any]] = []

    def __call__(self, validator: Any) -> None:
        rgb = self.rgb_capture.tensor
        ir = self.ir_capture.tensor
        if rgb is None or ir is None or rgb.ndim != 4 or ir.ndim != 4:
            raise RuntimeError("Evaluation callback did not receive P3 features.")
        batch_size = int(rgb.shape[0])
        paths = base.extract_paths_from_batch(validator, batch_size)
        if paths is None:
            all_paths = base.dataset_image_paths(validator)
            paths = all_paths[self.cursor : self.cursor + batch_size]
        if len(paths) != batch_size:
            raise RuntimeError("Evaluation image path cursor mismatch.")
        self.cursor += batch_size
        remaining = batch_size
        if self.args.max_val_images is not None:
            remaining = max(0, min(batch_size, self.args.max_val_images - self.seen_images))
        for batch_index in range(remaining):
            path = paths[batch_index]
            image_id = Path(path).stem
            self.seen_images += 1
            rows = self.groups.get(image_id)
            if rows is None or len(rows) == 0:
                continue
            orig_h, orig_w = base.image_size(path)
            records, stats = learned_image_records(
                self.matcher,
                rgb[batch_index],
                ir[batch_index],
                rows,
                image_id,
                path,
                orig_h,
                orig_w,
                self.args,
            )
            self.records.extend(records)
            self.processed_images += 1
            self.requested += stats["requested"]
        if (
            self.args.max_val_images is not None
            and self.seen_images >= self.args.max_val_images
        ):
            raise base.StopProbeEarly(
                "Reached --max-val-images={}.".format(self.args.max_val_images)
            )


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.std(a[mask]) <= 0 or np.std(b[mask]) <= 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    mask = np.isfinite(scores)
    labels = labels[mask].astype(bool)
    scores = scores[mask]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy(float)
    return float(
        (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def route_margin_metrics(df: pd.DataFrame) -> Dict[str, float]:
    score_columns = [column for column in df.columns if column.startswith("score_dy_")]
    center_column = "score_dy_z0_dx_z0"
    noncenter = [column for column in score_columns if column != center_column]
    route_margin = df[noncenter].max(axis=1) - df[center_column]
    labels = df["geometry_state"].eq("local_shift").to_numpy()
    raw_auc = binary_auc(labels, route_margin.to_numpy(float))
    work = pd.DataFrame(
        {
            "image_id": df["image_id"].astype(str),
            "label": labels,
            "margin": route_margin.to_numpy(float),
        }
    )
    centered = work["margin"] - work.groupby("image_id")["margin"].transform("mean")
    centered_auc = binary_auc(labels, centered.to_numpy(float))
    pair_values = []
    for _image, group in work.groupby("image_id"):
        positive = group.loc[group["label"], "margin"].to_numpy(float)
        negative = group.loc[~group["label"], "margin"].to_numpy(float)
        if len(positive) and len(negative):
            difference = positive[:, None] - negative[None, :]
            pair_values.extend(
                ((difference > 0).astype(float) + 0.5 * (difference == 0)).ravel()
            )
    return {
        "route_margin_state_auc": raw_auc,
        "route_margin_image_centered_state_auc": centered_auc,
        "route_margin_within_image_pair_auc": (
            float(np.mean(pair_values)) if pair_values else float("nan")
        ),
        "route_margin_within_image_pair_count": len(pair_values),
    }


def decision_summary(
    df: pd.DataFrame,
    train_global_mean: np.ndarray,
    stride: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"object_count": len(df)}
    near = df[df["geometry_state"].eq("near_center")]
    shift = df[df["geometry_state"].eq("local_shift")].copy().reset_index(drop=True)
    result["near_center"] = {
        "object_count": len(near),
        "zero_mean_epe_px": float(near["zero_epe_px"].mean()),
        "soft_mean_epe_px": float(near["soft_epe_px"].mean()),
        "false_movement_mean_px": float(
            np.linalg.norm(
                near[["pred_soft_dx_cell", "pred_soft_dy_cell"]].to_numpy(float), axis=1
            ).mean()
            * stride
        ),
    }
    if len(shift):
        gt = shift[["gt_dx_cell", "gt_dy_cell"]].to_numpy(float)
        pred = shift[["pred_soft_dx_cell", "pred_soft_dy_cell"]].to_numpy(float)
        fixed = np.tile(train_global_mean.reshape(1, 2), (len(gt), 1))
        result["local_shift"] = {
            "object_count": len(shift),
            "image_count": int(shift["image_id"].nunique()),
            "zero_mean_epe_px": float(np.linalg.norm(gt, axis=1).mean() * stride),
            "soft_mean_epe_px": float(np.linalg.norm(gt - pred, axis=1).mean() * stride),
            "train_global_mean_baseline_epe_px": float(
                np.linalg.norm(gt - fixed, axis=1).mean() * stride
            ),
            "dx_pearson": pearson(gt[:, 0], pred[:, 0]),
            "dy_pearson": pearson(gt[:, 1], pred[:, 1]),
            "magnitude_pearson": pearson(
                np.linalg.norm(gt, axis=1), np.linalg.norm(pred, axis=1)
            ),
        }
        sizes = shift.groupby("image_id")["image_id"].transform("size")
        mask = sizes.ge(2).to_numpy()
        gt_centered = np.zeros_like(gt)
        pred_centered = np.zeros_like(pred)
        for _image, group in shift.groupby("image_id"):
            indices = group.index.to_numpy()
            gt_centered[indices] = gt[indices] - gt[indices].mean(axis=0)
            pred_centered[indices] = pred[indices] - pred[indices].mean(axis=0)
        if mask.any():
            residual_zero = np.linalg.norm(gt_centered[mask], axis=1).mean() * stride
            residual_pred = (
                np.linalg.norm(gt_centered[mask] - pred_centered[mask], axis=1).mean()
                * stride
            )
            result["within_image_local_residual"] = {
                "object_count": int(mask.sum()),
                "image_count": int(shift.loc[mask, "image_id"].nunique()),
                "dx_pearson": pearson(
                    gt_centered[mask, 0], pred_centered[mask, 0]
                ),
                "dy_pearson": pearson(
                    gt_centered[mask, 1], pred_centered[mask, 1]
                ),
                "zero_residual_epe_px": float(residual_zero),
                "predicted_residual_epe_px": float(residual_pred),
                "residual_gain_px": float(residual_zero - residual_pred),
            }
    result.update(route_margin_metrics(df))
    result["interpretation_rules"] = {
        "local_correspondence_supported_only_if": [
            "local_shift soft EPE is below zero-offset EPE",
            "local_shift soft EPE is below the train-global-mean baseline",
            "near-center false movement is small",
            "within-image centered dx/dy correlations are positive and stable",
            "within-image predicted residual EPE is below zero-residual EPE",
            "within-image route-margin AUC is materially above 0.5",
        ],
        "warning": (
            "This experiment uses train RGB boxes. Positive results establish only "
            "a representation upper bound, not a deployable no-RGB-supervision method."
        ),
    }
    return result


def run_self_test() -> None:
    torch, F = require_torch()
    torch.manual_seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, k, p, c = 128, 25, 9, 16
    embed_dim = 16
    matcher = build_matcher(c, embed_dim).to(device)
    latent = torch.randn(n, p, c, device=device)
    rgb_transform = torch.randn(c, c, device=device)
    ir_transform = torch.randn(c, c, device=device)
    ir_patch = latent @ ir_transform
    target_index = torch.randint(0, k, (n,), device=device)
    rgb_patches = torch.randn(n, k, p, c, device=device)
    matching = latent @ rgb_transform
    rgb_patches[torch.arange(n, device=device), target_index] = matching
    optimizer = torch.optim.AdamW(matcher.parameters(), lr=3e-3)
    before = None
    for step in range(200):
        scores = matcher.forward_scores(ir_patch, rgb_patches)
        loss = F.cross_entropy(scores / 0.1, target_index)
        if before is None:
            before = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        scores = matcher.forward_scores(ir_patch, rgb_patches)
        accuracy = float((scores.argmax(dim=-1) == target_index).float().mean().item())
        after = float(F.cross_entropy(scores / 0.1, target_index).item())
    if not after < before or accuracy < 0.90:
        raise AssertionError(
            "Projection self-test failed: loss {:.4f}->{:.4f}, acc={:.3f}".format(
                before, after, accuracy
            )
        )
    print("[PASS] modality-specific projection gradients")
    print("[PASS] local candidate classification")
    print("[PASS] synthetic correspondence recovery, accuracy={:.3f}".format(accuracy))
    print("All learned common-space probe self-tests passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a P3 learned common-space correspondence upper bound."
    )
    parser.add_argument("--weights", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/obb/base_add/weights/best.pt",
                        help="Add-baseline best.pt")
    parser.add_argument("--data", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/data/dronevehicle_obb.yaml",
                        help="TwoStream dataset YAML")
    parser.add_argument("--train-state-csv", type=str,default="/storage/jyx4/projects/dataset/DroneVehicle/tool/postcrop_offset_stats/matched_offsets_postcrop.csv")
    parser.add_argument("--val-state-csv", type=str, default="/storage/jyx4/projects/dataset/DroneVehicle/tool/val_matched_state_index.csv")
    parser.add_argument("--output", type=str, default="runs/p3_common_space_probe/first_run")
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rgb-layer", type=int, default=11)
    parser.add_argument("--ir-layer", type=int, default=12)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--search-radius", type=int, default=2)
    parser.add_argument("--patch-radius", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--train-tau", type=float, default=0.10)
    parser.add_argument("--eval-tau", type=float, default=0.10)
    parser.add_argument("--target-sigma", type=float, default=0.50)
    parser.add_argument("--max-train-images", type=int, default=1000)
    parser.add_argument("--max-val-images", type=int, default=100)
    parser.add_argument("--max-objects-per-image", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    required = ["weights", "data", "train_state_csv", "val_state_csv"]
    for name in required:
        value = getattr(args, name)
        if not value:
            raise ValueError("--{} is required.".format(name.replace("_", "-")))
        if not Path(value).exists():
            raise FileNotFoundError("Missing {}: {}".format(name, value))
    if args.epochs <= 0 or args.lr <= 0 or args.train_tau <= 0 or args.eval_tau <= 0:
        raise ValueError("epochs/lr/tau values must be positive.")
    if args.max_train_images is not None and args.max_train_images <= 0:
        args.max_train_images = None
    if args.max_val_images is not None and args.max_val_images <= 0:
        args.max_val_images = None
    if args.max_objects_per_image is not None and args.max_objects_per_image <= 0:
        args.max_objects_per_image = None


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
            "Run this script from the modified TwoStream YOLO project root."
        ) from exc

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_df = read_probe_state_csv(args.train_state_csv, "train")
    val_df = read_probe_state_csv(args.val_state_csv, "val")
    counts = train_df["geometry_state"].value_counts().to_dict()
    print("Train states: {}".format(counts))
    print("Val states: {}".format(val_df["geometry_state"].value_counts().to_dict()))
    print("Training loss balances near_center/local_shift within each optimizer batch.")

    yolo = YOLO(args.weights)
    layers = base.resolve_model_layers(yolo)
    base.print_layer_check(layers, args.rgb_layer, args.ir_layer)
    rgb_capture = base.TensorCapture(layers[args.rgb_layer], "RGB P3")
    ir_capture = base.TensorCapture(layers[args.ir_layer], "IR P3")

    # Infer P3 channel count with one short validation callback-free forward run
    # by using the first training batch callback and lazy matcher initialization.
    matcher_holder: Dict[str, Any] = {}

    class InitCallback:
        def __call__(self, _validator: Any) -> None:
            rgb = rgb_capture.tensor
            if rgb is None or rgb.ndim != 4:
                raise RuntimeError("Cannot infer P3 channel count.")
            matcher_holder["channels"] = int(rgb.shape[1])
            raise base.StopProbeEarly("P3 channel initialization complete.")

    run_validator(yolo, InitCallback(), args, "train")
    channels = int(matcher_holder["channels"])
    feature_device = rgb_capture.tensor.device
    matcher = build_matcher(channels, args.embed_dim).to(feature_device)
    optimizer = torch.optim.AdamW(
        matcher.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    print("Projection matcher: {} -> {} dimensions".format(channels, args.embed_dim))

    training_history = []
    train_gt_vectors: List[np.ndarray] = []
    try:
        for epoch in range(args.epochs):
            matcher.train()
            callback = TrainingCallback(
                rgb_capture,
                ir_capture,
                matcher,
                optimizer,
                train_df,
                args,
                epoch,
            )
            run_validator(yolo, callback, args, "train")
            epoch_summary = callback.summary()
            training_history.append(epoch_summary)
            if epoch == 0:
                train_gt_vectors.extend(callback.gt_vectors)
            print(
                "Epoch {}/{}: loss={:.5f}, near={:.5f}, shift={:.5f}, objects={}".format(
                    epoch + 1,
                    args.epochs,
                    epoch_summary["mean_batch_loss"],
                    epoch_summary["near_mean_object_loss"],
                    epoch_summary["shift_mean_object_loss"],
                    epoch_summary["objects"],
                )
            )

        if not train_gt_vectors:
            raise RuntimeError("No training GT vectors were collected.")
        train_global_mean = np.concatenate(train_gt_vectors, axis=0).mean(axis=0)
        checkpoint_path = output / "projection_probe.pt"
        torch.save(
            {
                "matcher": matcher.state_dict(),
                "input_dim": channels,
                "embed_dim": args.embed_dim,
                "train_global_mean_offset_cell": train_global_mean.tolist(),
                "args": vars(args),
                "training_history": training_history,
            },
            checkpoint_path,
        )

        matcher.eval()
        evaluation = EvaluationCallback(
            rgb_capture, ir_capture, matcher, val_df, args
        )
        stopped = run_validator(yolo, evaluation, args, "val")
        if not evaluation.records:
            raise RuntimeError("No learned validation diagnostic was produced.")
        raw = pd.DataFrame(evaluation.records)
        raw_path = output / "learned_p3_raw.csv"
        raw.to_csv(raw_path, index=False, encoding="utf-8-sig")
        diagnostics = base.attach_soft_metrics(
            raw, args.search_radius, args.stride, args.eval_tau
        )
        diagnostics_path = output / "learned_p3_diagnostics.csv"
        diagnostics.to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
        summary = base.build_correspondence_summary(
            diagnostics, args.bootstrap, args.seed
        )
        summary_path = output / "learned_p3_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        confidence, risk = base.build_confidence_tables(
            diagnostics,
            coverage_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        )
        confidence_path = output / "learned_confidence_quality.csv"
        risk_path = output / "learned_risk_coverage.csv"
        confidence.to_csv(confidence_path, index=False, encoding="utf-8-sig")
        risk.to_csv(risk_path, index=False, encoding="utf-8-sig")
        decision = decision_summary(
            diagnostics, train_global_mean, args.stride
        )
        decision.update(
            {
                "weights": str(Path(args.weights).resolve()),
                "train_state_csv": str(Path(args.train_state_csv).resolve()),
                "val_state_csv": str(Path(args.val_state_csv).resolve()),
                "train_global_mean_offset_cell": train_global_mean.tolist(),
                "max_val_images": args.max_val_images,
                "val_stopped_early": stopped,
                "seen_val_images": evaluation.seen_images,
                "processed_val_images": evaluation.processed_images,
                "training_history": training_history,
            }
        )
        decision_path = output / "learned_probe_decision_summary.json"
        with decision_path.open("w", encoding="utf-8") as handle:
            json.dump(decision, handle, ensure_ascii=False, indent=2)
        history_path = output / "training_history.csv"
        pd.DataFrame(training_history).to_csv(
            history_path, index=False, encoding="utf-8-sig"
        )
        print("\nLearned common-space upper-bound probe completed.")
        print("  Output: {}".format(output))
        print("  Diagnostic objects: {}".format(len(diagnostics)))
        print("  Decision summary: {}".format(decision_path))
    finally:
        rgb_capture.close()
        ir_capture.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[FATAL] {}".format(exc), file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
