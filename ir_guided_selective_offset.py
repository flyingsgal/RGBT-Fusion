"""IR-guided confidence-rejected local RGB offset sampling.

This file is self-contained so the class can first be tested outside Ultralytics
and then copied into ``ultralytics/nn/modules/block.py``.

Direction convention:
    RGB->IR means that IR is the spatial reference.  At an IR position p, the
    module searches RGB positions p + delta and writes the sampled RGB feature
    into the IR-referenced fused representation.

The initial research version intentionally contains no cross-attention,
bidirectional update, low-light branch, or P4/P5 alignment.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IRGuidedSelectiveOffset(nn.Module):
    """Selectively sample RGB into the IR coordinate frame at P3.

    Args:
        c: RGB/IR input channel count. Both inputs must have this channel count.
        embed_dim: Common-space projection dimension. The supplied projection
            probe checkpoint currently uses 128.
        search_radius: Candidate-center radius in feature cells. Radius 2 gives
            a 5*5 candidate set.
        patch_radius: Local descriptor radius. Radius 1 gives a 3*3 patch.
        margin_threshold: The best-noncenter minus center score threshold.
        entropy_threshold: Maximum normalized candidate entropy for a reliable
            shift.
        softmax_tau: Temperature used for candidate probabilities and soft
            offset expectation.
        route_mode: One of ``center``, ``all``, or ``reliable``.
            center: always use zero offset (Add baseline behavior).
            all: apply the soft offset at every fully valid search position.
            reliable: apply it only where margin and entropy pass the route.
        freeze_projection: Freeze RGB/IR projection weights. This should be True
            for the current RGB-box-supervised upper-bound experiment.
        projection_path: Optional ``projection_probe.pt`` path. The checkpoint
            must contain ``matcher``, ``input_dim``, and ``embed_dim``.
        require_full_search: If True, only cells for which all candidates and
            all patch tokens are in bounds may move. Other cells keep zero
            offset.
        debug: Store detached CPU diagnostic maps in ``last_debug``. Do not
            enable this during normal training because it synchronizes devices.

    Input:
        ``[rgb, ir]``, both B*C*H*W tensors with identical shapes.

    Output:
        ``ir + routed_rgb``. At inactive locations, ``routed_rgb == rgb``, so
        the exact zero-offset behavior is the ordinary Add fusion.
    """

    ROUTE_INVALID = 0
    ROUTE_ALIGNED = 1
    ROUTE_RELIABLE = 2
    ROUTE_UNCERTAIN = 3
    VALID_ROUTE_MODES = {"center", "all", "reliable"}

    def __init__(
        self,
        c: int,
        embed_dim: int = 64,
        search_radius: int = 2,
        patch_radius: int = 1,
        margin_threshold: float = 0.10,
        entropy_threshold: float = 0.68,
        softmax_tau: float = 0.10,
        route_mode: str = "reliable",
        freeze_projection: bool = True,
        projection_path: str = "",
        require_full_search: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if c <= 0 or embed_dim <= 0:
            raise ValueError("c and embed_dim must be positive.")
        if search_radius < 0 or patch_radius < 0:
            raise ValueError("search_radius and patch_radius must be non-negative.")
        if softmax_tau <= 0:
            raise ValueError("softmax_tau must be positive.")
        if not 0.0 <= entropy_threshold <= 1.0:
            raise ValueError("entropy_threshold must be in [0, 1].")
        route_mode = str(route_mode).lower().strip()
        if route_mode not in self.VALID_ROUTE_MODES:
            raise ValueError(
                "route_mode must be one of {}, got {!r}.".format(
                    sorted(self.VALID_ROUTE_MODES), route_mode
                )
            )

        self.c = int(c)
        self.input_dim = int(c)
        self.embed_dim = int(embed_dim)
        self.search_radius = int(search_radius)
        self.patch_radius = int(patch_radius)
        self.margin_threshold = float(margin_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.softmax_tau = float(softmax_tau)
        self.route_mode = route_mode
        self.freeze_projection = bool(freeze_projection)
        self.projection_path = str(projection_path or "")
        self.require_full_search = bool(require_full_search)
        self.debug = bool(debug)
        self.last_debug: Optional[Dict[str, torch.Tensor]] = None

        # Keep nn.Linear rather than Conv2d so projection_probe.pt loads exactly.
        self.rgb_proj = nn.Linear(self.c, self.embed_dim, bias=True)
        self.ir_proj = nn.Linear(self.c, self.embed_dim, bias=True)
        nn.init.orthogonal_(self.rgb_proj.weight)
        nn.init.orthogonal_(self.ir_proj.weight)
        nn.init.zeros_(self.rgb_proj.bias)
        nn.init.zeros_(self.ir_proj.bias)

        candidates = [
            (dx, dy)
            for dy in range(-self.search_radius, self.search_radius + 1)
            for dx in range(-self.search_radius, self.search_radius + 1)
        ]
        self.candidate_list: List[Tuple[int, int]] = candidates
        self.center_index = candidates.index((0, 0))
        self.register_buffer(
            "candidate_offsets",
            torch.tensor(candidates, dtype=torch.float32),
            persistent=False,
        )

        if self.projection_path:
            self.load_projection_checkpoint(self.projection_path)
        self.set_projection_trainable(not self.freeze_projection)

    def extra_repr(self) -> str:
        return (
            "c={}, embed_dim={}, search_radius={}, patch_radius={}, "
            "margin_threshold={}, entropy_threshold={}, softmax_tau={}, "
            "route_mode={!r}, freeze_projection={}, require_full_search={}".format(
                self.c,
                self.embed_dim,
                self.search_radius,
                self.patch_radius,
                self.margin_threshold,
                self.entropy_threshold,
                self.softmax_tau,
                self.route_mode,
                self.freeze_projection,
                self.require_full_search,
            )
        )

    def set_route_mode(self, route_mode: str) -> None:
        """Change the ablation mode without rebuilding the model."""
        route_mode = str(route_mode).lower().strip()
        if route_mode not in self.VALID_ROUTE_MODES:
            raise ValueError(
                "route_mode must be one of {}, got {!r}.".format(
                    sorted(self.VALID_ROUTE_MODES), route_mode
                )
            )
        self.route_mode = route_mode

    def set_projection_trainable(self, trainable: bool) -> None:
        for parameter in self.rgb_proj.parameters():
            parameter.requires_grad_(bool(trainable))
        for parameter in self.ir_proj.parameters():
            parameter.requires_grad_(bool(trainable))
        self.freeze_projection = not bool(trainable)

    def load_projection_checkpoint(self, path: str) -> None:
        """Load the RGB-box-supervised common-space probe checkpoint."""
        checkpoint_path = Path(path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Projection checkpoint does not exist: {}".format(checkpoint_path)
            )
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state = checkpoint.get("matcher", checkpoint)

        checkpoint_input = checkpoint.get("input_dim")
        checkpoint_embed = checkpoint.get("embed_dim")
        if checkpoint_input is not None and int(checkpoint_input) != self.c:
            raise ValueError(
                "Projection input mismatch: module c={} but checkpoint input_dim={}."
                .format(self.c, checkpoint_input)
            )
        if checkpoint_embed is not None and int(checkpoint_embed) != self.embed_dim:
            raise ValueError(
                "Projection embed mismatch: module embed_dim={} but checkpoint "
                "embed_dim={}.".format(self.embed_dim, checkpoint_embed)
            )

        required = {
            "rgb_proj.weight",
            "rgb_proj.bias",
            "ir_proj.weight",
            "ir_proj.bias",
        }
        missing = sorted(required.difference(state.keys()))
        if missing:
            raise ValueError(
                "Projection checkpoint is missing keys: {}".format(missing)
            )
        self.rgb_proj.load_state_dict(
            {"weight": state["rgb_proj.weight"], "bias": state["rgb_proj.bias"]},
            strict=True,
        )
        self.ir_proj.load_state_dict(
            {"weight": state["ir_proj.weight"], "bias": state["ir_proj.bias"]},
            strict=True,
        )
        self.projection_path = str(checkpoint_path.resolve())

    @staticmethod
    def _shift_source_to_destination(
        tensor: torch.Tensor, dx: int, dy: int
    ) -> torch.Tensor:
        """Return out[..., y, x] = tensor[..., y + dy, x + dx]."""
        if tensor.ndim < 2:
            raise ValueError("Shifted tensor must have at least two dimensions.")
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

    def _project(self, feature: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        # Linear consumes the last dimension and therefore matches the probe.
        projected = projection(feature.permute(0, 2, 3, 1))
        projected = F.normalize(projected, dim=-1, eps=1e-6)
        return projected.permute(0, 3, 1, 2).contiguous()

    def _dense_candidate_scores(
        self, rgb: torch.Tensor, ir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return float32 score maps and validity masks, both B*K*H*W."""
        rgb_projected = self._project(rgb, self.rgb_proj)
        ir_projected = self._project(ir, self.ir_proj)
        batch, _, height, width = rgb_projected.shape
        kernel = 2 * self.patch_radius + 1

        # float32 score computation keeps entropy/margin stable under AMP.
        ir_score = ir_projected.float()
        base_valid = torch.ones(
            (batch, 1, height, width),
            device=rgb.device,
            dtype=torch.float32,
        )
        score_maps: List[torch.Tensor] = []
        valid_maps: List[torch.Tensor] = []
        for dx, dy in self.candidate_list:
            shifted_rgb = self._shift_source_to_destination(rgb_projected, dx, dy)
            shifted_valid = self._shift_source_to_destination(base_valid, dx, dy)
            point_score = (ir_score * shifted_rgb.float()).sum(dim=1, keepdim=True)
            point_score = point_score * shifted_valid
            patch_score = F.avg_pool2d(
                point_score,
                kernel_size=kernel,
                stride=1,
                padding=self.patch_radius,
                count_include_pad=True,
            )
            patch_valid_fraction = F.avg_pool2d(
                shifted_valid,
                kernel_size=kernel,
                stride=1,
                padding=self.patch_radius,
                count_include_pad=True,
            )
            candidate_valid = patch_valid_fraction >= (1.0 - 1e-6)
            score_maps.append(patch_score[:, 0])
            valid_maps.append(candidate_valid[:, 0])

        scores = torch.stack(score_maps, dim=1)
        valid = torch.stack(valid_maps, dim=1)
        return scores, valid

    def _route_from_scores(
        self, scores: torch.Tensor, valid: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Build probability, offset, confidence, and three-state route maps."""
        if scores.ndim != 4 or valid.shape != scores.shape:
            raise ValueError("scores and valid must both have shape B*K*H*W.")
        candidate_count = int(scores.shape[1])
        if candidate_count != len(self.candidate_list):
            raise ValueError("Candidate count does not match module geometry.")

        center_valid = valid[:, self.center_index]
        full_search_valid = valid.all(dim=1)
        any_search_valid = valid.any(dim=1)
        usable = center_valid & any_search_valid
        if self.require_full_search:
            usable = usable & full_search_valid

        negative = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~valid, negative)
        maximum = masked_scores.max(dim=1).values
        maximum = torch.where(usable, maximum, torch.zeros_like(maximum))
        logits = (masked_scores - maximum[:, None]) / self.softmax_tau
        exp_scores = torch.where(valid, torch.exp(logits), torch.zeros_like(logits))
        denominator = exp_scores.sum(dim=1).clamp_min(1e-12)
        probabilities = exp_scores / denominator[:, None]
        probabilities = torch.where(
            usable[:, None], probabilities, torch.zeros_like(probabilities)
        )

        offsets = self.candidate_offsets.to(
            device=scores.device, dtype=probabilities.dtype
        )
        pred_dx = (probabilities * offsets[:, 0][None, :, None, None]).sum(dim=1)
        pred_dy = (probabilities * offsets[:, 1][None, :, None, None]).sum(dim=1)

        entropy_raw = -torch.where(
            probabilities > 0,
            probabilities * torch.log(probabilities.clamp_min(1e-12)),
            torch.zeros_like(probabilities),
        ).sum(dim=1)
        valid_count = valid.sum(dim=1)
        entropy_denominator = torch.log(
            valid_count.clamp_min(2).to(dtype=probabilities.dtype)
        )
        normalized_entropy = entropy_raw / entropy_denominator
        normalized_entropy = torch.where(
            usable, normalized_entropy, torch.ones_like(normalized_entropy)
        )

        noncenter = [
            index for index in range(candidate_count) if index != self.center_index
        ]
        best_noncenter = masked_scores[:, noncenter].max(dim=1).values
        center_score = masked_scores[:, self.center_index]
        margin = best_noncenter - center_score
        margin = torch.where(usable, margin, torch.zeros_like(margin))

        route = torch.full_like(valid_count, self.ROUTE_INVALID, dtype=torch.uint8)
        aligned = usable & (margin <= self.margin_threshold)
        reliable = (
            usable
            & (margin > self.margin_threshold)
            & (normalized_entropy <= self.entropy_threshold)
        )
        uncertain = (
            usable
            & (margin > self.margin_threshold)
            & (normalized_entropy > self.entropy_threshold)
        )
        route[aligned] = self.ROUTE_ALIGNED
        route[reliable] = self.ROUTE_RELIABLE
        route[uncertain] = self.ROUTE_UNCERTAIN

        return {
            "probabilities": probabilities,
            "pred_dx": pred_dx,
            "pred_dy": pred_dy,
            "entropy": normalized_entropy,
            "margin": margin,
            "route": route,
            "usable": usable,
            "full_search_valid": full_search_valid,
        }

    @staticmethod
    def _base_grid(
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # align_corners=False: feature-cell center x maps to 2*(x+0.5)/W-1.
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5)
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5)
        ys = 2.0 * ys / float(height) - 1.0
        xs = 2.0 * xs / float(width) - 1.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1).clone()

    def _sample_rgb(
        self,
        rgb: torch.Tensor,
        pred_dx: torch.Tensor,
        pred_dy: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = rgb.shape
        dx = torch.where(active, pred_dx, torch.zeros_like(pred_dx))
        dy = torch.where(active, pred_dy, torch.zeros_like(pred_dy))
        dx = dx.clamp(-float(self.search_radius), float(self.search_radius))
        dy = dy.clamp(-float(self.search_radius), float(self.search_radius))

        grid = self._base_grid(
            batch, height, width, rgb.device, rgb.dtype
        )
        grid[..., 0] = grid[..., 0] + dx.to(rgb.dtype) * (2.0 / float(width))
        grid[..., 1] = grid[..., 1] + dy.to(rgb.dtype) * (2.0 / float(height))
        sampled_rgb = F.grid_sample(
            rgb,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        # reliable位置使用重采样结果；
        # aligned、uncertain及边界无效位置严格保留原RGB。
        return torch.where(active[:, None], sampled_rgb, rgb)

    def _save_debug(
        self, route_data: Dict[str, torch.Tensor], active: torch.Tensor
    ) -> None:
        if not self.debug:
            self.last_debug = None
            return
        self.last_debug = {
            "route": route_data["route"].detach().cpu(),
            "margin": route_data["margin"].detach().cpu(),
            "entropy": route_data["entropy"].detach().cpu(),
            "pred_dx": route_data["pred_dx"].detach().cpu(),
            "pred_dy": route_data["pred_dy"].detach().cpu(),
            "active": active.detach().cpu(),
            "full_search_valid": route_data["full_search_valid"].detach().cpu(),
        }

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(
                "IRGuidedSelectiveOffset expects [rgb_feat, ir_feat]."
            )
        rgb, ir = x
        if not isinstance(rgb, torch.Tensor) or not isinstance(ir, torch.Tensor):
            raise TypeError("Both inputs must be torch.Tensor objects.")
        if rgb.shape != ir.shape or rgb.ndim != 4:
            raise ValueError(
                "RGB/IR must have identical B*C*H*W shapes, got {} and {}."
                .format(tuple(rgb.shape), tuple(ir.shape))
            )
        if int(rgb.shape[1]) != self.c:
            raise ValueError(
                "Input channel mismatch: module c={} but feature channels={}."
                .format(self.c, int(rgb.shape[1]))
            )
        if rgb.device != ir.device or rgb.dtype != ir.dtype:
            ir = ir.to(device=rgb.device, dtype=rgb.dtype)

        if self.route_mode == "center":
            self.last_debug = None
            return ir + rgb

        scores, valid = self._dense_candidate_scores(rgb, ir)
        route_data = self._route_from_scores(scores, valid)
        if self.route_mode == "all":
            active = route_data["usable"]
        else:
            active = route_data["route"].eq(self.ROUTE_RELIABLE)

        routed_rgb = self._sample_rgb(
            rgb, route_data["pred_dx"], route_data["pred_dy"], active
        )
        self._save_debug(route_data, active)
        return ir + routed_rgb

