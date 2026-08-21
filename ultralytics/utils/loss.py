# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors

from .metrics import bbox_iou, probiou
from .tal import bbox2dist
import os
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image

class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()



from .metrics import piou

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # 选择piou还是ciou
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # iou = 1-piou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, PIoU2=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        
        # iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, WIoU=True, scale=True)
        # if type(iou) is tuple:
        #     if len(iou) == 2:
        #         loss_iou = ((1.0 - iou[0]) * iou[1].detach() * weight).sum() / target_scores_sum
        #     else:
        #         loss_iou = (iou[0] * iou[1] * weight).sum() / target_scores_sum
        # else:
        #     loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum


        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """
        Return sum of left and right DFL losses.

        Distribution Focal Loss (DFL) proposed in Generalized Focal Loss
        https://ieeexplore.ieee.org/document/9792391
        """
        tl = target.long()  # target left yi
        tr = tl + 1  # target right  yi+1
        wl = tr - target  # weight left (yi+1-y)
        wr = 1 - wl  # weight right (y-yi)
        
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max, use_dfl)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, xywh2xyxy(target_bboxes[..., :4]), self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
        # -------- low-light uncertain negative suppression --------
        self.lowlight_enable = False
        self.lowlight_thr = 65.0         # RGB均值阈值，越低越暗
        self.lowlight_neg_lambda = 0.30  # 降权强度
        self.lowlight_neg_floor = 0.35   # 负样本最小权重，防止完全不学

        # =========================================================
        # Generic Frequency-MoE auxiliary loss / debug
        # =========================================================
        # MoE 与 HBB / OBB 的框类型无关，因此统一放在 DetectionLoss 父类。
        self.model = model
        self._init_moe_aux()

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        ## Cls loss
        ## loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        # loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE 这行是旧的

        ###################光照感知损失###########
        # Cls loss
        cls_loss_raw = F.binary_cross_entropy_with_logits(
            pred_scores,
            target_scores.to(dtype),
            reduction="none"
        )  # [B,N,C]
        neg_weight = self._build_lowlight_neg_weight(
            target_scores=target_scores,
            feats=feats,
            batch=batch
        )
        if neg_weight is not None:
            cls_loss_raw = cls_loss_raw * neg_weight  # 广播到 [B,N,C]

        if neg_weight is not None and torch.rand(1).item() < 0.01:
            affected_ratio = (neg_weight < 0.999).float().mean().item()
            print(
                f"[LowLightLoss] "
                f"neg_weight_mean={neg_weight.mean().item():.4f}, "
                f"neg_weight_min={neg_weight.min().item():.4f}, "
                f"neg_weight_max={neg_weight.max().item():.4f}"
                f"affected_ratio={affected_ratio:.4f}"
            )
        loss[1] = cls_loss_raw.sum() / target_scores_sum
        #######################################################################

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        # Generic Frequency-MoE auxiliary loss.
        # loss.detach() 仍只保留 box / cls / dfl，避免改变 trainer 日志格式。
        total_loss = loss.sum()
        total_loss = self._apply_moe_aux(total_loss)

        return total_loss * batch_size, loss.detach()  # loss(box, cls, dfl)
    
    def _rgb_lowlight_score(self, imgs):
        """
        imgs: [B,C,H,W]，默认前3通道是 RGB
        返回: [B,1,1]，范围 0~1
        0 表示正常亮度，1 表示非常暗
        """
        if imgs.shape[1] < 3:
            return None

        rgb = imgs[:, :3].float()
        mean_val = rgb.mean(dim=(1, 2, 3))

        # 兼容 0~1 或 0~255
        if mean_val.max() <= 1.5:
            mean_val = mean_val * 255.0

        score = (self.lowlight_thr - mean_val) / self.lowlight_thr
        score = score.clamp(0.0, 1.0)
        return score.view(-1, 1, 1)

    def _ir_anchor_prior(self, imgs, feats):
        """
        根据 IR 原图构建 anchor-level prior
        imgs:  [B,C,H,W]，默认后3通道是 IR
        feats: list，每层特征图 [B, C, h, w]

        返回:
            prior: [B, N, 1]
        """
        if imgs.shape[1] < 6:
            return None

        # 后3通道作为 IR，先转成单通道强度图
        ir = imgs[:, 3:6].float().mean(dim=1, keepdim=True)  # [B,1,H,W]

        # 每张图内部归一化到 0~1
        ir_min = ir.amin(dim=(2, 3), keepdim=True)
        ir_max = ir.amax(dim=(2, 3), keepdim=True)
        ir = (ir - ir_min) / (ir_max - ir_min + 1e-6)

        priors = []
        for feat in feats:
            h, w = feat.shape[2], feat.shape[3]
            p = F.adaptive_avg_pool2d(ir, (h, w))   # [B,1,h,w]
            p = p.flatten(2).transpose(1, 2)        # [B,hw,1]
            priors.append(p)

        return torch.cat(priors, dim=1)             # [B,N,1]

    def _build_lowlight_neg_weight(self, target_scores, feats, batch):
        """
        构建负样本降权矩阵，返回形状 [B,N,1]
        只对负样本位置生效；正样本权重恒为1
        """
        if (not self.lowlight_enable) or ("img" not in batch):
            return None

        imgs = batch["img"]
        if imgs.ndim != 4:
            return None

        lowlight_score = self._rgb_lowlight_score(imgs)   # [B,1,1]
        ir_prior = self._ir_anchor_prior(imgs, feats)     # [B,N,1]

        if lowlight_score is None or ir_prior is None:
            return None

        # target_scores 全0的位置视为负样本
        neg_mask = (target_scores.sum(-1, keepdim=True) <= 1e-6).float()  # [B,N,1]

        # 低光 + IR高响应处，降低“背景惩罚”
        neg_weight = 1.0 - self.lowlight_neg_lambda * lowlight_score * ir_prior
        # neg_weight = 1.0 - self.lowlight_neg_lambda * lowlight_score * (ir_prior ** 2)  # IR响应平方，增强高响应处的降权
        neg_weight = neg_weight.clamp(self.lowlight_neg_floor, 1.0)

        # 正样本保持1，只有负样本降权
        full_weight = 1.0 - neg_mask + neg_mask * neg_weight
        return full_weight


    # =============================================================
    # Frequency-MoE auxiliary loss / debug
    # =============================================================

    def _init_moe_aux(self):
        """
        Initialize generic Frequency-MoE balance/debug configuration.

        The MoE fusion mechanism is independent of HBB/OBB, so its loss and
        debug logic lives in v8DetectionLoss and is shared by v8OBBLoss.

        Segment/Pose inherit these helpers, but cache is only enabled for
        Detect/OBB heads to avoid retaining unused autograd graphs.
        """
        head_name = self.model.model[-1].__class__.__name__.lower()
        self.moe_task_enable = head_name in {"detect", "obb"}

        # ---------------- Balance config ----------------
        self.moe_balance_enable = False

        self.moe_ll_balance_gain = float(
            os.getenv("MOE_LL_BALANCE_GAIN", "0.01")
        )
        self.moe_hf_balance_gain = float(
            os.getenv("MOE_HF_BALANCE_GAIN", "0.01")
        )

        # Legacy module fallback.
        self.moe_balance_gain = float(
            os.getenv("MOE_BALANCE_GAIN", "0.01")
        )

        # ---------------- CSV debug config ----------------
        self.moe_debug_enable = bool(
            int(os.getenv("MOE_DEBUG_ENABLE", "1"))
        )
        self.moe_debug_interval = int(
            os.getenv("MOE_DEBUG_INTERVAL", "200")
        )
        self.moe_debug_dir = Path(
            os.getenv("MOE_DEBUG_DIR", "runs/moe_debug")
        )
        self.moe_exp_name = os.getenv(
            "MOE_EXP_NAME", "default"
        )

        # Parallel experiments must never append to one folder.
        self.moe_debug_run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + f"_pid{os.getpid()}"
        )
        self._moe_debug_step = 0

        # MoE modules cache router/balance tensors during model forward.
        if self.moe_task_enable:
            cache_enable = bool(
                self.moe_balance_enable or self.moe_debug_enable
            )
            for module in self.model.modules():
                if hasattr(module, "cache_moe_aux"):
                    module.cache_moe_aux = cache_enable

    def _collect_moe_aux(self):
        """
        Collect cached band-specific balance terms and router debug.

        New modules:
            last_moe_balance_terms = {ll, lh, hl, hh}

        Same-band terms from P3/P4/P5 are averaged.

        Legacy modules:
            last_moe_balance_loss
        """
        band_terms = {}
        fallback_terms = []
        debug_rows = []

        for module_idx, module in enumerate(self.model.modules()):
            terms = getattr(
                module, "last_moe_balance_terms", None
            )

            if isinstance(terms, dict):
                for band, value in terms.items():
                    if not isinstance(value, torch.Tensor):
                        continue
                    band = str(band).lower()
                    band_terms.setdefault(band, []).append(
                        value.reshape(())
                    )
            else:
                fallback = getattr(
                    module, "last_moe_balance_loss", None
                )
                if isinstance(fallback, torch.Tensor):
                    fallback_terms.append(
                        fallback.reshape(())
                    )

            stats = getattr(
                module, "last_moe_debug_stats", None
            )
            if isinstance(stats, (list, tuple)):
                for item in stats:
                    if not isinstance(item, dict):
                        continue

                    row = dict(item)
                    row.setdefault("module_idx", module_idx)
                    row.setdefault(
                        "module_name",
                        module.__class__.__name__,
                    )
                    row.setdefault(
                        "variant",
                        getattr(
                            module,
                            "moe_variant",
                            "unknown",
                        ),
                    )
                    debug_rows.append(row)

        band_mean = {
            band: torch.stack(values).mean()
            for band, values in band_terms.items()
            if values
        }

        fallback_mean = (
            torch.stack(fallback_terms).mean()
            if fallback_terms
            else None
        )

        return band_mean, fallback_mean, debug_rows

    def _clear_moe_aux_cache(self):
        """Clear cached tensors attached to the current autograd graph."""
        for module in self.model.modules():
            if hasattr(module, "last_moe_balance_loss"):
                module.last_moe_balance_loss = None
            if hasattr(module, "last_moe_balance_terms"):
                module.last_moe_balance_terms = None
            if hasattr(module, "last_moe_debug_stats"):
                module.last_moe_debug_stats = None

    @staticmethod
    def _moe_to_float(x, default=0.0):
        if isinstance(x, torch.Tensor):
            if x.numel() == 0:
                return float(default)
            return float(
                x.detach().float().mean().cpu()
            )
        if x is None:
            return float(default)
        return float(x)

    @staticmethod
    def _moe_to_list(x):
        if isinstance(x, torch.Tensor):
            return (
                x.detach().float().flatten().cpu().tolist()
            )
        if isinstance(x, np.ndarray):
            return x.reshape(-1).astype(float).tolist()
        if isinstance(x, (list, tuple)):
            return [float(v) for v in x]
        return []

    def _apply_moe_aux(self, total_loss):
        """
        Add MoE balance loss and write debug for both HBB and OBB.

        Full-independent/V2 normalization is intentionally kept identical
        to the old OBB implementation:

            0.25 * [
                lambda_LL * LL
                + lambda_HF * (LH + HL + HH)
            ]

        Therefore HF=0 does not accidentally make the LL auxiliary term
        four times stronger.
        """
        if not self.moe_task_enable:
            return total_loss

        try:
            (
                band_losses,
                fallback_loss,
                debug_rows,
            ) = self._collect_moe_aux()

            raw_moe_loss = None
            weighted_moe_loss = None

            if band_losses:
                zero = next(
                    iter(band_losses.values())
                ).new_tensor(0.0)

                l_ll = band_losses.get("ll", zero)
                l_lh = band_losses.get("lh", zero)
                l_hl = band_losses.get("hl", zero)
                l_hh = band_losses.get("hh", zero)

                # Raw four-band mean: debug only.
                raw_moe_loss = 0.25 * (
                    l_ll + l_lh + l_hl + l_hh
                )

                # Actual optimization term.
                weighted_moe_loss = 0.25 * (
                    self.moe_ll_balance_gain * l_ll
                    + self.moe_hf_balance_gain
                    * (l_lh + l_hl + l_hh)
                )

            elif fallback_loss is not None:
                raw_moe_loss = fallback_loss
                weighted_moe_loss = (
                    self.moe_balance_gain * fallback_loss
                )

            if (
                self.model.training
                and weighted_moe_loss is not None
            ):
                total_loss = total_loss + weighted_moe_loss

            # Validation / EMA do not create another debug stream.
            if self.model.training:
                self._maybe_write_moe_debug(
                    debug_rows=debug_rows,
                    raw_moe_balance_loss=raw_moe_loss,
                    weighted_moe_balance_loss=weighted_moe_loss,
                )

            return total_loss

        finally:
            self._clear_moe_aux_cache()

    def _maybe_write_moe_debug(
        self,
        debug_rows,
        raw_moe_balance_loss=None,
        weighted_moe_balance_loss=None,
    ):
        """
        Write MoE routing statistics to CSV.

        Directory:
            runs/moe_debug/<variant>/<MOE_EXP_NAME>/<run_id>/
                router_stats.csv
                balance_loss.csv

        balance_loss.csv:
            balance_loss
                raw unweighted four-band mean

            weighted_balance_loss
                the ACTUAL auxiliary value added to total loss

        The old file multiplied an already weighted loss by 0.01 again only
        while writing CSV. That bookkeeping bug is corrected here; it never
        affected optimization.
        """
        if not self.moe_debug_enable:
            return

        rank = int(os.environ.get("RANK", "0"))
        if rank not in (-1, 0):
            return

        self._moe_debug_step += 1
        if self._moe_debug_step % self.moe_debug_interval != 0:
            return

        if (
            not debug_rows
            and raw_moe_balance_loss is None
            and weighted_moe_balance_loss is None
        ):
            return

        variant = "unknown"
        if debug_rows:
            variant = str(
                debug_rows[0].get("variant", "unknown")
            )

        variant = (
            variant.replace("/", "_")
            .replace("\\", "_")
            .replace(" ", "_")
        )

        save_dir = (
            self.moe_debug_dir
            / variant
            / self.moe_exp_name
            / self.moe_debug_run_id
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        # ---------------- router_stats.csv ----------------
        if debug_rows:
            router_file = save_dir / "router_stats.csv"
            router_exists = router_file.exists()

            fieldnames = [
                "step",
                "variant",
                "module_idx",
                "module_name",
                "bank",
                "route",
                "feature_h",
                "feature_w",
                "expert_id",
                "prob_mean",
                "prob_std",
                "selection_rate",
                "selected_weight_mean",
                "router_entropy",
                "top2_top3_margin",
                "pair_01",
                "pair_02",
                "pair_03",
                "pair_12",
                "pair_13",
                "pair_23",
                "balance_loss",
            ]

            with router_file.open(
                "a", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(
                    f, fieldnames=fieldnames
                )
                if not router_exists:
                    writer.writeheader()

                for row in debug_rows:
                    probs = self._moe_to_list(
                        row.get("prob_mean")
                    )
                    prob_stds = self._moe_to_list(
                        row.get("prob_std")
                    )
                    sels = self._moe_to_list(
                        row.get("selection_rate")
                    )
                    weights = self._moe_to_list(
                        row.get("selected_weight_mean")
                    )

                    margin = self._moe_to_float(
                        row.get("top2_top3_margin")
                    )
                    entropy = self._moe_to_float(
                        row.get("entropy")
                    )
                    balance = self._moe_to_float(
                        row.get("balance_loss")
                    )

                    pair_values = {
                        key: self._moe_to_float(
                            row.get(key)
                        )
                        for key in (
                            "pair_01",
                            "pair_02",
                            "pair_03",
                            "pair_12",
                            "pair_13",
                            "pair_23",
                        )
                    }

                    n_experts = max(
                        len(probs),
                        len(prob_stds),
                        len(sels),
                        len(weights),
                    )

                    for expert_id in range(n_experts):
                        writer.writerow(
                            {
                                "step":
                                    self._moe_debug_step,
                                "variant":
                                    row.get(
                                        "variant", variant
                                    ),
                                "module_idx":
                                    row.get(
                                        "module_idx", -1
                                    ),
                                "module_name":
                                    row.get(
                                        "module_name", ""
                                    ),
                                "bank":
                                    row.get("bank", ""),
                                "route":
                                    row.get("route", ""),
                                "feature_h":
                                    row.get(
                                        "feature_h", -1
                                    ),
                                "feature_w":
                                    row.get(
                                        "feature_w", -1
                                    ),
                                "expert_id":
                                    expert_id,
                                "prob_mean":
                                    probs[expert_id]
                                    if expert_id < len(probs)
                                    else 0.0,
                                "prob_std":
                                    prob_stds[expert_id]
                                    if expert_id < len(prob_stds)
                                    else 0.0,
                                "selection_rate":
                                    sels[expert_id]
                                    if expert_id < len(sels)
                                    else 0.0,
                                "selected_weight_mean":
                                    weights[expert_id]
                                    if expert_id < len(weights)
                                    else 0.0,
                                "router_entropy":
                                    entropy,
                                "top2_top3_margin":
                                    margin,
                                "pair_01":
                                    pair_values["pair_01"],
                                "pair_02":
                                    pair_values["pair_02"],
                                "pair_03":
                                    pair_values["pair_03"],
                                "pair_12":
                                    pair_values["pair_12"],
                                "pair_13":
                                    pair_values["pair_13"],
                                "pair_23":
                                    pair_values["pair_23"],
                                "balance_loss":
                                    balance,
                            }
                        )

        # ---------------- balance_loss.csv ----------------
        if (
            raw_moe_balance_loss is not None
            or weighted_moe_balance_loss is not None
        ):
            balance_file = save_dir / "balance_loss.csv"
            balance_exists = balance_file.exists()

            with balance_file.open(
                "a", newline="", encoding="utf-8"
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        "variant",
                        "balance_loss",
                        "weighted_balance_loss",
                    ],
                )
                if not balance_exists:
                    writer.writeheader()

                writer.writerow(
                    {
                        "step":
                            self._moe_debug_step,
                        "variant":
                            variant,
                        "balance_loss":
                            self._moe_to_float(
                                raw_moe_balance_loss
                            ),
                        "weighted_balance_loss":
                            self._moe_to_float(
                                weighted_moe_balance_loss
                            ),
                    }
                )



class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n"
                "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-seg.pt data=coco8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'segment' dataset using 'data=coco8-seg.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/segment/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )
            # Masks loss
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(
        gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor, area: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(
        self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
    ):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch["cls"], reduction="mean")
        loss_items = loss.detach()
        return loss, loss_items


class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model):
        """
        Initialize OBB-specific criterion components.

        Frequency-MoE auxiliary loss/debug is inherited from
        v8DetectionLoss and is no longer duplicated here.
        """
        super().__init__(model)

        self.assigner = RotatedTaskAlignedAssigner(
            topk=10,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
        )
        self.bbox_loss = RotatedBboxLoss(
            self.reg_max - 1,
            use_dfl=self.use_dfl,
        ).to(self.device)

        # ---------------- OBB-only mask auxiliary branch ----------------
        self.mask_loss_gain = 0.05
        self.mask_eta = 0.35
        self.mask_loss_enable = False

        self.mask_vis_enable = False
        self.mask_vis_interval = 5000
        self.mask_vis_dir = Path("runs/maskv1_debug")
        self.mask_vis_max_layers = 3
        self._mask_vis_step = 0
        self.mask_debug_print = False

        for module in self.model.modules():
            if hasattr(module, "cache_aux_mask_logits"):
                module.cache_aux_mask_logits = bool(
                    self.mask_loss_enable
                )

    def _clear_aux_mask_logits(self):
        for m in self.model.modules():
            if hasattr(m, "last_mask_logits"):
                m.last_mask_logits = None
    #收集 LASCIModule 缓存的 mask_logits
    def _collect_aux_mask_logits(self):
        aux = []

        for m in self.model.modules():
            mask_logits = getattr(m, "last_mask_logits", None)
            if isinstance(mask_logits, torch.Tensor):
                aux.append(mask_logits)

        return aux
    
    # 用 OBB 标签生成 rotated Gaussian mask
    def _build_rotated_gaussian_masks(self, batch, mask_logits):
        """
        Args:
            batch: YOLO batch dict, contains batch_idx and bboxes [N,5]
            mask_logits: list of [B,1,H,W]

        Returns:
            targets: list of [B,1,H,W]
        """
        targets = []

        batch_idx = batch["batch_idx"].view(-1).long().to(self.device)
        bboxes = batch["bboxes"].view(-1, 5).to(self.device)
        # if torch.rand(1).item() < 0.001 and bboxes.numel():
        #     print(
        #         "[OBB batch bboxes]",
        #         "shape=", tuple(bboxes.shape),
        #         "first=", bboxes[0].detach().cpu().tolist(),
        #         "angle_min=", float(bboxes[:, 4].min().detach().cpu()),
        #         "angle_max=", float(bboxes[:, 4].max().detach().cpu()),
        #         "angle_mean=", float(bboxes[:, 4].mean().detach().cpu()),
        #     )

        for logits in mask_logits:
            b, _, h, w = logits.shape
            target = torch.zeros((b, 1, h, w), device=logits.device, dtype=torch.float32)

            # grid: [H,W]
            ys = torch.arange(h, device=logits.device, dtype=torch.float32)
            xs = torch.arange(w, device=logits.device, dtype=torch.float32)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")

            for i in range(bboxes.shape[0]):
                bi = int(batch_idx[i].item())
                if bi < 0 or bi >= b:
                    continue

                cx, cy, bw, bh, angle = bboxes[i]

                # normalized -> feature coordinates
                cx = cx * w
                cy = cy * h
                bw = bw * w
                bh = bh * h

                # 过滤极小框，和 OBB loss 的稳定性逻辑一致
                if bw < 1.0 or bh < 1.0:
                    continue

                cos_a = torch.cos(angle)
                sin_a = torch.sin(angle)

                dx = xx - cx
                dy = yy - cy

                # rotate to box local coordinates
                u = cos_a * dx + sin_a * dy
                v = -sin_a * dx + cos_a * dy

                sigma_w = torch.clamp(self.mask_eta * bw, min=1.0)
                sigma_h = torch.clamp(self.mask_eta * bh, min=1.0)

                g = torch.exp(
                    -0.5 * ((u / sigma_w) ** 2 + (v / sigma_h) ** 2)
                )

                target[bi, 0] = torch.maximum(target[bi, 0], g)

            targets.append(target)

        return targets
    # 计算 mask loss
    def _dice_loss(self, pred_prob, target, eps=1e-6):
        """
        pred_prob, target: [B,1,H,W]
        """
        pred = pred_prob.flatten(1)
        tgt = target.flatten(1)

        inter = (pred * tgt).sum(dim=1)
        denom = pred.sum(dim=1) + tgt.sum(dim=1)

        loss = 1.0 - (2.0 * inter + eps) / (denom + eps)
        return loss.mean()
    def _compute_aux_mask_loss(self, batch):
        mask_logits = self._collect_aux_mask_logits()
        # if torch.rand(1).item() < 0.001:
        #     print("[AuxMask] num_layers =", len(mask_logits), 
        #         [tuple(x.shape) for x in mask_logits])

        if len(mask_logits) == 0:
            return None

        targets = self._build_rotated_gaussian_masks(batch, mask_logits)

        total = 0.0
        valid = 0

        for logits, target in zip(mask_logits, targets):
            logits_f = logits.float()
            target = target.to(device=logits.device, dtype=torch.float32)

            bce = F.binary_cross_entropy_with_logits(logits_f, target, reduction="mean")
            prob = torch.sigmoid(logits_f)
            dice = self._dice_loss(prob, target)

            total = total + bce + dice
            valid += 1

        if valid == 0:
            return None

        aux_loss = total / valid

        self._maybe_visualize_aux_masks(
            mask_logits=mask_logits,
            targets=targets,
            aux_mask_loss=aux_loss,
        )

        self._clear_aux_mask_logits()
        return aux_loss

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 6, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 6, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    bboxes = targets[matches, 2:]
                    bboxes[..., :4].mul_(scale_tensor)
                    out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)
        return out

    @staticmethod
    def _to_uint8_map(x):
        """
        x: [H,W] tensor
        return: uint8 numpy [H,W]
        """
        # x = x.detach().float().cpu()
        # x_min = x.min()
        # x_max = x.max()
        # x = (x - x_min) / (x_max - x_min + 1e-6)
        # return (x.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        x = x.detach().float().cpu()
        x = x.clamp(0.0, 1.0)
        return (x.numpy() * 255.0).astype(np.uint8)

    def _collect_mask_alphas(self):
        alphas = []
        for m in self.model.modules():
            if hasattr(m, "mask_alpha_raw") and hasattr(m, "mask_alpha_max"):
                alpha = m.mask_alpha_max * torch.sigmoid(m.mask_alpha_raw.detach())
                alphas.append(float(alpha.cpu()))

        return alphas
    def _maybe_visualize_aux_masks(self, mask_logits, targets, aux_mask_loss=None):
        """
        保存每层的:
        pred mask / target mask / abs error
        """
        if not self.mask_vis_enable:
            return

        # DDP 下只让 rank0 保存，避免多卡重复写
        rank = int(os.environ.get("RANK", "0"))
        if rank not in (-1, 0):
            return

        self._mask_vis_step += 1

        if self._mask_vis_step % self.mask_vis_interval != 0:
            return

        self.mask_vis_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            msg = [f"[MaskVis] step={self._mask_vis_step}"]

            if aux_mask_loss is not None:
                msg.append(f"loss={float(aux_mask_loss.detach().cpu()):.4f}")
            alphas = self._collect_mask_alphas()
            if len(alphas):
                msg.append("alpha=" + ",".join([f"{a:.5f}" for a in alphas]))
            for li, (logits, target) in enumerate(zip(mask_logits, targets)):
                if li >= self.mask_vis_max_layers:
                    break

                pred = torch.sigmoid(logits[0, 0])
                tgt = target[0, 0].to(device=pred.device, dtype=pred.dtype)
                err = (pred - tgt).abs()

                pred_u8 = self._to_uint8_map(pred)
                tgt_u8 = self._to_uint8_map(tgt)
                err_u8 = self._to_uint8_map(err)

                # 横向拼接：pred | target | error
                canvas = np.concatenate([pred_u8, tgt_u8, err_u8], axis=1)

                save_path = self.mask_vis_dir / f"step_{self._mask_vis_step:07d}_layer_{li}.png"
                Image.fromarray(canvas).save(save_path)

                msg.append(
                    f"L{li}: "
                    f"pred_mean={pred.mean().item():.4f}, "
                    f"pred_max={pred.max().item():.4f}, "
                    f"tgt_mean={tgt.mean().item():.4f}, "
                    f"alpha=?"
                )

            print(" | ".join(msg))


    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        # b, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # targets
        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1)
            rw, rh = targets[:, 4] * imgsz[0].item(), targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]  # filter rboxes of tiny size to stabilize training
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)  # cls, xywhr
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError(
                "ERROR ❌ OBB dataset incorrectly formatted or not a OBB dataset.\n"
                "This error can occur when incorrectly training a 'OBB' model on a 'detect' dataset, "
                "i.e. 'yolo train model=yolov8n-obb.pt data=dota8.yaml'.\nVerify your dataset is a "
                "correctly formatted 'OBB' dataset using 'data=dota8.yaml' "
                "as an example.\nSee https://docs.ultralytics.com/datasets/obb/ for help."
            ) from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)  # xyxy, (b, h*w, 4)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        # Only the first four elements need to be scaled
        bboxes_for_assigner[..., :4] *= stride_tensor
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        total_loss = loss.sum()

        if self.mask_loss_enable:
            aux_mask_loss = self._compute_aux_mask_loss(batch)
            if aux_mask_loss is not None:
                total_loss = total_loss + self.mask_loss_gain * aux_mask_loss

        # Generic Frequency-MoE auxiliary loss/debug inherited from parent.
        total_loss = self._apply_moe_aux(
            total_loss
        )
        return total_loss * batch_size, loss.detach()  # loss(box, cls, dfl)

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        """
        Decode predicted object bounding box coordinates from anchor points and distribution.

        Args:
            anchor_points (torch.Tensor): Anchor points, (h*w, 2).
            pred_dist (torch.Tensor): Predicted rotated distance, (bs, h*w, 4).
            pred_angle (torch.Tensor): Predicted angle, (bs, h*w, 1).

        Returns:
            (torch.Tensor): Predicted rotated bounding boxes with angles, (bs, h*w, 5).
        """
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)