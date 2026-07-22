# -*- coding: utf-8 -*-
"""Train the P3 IR-guided selective-offset upper-bound experiment.

Place this file and projection_loader.py in the repository root, then run it
from that root. The IRGuidedSelectiveOffset class must already be registered in
the local Ultralytics fork.
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.block import IRGuidedSelectiveOffset

from projection_loader import load_projection_into_yolo


def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def check_file_exists(path_str: str, desc: str):
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{desc} 不存在: {path}")
    return str(path)


def auto_device(device_arg: str):
    if device_arg is not None and str(device_arg).strip() != "":
        return device_arg
    return "0" if torch.cuda.is_available() else "cpu"


def print_env_info(args):
    print("=" * 80)
    print("[INFO] Selective-offset upper-bound configuration")
    print(f"model yaml        : {args.model}")
    print(f"initial Add pt    : {args.weights}")
    print(f"projection pt     : {args.projection}")
    print(f"route mode        : {args.route_mode}")
    print(f"data              : {args.data}")
    print(f"epochs            : {args.epochs}")
    print(f"imgsz             : {args.imgsz}")
    print(f"batch             : {args.batch}")
    print(f"freeze            : {args.freeze}")
    print(f"device            : {args.device}")
    print(f"workers           : {args.workers}")
    print(f"project           : {args.project}")
    print(f"name              : {args.name}")
    print(f"optimizer         : {args.optimizer}")
    print(f"lr0               : {args.lr0}")
    print(f"patience          : {args.patience}")
    print(f"amp               : {args.amp}")
    print(f"cache             : {args.cache}")
    print(f"seed              : {args.seed}")
    print("=" * 80)
    print("[INFO] Environment")
    print(f"Python            : {sys.version.split()[0]}")
    print(f"PyTorch           : {torch.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device       : {torch.cuda.get_device_name(0)}")
        print(f"CUDA count        : {torch.cuda.device_count()}")
    print("=" * 80)


def _projection_state(module):
    """Clone only the two common-space projections for later verification."""
    return {
        name: value.detach().float().cpu().clone()
        for name, value in module.state_dict().items()
        if name.startswith("rgb_proj.") or name.startswith("ir_proj.")
    }


def train_once(args, batch_size):
    # Build the new graph first. Constructing YOLO(args.weights) would restore
    # the old Add graph and therefore cannot create the offset module.
    model = YOLO(args.model)

    # Transfer the Add baseline's backbone, neck, and head weights into the new
    # graph. The new offset module has no counterpart in the Add checkpoint.
    model.load(args.weights)

    # Load the separately trained P3 common-space projection into the new
    # offset module before Ultralytics creates its Trainer.
    offset_modules = load_projection_into_yolo(
        model,
        args.projection,
        freeze=True,
        route_mode=args.route_mode,
    )
    if len(offset_modules) != 1:
        raise RuntimeError(
            f"应找到一个 IRGuidedSelectiveOffset，实际找到 {len(offset_modules)} 个"
        )

    offset_module = offset_modules[0]
    expected_projection = _projection_state(offset_module)
    print(
        "[Projection injected before training]",
        f"c={offset_module.c}",
        f"embed_dim={offset_module.embed_dim}",
        f"route_mode={offset_module.route_mode}",
        "trainable={}".format(
            any(
                parameter.requires_grad
                for parameter in list(offset_module.rgb_proj.parameters())
                + list(offset_module.ir_proj.parameters())
            )
        ),
    )

    def verify_and_lock_projection_on_train_start(trainer):
        """Lock both the optimization model and the EMA validation model.

        Ultralytics validates and writes best.pt from EMA. Submodule string
        attributes such as route_mode are not updated by EMA tensor updates, so
        configuring trainer.model alone would make a center run validate with
        the YAML's default route mode.
        """

        targets = [("trainer.model", trainer.model)]
        ema_wrapper = getattr(trainer, "ema", None)
        ema_model = getattr(ema_wrapper, "ema", None)
        if ema_model is not None:
            targets.append(("trainer.ema.ema", ema_model))

        for target_name, target_model in targets:
            modules = [
                module
                for module in target_model.modules()
                if isinstance(module, IRGuidedSelectiveOffset)
            ]
            if len(modules) != 1:
                raise RuntimeError(
                    "{} 中应有一个 IRGuidedSelectiveOffset，实际为 {}".format(
                        target_name, len(modules)
                    )
                )

            current_module = modules[0]
            current_state = current_module.state_dict()
            max_difference = 0.0
            for name, expected in expected_projection.items():
                if name not in current_state:
                    raise RuntimeError(
                        f"{target_name} 中缺少 projection 参数: {name}"
                    )
                difference = (
                    current_state[name].detach().float().cpu() - expected
                ).abs().max().item()
                max_difference = max(max_difference, difference)

            if max_difference >= 1e-7:
                raise RuntimeError(
                    f"{target_name} 中 projection 权重发生变化，"
                    f"max_diff={max_difference:.8g}"
                )

            current_module.set_route_mode(args.route_mode)
            current_module.set_projection_trainable(False)
            projection_trainable = any(
                parameter.requires_grad
                for parameter in list(current_module.rgb_proj.parameters())
                + list(current_module.ir_proj.parameters())
            )
            print(
                "[Projection train-start check]",
                f"target={target_name}",
                f"max_diff={max_difference:.8g}",
                f"route_mode={current_module.route_mode}",
                f"trainable={projection_trainable}",
            )
            if projection_trainable:
                raise RuntimeError(
                    f"{target_name} 的 projection 在训练开始时仍为可训练状态"
                )

        if ema_model is None:
            print("[WARN] Trainer 未创建 EMA；验证将直接使用 trainer.model。")

    model.add_callback(
        "on_train_start",
        verify_and_lock_projection_on_train_start,
    )

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch_size,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        # Add best.pt was loaded explicitly above. Do not request a second
        # implicit pretrained source from the Trainer.
        pretrained=False,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        cache=args.cache,
        amp=args.amp,
        seed=args.seed,
        cos_lr=args.cos_lr,
        close_mosaic=args.close_mosaic,
        verbose=True,
        plots=True,
        save=True,
        val=True,
    )

    if args.save_period > 0:
        train_kwargs["save_period"] = args.save_period
    if args.freeze is not None and args.freeze >= 0:
        train_kwargs["freeze"] = args.freeze
    if args.rect:
        train_kwargs["rect"] = True
    if args.multi_scale:
        train_kwargs["multi_scale"] = True

    return model.train(**train_kwargs)


def robust_train(args):
    batch = args.batch
    min_batch = 1
    last_error = None

    while batch >= min_batch:
        try:
            print(f"[INFO] 开始训练，当前 batch={batch}")
            results = train_once(args, batch)
            print(f"[INFO] 训练完成，最终 batch={batch}")
            return results
        except torch.cuda.OutOfMemoryError as error:
            last_error = error
            print(f"[WARN] CUDA OOM，当前 batch={batch} 失败，尝试减半重试...")
            torch.cuda.empty_cache()
            if batch == min_batch:
                break
            batch = max(batch // 2, min_batch)
        except KeyboardInterrupt:
            print("\n[WARN] 用户手动中断训练。")
            raise
        except Exception as error:
            last_error = error
            print(f"[ERROR] 训练失败: {repr(error)}")
            raise

    raise RuntimeError(f"训练最终失败，最后错误: {repr(last_error)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="P3 IR-guided selective-offset upper-bound experiment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=(
            "/storage/jyx4/projects/TwoStream_Yolov8-main/"
            "yaml/offset/IRGuidedSelectiveOffset_P3.yaml"
        ),
        help="包含 IRGuidedSelectiveOffset 的新模型 YAML",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=(
            "/storage/jyx4/projects/TwoStream_Yolov8-main/"
            "runs/dronevehicle/obb/base_add/weights/best.pt"
        ),
        help="同配置 Add baseline 的 best.pt，用作统一初始化",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default=(
            "/storage/jyx4/projects/TwoStream_Yolov8-main/"
            "runs/p3_common_space_probe/first_run/projection_probe.pt"
        ),
        help="train_p3_common_space_probe.py 生成的 projection_probe.pt",
    )
    parser.add_argument(
        "--route-mode",
        type=str,
        choices=("center", "reliable", "all"),
        default="reliable",
        help="首轮正式比较只运行 center 和 reliable",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=(
            "/storage/jyx4/projects/TwoStream_Yolov8-main/"
            "data/dronevehicle_obb.yaml"
        ),
    )

    # This is a short upper-bound screening experiment, not the final 120-epoch
    # ablation. Keep the feature extractor frozen to avoid projection drift.
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--freeze", type=int, default=23)

    parser.add_argument("--project", type=str, default="runs/dronevehicle/obb")
    parser.add_argument(
        "--name",
        type=str,
        default="offset_reliable_p3_e10_freeze23_s42",
    )
    parser.add_argument("--exist-ok", action="store_true")

    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--amp", type=bool, default=True)
    parser.add_argument("--cache", type=str, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--cos-lr", action="store_true")
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--save-period", type=int, default=-1)
    parser.add_argument("--rect", action="store_true")
    parser.add_argument("--multi-scale", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.model = check_file_exists(args.model, "offset 模型 YAML")
    args.weights = check_file_exists(args.weights, "Add baseline 权重")
    args.projection = check_file_exists(args.projection, "projection 权重")
    args.data = check_file_exists(args.data, "数据集配置文件")
    args.device = auto_device(args.device)

    if args.name.strip() == "":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.name = (
            f"offset_{args.route_mode}_e{args.epochs}_"
            f"freeze{args.freeze}_s{args.seed}_{timestamp}"
        )

    Path(args.project).mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, deterministic=args.deterministic)
    print_env_info(args)

    try:
        results = robust_train(args)
        print("[INFO] 训练成功结束。")
        print(f"[INFO] 结果保存目录: {Path(args.project) / args.name}")
        return results
    except Exception as error:
        print("=" * 80)
        print(f"[FATAL] 训练脚本退出，错误信息: {repr(error)}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()
