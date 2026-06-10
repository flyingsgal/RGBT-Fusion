# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import argparse
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


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
    if not path.exists():
        raise FileNotFoundError(f"{desc} 不存在: {path}")
    return str(path)


def auto_device(device_arg: str):
    if device_arg is not None and str(device_arg).strip() != "":
        return device_arg
    return "0" if torch.cuda.is_available() else "cpu"


def print_env_info(args):
    print("=" * 80)
    print("[INFO] Training Configuration")
    print(f"model        : {args.model}")
    print(f"data         : {args.data}")
    print(f"epochs       : {args.epochs}")
    print(f"imgsz        : {args.imgsz}")
    print(f"batch        : {args.batch}")
    print(f"device       : {args.device}")
    print(f"workers      : {args.workers}")
    print(f"project      : {args.project}")
    print(f"name         : {args.name}")
    print(f"resume       : {args.resume}")
    print(f"pretrained   : {args.pretrained}")
    print(f"optimizer    : {args.optimizer}")
    print(f"lr0          : {args.lr0}")
    print(f"patience     : {args.patience}")
    print(f"amp          : {args.amp}")
    print(f"cache        : {args.cache}")
    print(f"seed         : {args.seed}")
    print("=" * 80)

    print("[INFO] Environment")
    print(f"Python       : {sys.version.split()[0]}")
    print(f"PyTorch      : {torch.__version__}")
    print(f"CUDA Avail   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device  : {torch.cuda.get_device_name(0)}")
        print(f"CUDA Count   : {torch.cuda.device_count()}")
    print("=" * 80)


def build_model(model_path: str):
    suffix = Path(model_path).suffix.lower()
    if suffix in [".yaml", ".yml", ".pt"]:
        return YOLO(model_path)
    raise ValueError(f"不支持的 model 文件类型: {model_path}")


def train_once(args, batch_size):
    model = build_model(args.model)


    def print_dssf_gamma(trainer):
        print("\n[DSSF gamma]")
        for name, m in trainer.model.named_modules():
            if m.__class__.__name__ == "DSSF_SS2D":
                print(f"{name}: gamma = {m.gamma.detach().cpu().item():.6f}")
    try:
        model.add_callback("on_train_epoch_end", print_dssf_gamma)
    except AttributeError:
        model.callbacks["on_train_epoch_end"].append(print_dssf_gamma)

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
        pretrained=args.pretrained,
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

    # 可选参数：resume
    if args.resume:
        train_kwargs["resume"] = True

    # 可选参数：单独控制保存周期
    if args.save_period > 0:
        train_kwargs["save_period"] = args.save_period

    # 可选参数：冻结层
    if args.freeze is not None and args.freeze >= 0:
        train_kwargs["freeze"] = args.freeze

    # 可选参数：矩形训练
    if args.rect:
        train_kwargs["rect"] = True

    # 可选参数：多尺度
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

        except torch.cuda.OutOfMemoryError as e:
            last_error = e
            print(f"[WARN] CUDA OOM，当前 batch={batch} 失败，尝试减半重试...")
            torch.cuda.empty_cache()
            if batch == min_batch:
                break
            batch = max(batch // 2, min_batch)

        except KeyboardInterrupt:
            print("\n[WARN] 用户手动中断训练。")
            raise

        except Exception as e:
            last_error = e
            print(f"[ERROR] 训练失败: {repr(e)}")
            raise

    raise RuntimeError(f"训练最终失败，最后错误: {repr(last_error)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Robust Ultralytics YOLO Training Script")

    parser.add_argument("--model", type=str,
                        default="/storage/jyx4/projects/TwoStream_Yolov8-main/yaml/new_ACM/LASCI_DSSF.yaml",
                        help="模型结构 yaml 或权重 pt 路径")
    parser.add_argument("--data", type=str,
                        default="/storage/jyx4/projects/TwoStream_Yolov8-main/data/dronevehicle_obb.yaml",
                        help="数据集 yaml 路径")

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--project", type=str, default="runs/dronevehicle/obb")
    parser.add_argument("--name", type=str, default="LASCI_DSSF")
    parser.add_argument("--exist-ok", action="store_true")

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pretrained", type=bool, default=True)
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
    parser.add_argument("--freeze", type=int, default=-1)
    parser.add_argument("--rect", action="store_true")
    parser.add_argument("--multi-scale", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    args.model = check_file_exists(args.model, "模型文件")
    args.data = check_file_exists(args.data, "数据集配置文件")
    args.device = auto_device(args.device)

    if args.name.strip() == "":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_stem = Path(args.model).stem
        data_stem = Path(args.data).stem
        args.name = f"{model_stem}_{data_stem}_{timestamp}"

    Path(args.project).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed, deterministic=args.deterministic)
    print_env_info(args)

    try:
        results = robust_train(args)
        print("[INFO] 训练成功结束。")
        print(f"[INFO] 结果保存目录: {Path(args.project) / args.name}")
        return results
    except Exception as e:
        print("=" * 80)
        print(f"[FATAL] 训练脚本退出，错误信息: {repr(e)}")
        print("=" * 80)
        raise


if __name__ == "__main__":
    main()