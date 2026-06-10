# -*- coding: utf-8 -*-
import os
import yaml
import argparse
import tempfile
from pathlib import Path

from ultralytics import YOLO


def load_yaml(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data, yaml_path):
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def resolve_path(root, value):
    """把相对路径转成绝对路径"""
    if value is None:
        return None

    if isinstance(value, str):
        p = Path(value)
        if not p.is_absolute():
            p = (root / p).resolve()
        return str(p)

    if isinstance(value, list):
        result = []
        for x in value:
            p = Path(x)
            if not p.is_absolute():
                p = (root / p).resolve()
            result.append(str(p))
        return result

    return value


def path_exists(x):
    if x is None:
        return False
    if isinstance(x, str):
        return Path(x).exists()
    if isinstance(x, list):
        return all(Path(p).exists() for p in x)
    return False


def normalize_dataset_yaml(data_yaml, split):
    """
    读取并修正双流数据集 yaml：
    1. 所有 train/val/test/train_ir/val_ir/test_ir 转绝对路径
    2. 检查 split 对应路径是否存在
    """
    data = load_yaml(data_yaml)

    if "path" not in data:
        raise ValueError(f"数据集 yaml 缺少 path 字段: {data_yaml}")

    root = Path(data["path"]).expanduser().resolve()
    data["path"] = str(root)

    for key in ["train", "val", "test", "train_ir", "val_ir", "test_ir"]:
        if key in data:
            data[key] = resolve_path(root, data[key])

    # 基础必需项
    for key in ["train", "val", "train_ir", "val_ir"]:
        if key not in data or data[key] is None:
            raise ValueError(f"数据集 yaml 缺少必需字段: {key}")

    # 根据 split 选择使用哪个集合
    if split == "test":
        rgb_key = "test" if data.get("test") else "val"
        ir_key = "test_ir" if data.get("test_ir") else "val_ir"
    elif split == "val":
        rgb_key = "val"
        ir_key = "val_ir"
    elif split == "train":
        rgb_key = "train"
        ir_key = "train_ir"
    else:
        raise ValueError(f"不支持的 split: {split}")

    rgb_path = data.get(rgb_key)
    ir_path = data.get(ir_key)

    if not path_exists(rgb_path):
        raise FileNotFoundError(f"RGB 路径不存在: key={rgb_key}, path={rgb_path}")

    if not path_exists(ir_path):
        raise FileNotFoundError(f"IR 路径不存在: key={ir_key}, path={ir_path}")

    print("=" * 80)
    print("数据集检查通过")
    print("当前工作目录 :", os.getcwd())
    print("数据集 yaml   :", str(Path(data_yaml).resolve()))
    print("数据集根目录 :", str(root))
    print("请求 split   :", split)
    print("RGB 使用键   :", rgb_key)
    print("IR  使用键   :", ir_key)
    print("RGB 路径     :", rgb_path)
    print("IR  路径     :", ir_path)
    print("=" * 80)

    return data


def write_temp_yaml(data):
    tmp_dir = Path(tempfile.mkdtemp(prefix="twostream_eval_"))
    tmp_yaml = tmp_dir / "dataset_fixed.yaml"
    save_yaml(data, tmp_yaml)
    return tmp_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/TSCIv2RawCueFusion/with_r_dark_ir_max/weights/best.pt", 
                        help="模型权重路径")
    parser.add_argument("--data", type=str, default="/storage/jyx4/projects/TwoStream_Yolov8-main/data/dronevehicle.yaml", 
                        help="数据集 yaml 路径")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--project", type=str, default="runs/test", help="结果保存的父目录")
    parser.add_argument("--name", type=str, default="exp", help="本次实验保存的子目录名")
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    weights = Path(args.weights).expanduser().resolve()
    data_yaml = Path(args.data).expanduser().resolve()

    if not weights.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights}")

    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集 yaml 不存在: {data_yaml}")

    print("weights:", weights)
    print("data   :", data_yaml)

    # 先修正 yaml 中的路径
    fixed_data = normalize_dataset_yaml(data_yaml, args.split)

    # 写临时 yaml，避免源码里 test_ir 相对路径解析出问题
    temp_yaml = write_temp_yaml(fixed_data)
    print("临时 yaml:", temp_yaml)

    model = YOLO(str(weights))

    metrics = model.val(
        data=str(temp_yaml),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
        verbose=args.verbose,
    )

    print("\n验证完成")
    print(metrics)


if __name__ == "__main__":
    main()