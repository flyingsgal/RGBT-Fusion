# -*- coding: utf-8 -*-
"""
Two-stream YOLO validation/test script.

主要功能：
1. 修正双流数据集 yaml 中 RGB/IR 路径为绝对路径；
2. 支持 train/val/test split 检查；
3. 调用 Ultralytics YOLO model.val() 测试；
4. 额外输出并保存整体指标和各类别 Precision/Recall/mAP50/mAP50-95；
5. 将结果保存为 CSV 和 JSON，便于论文表格整理。
"""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import yaml
from ultralytics import YOLO

PathLikeValue = Union[str, List[str], None]


# -------------------------
# YAML I/O
# -------------------------
def load_yaml(yaml_path: Union[str, Path]) -> Dict[str, Any]:
    yaml_path = Path(yaml_path)
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"数据集 yaml 内容为空或格式错误: {yaml_path}")
    return data


def save_yaml(data: Dict[str, Any], yaml_path: Union[str, Path]) -> None:
    yaml_path = Path(yaml_path)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


# -------------------------
# Path processing
# -------------------------
def _resolve_one_path(root: Path, value: str) -> str:
    """将单个路径转成绝对路径。"""
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = root / p
    return str(p.resolve())


def resolve_path(root: Path, value: PathLikeValue) -> PathLikeValue:
    """把相对路径转成绝对路径，支持 str 或 list[str]。"""
    if value is None:
        return None

    if isinstance(value, str):
        return _resolve_one_path(root, value)

    if isinstance(value, list):
        return [_resolve_one_path(root, x) for x in value]

    return value


def path_exists(value: PathLikeValue) -> bool:
    """检查路径是否存在，支持 str 或 list[str]。"""
    if value is None:
        return False

    if isinstance(value, str):
        return Path(value).exists()

    if isinstance(value, list):
        return len(value) > 0 and all(Path(p).exists() for p in value)

    return False


def split_keys(split: str, data: Dict[str, Any]) -> Tuple[str, str]:
    """根据 split 返回 RGB/IR 使用的 yaml key。test 不存在时回退到 val。"""
    if split == "test":
        rgb_key = "test" if data.get("test") else "val"
        ir_key = "test_ir" if data.get("test_ir") else "val_ir"
    elif split == "val":
        rgb_key, ir_key = "val", "val_ir"
    elif split == "train":
        rgb_key, ir_key = "train", "train_ir"
    else:
        raise ValueError(f"不支持的 split: {split}")
    return rgb_key, ir_key


def normalize_dataset_yaml(data_yaml: Union[str, Path], split: str) -> Dict[str, Any]:
    """
    读取并修正双流数据集 yaml：
    1. 所有 train/val/test/train_ir/val_ir/test_ir 转绝对路径；
    2. 检查 split 对应的 RGB/IR 路径是否存在；
    3. test/test_ir 缺失时自动回退到 val/val_ir。
    """
    data_yaml = Path(data_yaml)
    data = load_yaml(data_yaml)

    if "path" not in data or data["path"] is None:
        raise ValueError(f"数据集 yaml 缺少 path 字段: {data_yaml}")

    root = Path(str(data["path"])).expanduser().resolve()
    data["path"] = str(root)

    for key in ["train", "val", "test", "train_ir", "val_ir", "test_ir"]:
        if key in data:
            data[key] = resolve_path(root, data[key])

    # 基础必需项：训练和验证路径通常是 YOLO 数据集 yaml 的必备字段。
    for key in ["train", "val", "train_ir", "val_ir"]:
        if key not in data or data[key] is None:
            raise ValueError(f"数据集 yaml 缺少必需字段: {key}")

    rgb_key, ir_key = split_keys(split, data)
    rgb_path = data.get(rgb_key)
    ir_path = data.get(ir_key)

    if not path_exists(rgb_path):
        raise FileNotFoundError(f"RGB 路径不存在: key={rgb_key}, path={rgb_path}")

    if not path_exists(ir_path):
        raise FileNotFoundError(f"IR 路径不存在: key={ir_key}, path={ir_path}")

    print("=" * 100)
    print("数据集检查通过")
    print("当前工作目录 :", os.getcwd())
    print("数据集 yaml   :", str(data_yaml.resolve()))
    print("数据集根目录 :", str(root))
    print("请求 split   :", split)
    print("RGB 使用键   :", rgb_key)
    print("IR  使用键   :", ir_key)
    print("RGB 路径     :", rgb_path)
    print("IR  路径     :", ir_path)
    print("类别 names   :", data.get("names", "未在 yaml 中找到 names"))
    print("=" * 100)

    return data


def write_temp_yaml(data: Dict[str, Any], keep_dir: Optional[Union[str, Path]] = None) -> Path:
    """写入临时 yaml。keep_dir 不为空时写到指定目录，便于复现实验。"""
    if keep_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="twostream_eval_"))
    else:
        tmp_dir = Path(keep_dir).expanduser().resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_yaml = tmp_dir / "dataset_fixed.yaml"
    save_yaml(data, tmp_yaml)
    return tmp_yaml


# -------------------------
# Metrics processing
# -------------------------
def to_float(value: Any) -> Optional[float]:
    """将 numpy 标量/普通数字转成 float；不能转换时返回 None。"""
    try:
        return float(value)
    except Exception:
        return None


def as_list(value: Any) -> List[Any]:
    """兼容 numpy array / list / tuple / None。"""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def get_names(metrics: Any, data: Dict[str, Any]) -> Dict[int, str]:
    """优先从 metrics.names 取类别名，失败则从 data['names'] 取。"""
    names = getattr(metrics, "names", None)
    if names is None:
        names = data.get("names", {})

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}

    return {}


def get_metric_box(metrics: Any) -> Any:
    """
    Ultralytics 检测任务一般是 metrics.box。
    某些版本/任务可能名称不同，这里做一个保守兼容。
    """
    for attr in ["box", "obb", "seg", "pose"]:
        obj = getattr(metrics, attr, None)
        if obj is not None:
            return obj
    raise AttributeError("metrics 中未找到 box/obb/seg/pose 指标对象，请检查 Ultralytics 版本。")


def collect_overall_metrics(metrics: Any) -> Dict[str, Any]:
    """收集整体指标。"""
    box = get_metric_box(metrics)
    results_dict = getattr(metrics, "results_dict", {}) or {}

    summary = {
        "precision": to_float(results_dict.get("metrics/precision(B)", getattr(box, "mp", None))),
        "recall": to_float(results_dict.get("metrics/recall(B)", getattr(box, "mr", None))),
        "mAP50": to_float(results_dict.get("metrics/mAP50(B)", getattr(box, "map50", None))),
        "mAP50-95": to_float(results_dict.get("metrics/mAP50-95(B)", getattr(box, "map", None))),
        "fitness": to_float(getattr(metrics, "fitness", None)),
        "save_dir": str(getattr(metrics, "save_dir", "")),
    }
    return summary


def collect_per_class_metrics(metrics: Any, names: Dict[int, str]) -> List[Dict[str, Any]]:
    """收集每类 P/R/mAP50/mAP50-95。"""
    box = get_metric_box(metrics)

    p = as_list(getattr(box, "p", None))
    r = as_list(getattr(box, "r", None))
    ap50 = as_list(getattr(box, "ap50", None))
    ap5095 = as_list(getattr(box, "ap", None))
    cls_idx = as_list(getattr(box, "ap_class_index", None))

    # 如果 ap_class_index 为空，但 ap50 有值，则默认按 0..N-1 对应。
    if not cls_idx and ap50:
        cls_idx = list(range(len(ap50)))

    rows: List[Dict[str, Any]] = []
    for i, cls_id in enumerate(cls_idx):
        cls_id = int(cls_id)
        rows.append(
            {
                "class_id": cls_id,
                "class_name": names.get(cls_id, str(cls_id)),
                "precision": to_float(p[i]) if i < len(p) else None,
                "recall": to_float(r[i]) if i < len(r) else None,
                "mAP50": to_float(ap50[i]) if i < len(ap50) else None,
                "mAP50-95": to_float(ap5095[i]) if i < len(ap5095) else None,
            }
        )
    return rows


def print_overall_metrics(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("整体测试结果")
    print("-" * 100)
    for key in ["precision", "recall", "mAP50", "mAP50-95", "fitness"]:
        value = summary.get(key)
        if value is None:
            print(f"{key:<12}: None")
        else:
            print(f"{key:<12}: {value:.6f}")
    print(f"{'save_dir':<12}: {summary.get('save_dir', '')}")
    print("=" * 100)


def print_per_class_metrics(rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    print("\n" + "=" * 100)
    print("各类别测试结果")
    print("-" * 100)
    print(f"{'id':>4}  {'class':<18}  {'P':>10}  {'R':>10}  {'mAP50':>10}  {'mAP50-95':>10}")
    print("-" * 100)

    if not rows:
        print("未获得每类指标。可能原因：当前 Ultralytics 版本未暴露 box.ap50 / box.ap。")
        print("=" * 100)
        return

    for row in rows:
        def fmt(x: Any) -> str:
            return "nan" if x is None else f"{x:.6f}"

        print(
            f"{row['class_id']:>4}  "
            f"{row['class_name']:<18}  "
            f"{fmt(row.get('precision')):>10}  "
            f"{fmt(row.get('recall')):>10}  "
            f"{fmt(row.get('mAP50')):>10}  "
            f"{fmt(row.get('mAP50-95')):>10}"
        )
    print("=" * 100)


def save_metrics_files(
    save_dir: Union[str, Path],
    summary: Dict[str, Any],
    per_class_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    temp_yaml: Path,
) -> Tuple[Path, Path]:
    """保存 summary_metrics.json 和 per_class_metrics.csv。"""
    save_dir = Path(save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    json_path = save_dir / "summary_metrics.json"
    csv_path = save_dir / "per_class_metrics.csv"

    payload = {
        "summary": summary,
        "args": vars(args),
        "fixed_dataset_yaml": str(temp_yaml),
        "per_class": per_class_rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    fieldnames = ["class_id", "class_name", "precision", "recall", "mAP50", "mAP50-95"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class_rows)

    return json_path, csv_path


# -------------------------
# CLI
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stream YOLO test/val script with per-class mAP50 export.")

    parser.add_argument(
        "--weights",
        type=str,
        default="/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/obb/LASCI_add/weights/best.pt",
        help="模型权重路径，例如 runs/train/exp/weights/best.pt",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="/storage/jyx4/projects/TwoStream_Yolov8-main/data/dronevehicle_obb.yaml",
        help="数据集 yaml 路径",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="测试哪个 split")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸，需与训练/论文实验设置保持一致")
    parser.add_argument("--batch", type=int, default=16, help="batch size")
    parser.add_argument("--device", type=str, default="1", help="GPU id，例如 0；CPU 用 cpu")
    parser.add_argument("--workers", type=int, default=8, help="dataloader workers")

    # 正式测 mAP 建议 conf=0.001；可视化预测才常用 conf=0.25。
    parser.add_argument("--conf", type=float, default=0.001, help="置信度阈值。正式测 mAP 建议 0.001")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值，注意不是 mAP50 的评价 IoU")
    parser.add_argument("--max-det", type=int, default=300, help="每张图最多保留检测框数量")

    parser.add_argument("--project", type=str, default="runs/test", help="结果保存的父目录")
    parser.add_argument("--name", type=str, default="LASCI_add2", help="本次实验保存的子目录名")
    parser.add_argument("--exist-ok", action="store_true", help="允许覆盖/复用已有 project/name 目录")

    parser.add_argument("--save-json", action="store_true", help="保存 COCO JSON 结果，若该任务支持")
    parser.add_argument("--save-txt", action="store_true", help="保存预测 txt")
    parser.add_argument("--plots", action="store_true", default=True, help="保存 PR/F1/confusion matrix 等图")
    parser.add_argument("--no-plots", dest="plots", action="store_false", help="不保存曲线图/混淆矩阵图")
    parser.add_argument("--verbose", action="store_true", help="Ultralytics 输出更详细日志")
    parser.add_argument("--half", action="store_true", help="使用 FP16 验证/测试，GPU 支持时可加速")
    parser.add_argument("--augment", action="store_true", help="测试时增强 TTA，一般正式消融不建议开启")

    parser.add_argument(
        "--keep-temp-yaml",
        action="store_true",
        help="将修正后的 dataset_fixed.yaml 保存到结果目录，而不是系统临时目录",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weights = Path(args.weights).expanduser().resolve()
    data_yaml = Path(args.data).expanduser().resolve()

    if not weights.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights}")

    if not data_yaml.exists():
        raise FileNotFoundError(f"数据集 yaml 不存在: {data_yaml}")

    print("weights:", weights)
    print("data   :", data_yaml)
    print("conf   :", args.conf, "  # 正式测 mAP 通常建议 0.001")
    print("iou    :", args.iou, "  # 这是 NMS IoU，不是 mAP50 的评价 IoU")

    # 先修正 yaml 中的路径。
    fixed_data = normalize_dataset_yaml(data_yaml, args.split)

    # 如果用户希望保留临时 yaml，则预先写入 project/name 下，方便复现实验。
    temp_yaml_dir = None
    if args.keep_temp_yaml:
        temp_yaml_dir = Path(args.project).expanduser().resolve() / args.name
    temp_yaml = write_temp_yaml(fixed_data, keep_dir=temp_yaml_dir)
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
        exist_ok=args.exist_ok,
        save_json=args.save_json,
        save_txt=args.save_txt,
        plots=args.plots,
        verbose=args.verbose,
        half=args.half,
        augment=args.augment,
    )

    names = get_names(metrics, fixed_data)
    summary = collect_overall_metrics(metrics)
    per_class_rows = collect_per_class_metrics(metrics, names)

    print("\n验证完成")
    print_overall_metrics(summary)
    print_per_class_metrics(per_class_rows)

    # metrics.save_dir 是 Ultralytics 实际保存目录；如果取不到，则退回 project/name。
    save_dir = summary.get("save_dir") or str(Path(args.project) / args.name)
    json_path, csv_path = save_metrics_files(save_dir, summary, per_class_rows, args, temp_yaml)

    print("\n结果文件已保存：")
    print("summary json:", json_path)
    print("per-class csv:", csv_path)
    print("\n说明：")
    print("1. per_class_metrics.csv 中的 mAP50 是各类别 AP@0.5。")
    print("2. per_class_metrics.csv 中的 mAP50-95 是各类别 AP@[0.5:0.95]。")
    print("3. --iou 控制的是 NMS 去重阈值，不是 mAP50 的评价 IoU。")


if __name__ == "__main__":
    main()
