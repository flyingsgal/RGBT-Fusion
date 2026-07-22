from ultralytics import YOLO

pt_path = "/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/obb/LASCI_add/weights/best.pt"
model = YOLO(pt_path).model

for name, module in model.named_modules():
    if "LASCI" not in module.__class__.__name__:
        continue

    print("\n" + "=" * 80)
    print("module:", name)
    print("class:", module.__class__.__name__)

    print("\n[全部标量属性]")
    for key, value in vars(module).items():
        if isinstance(value, (int, float, bool, str, type(None))):
            print(f"{key}: {value}")

    print("\n[重点参数]")
    keys = [
        "small_win",
        "large_win",
        "threshold",
        "sim_threshold",
        "margin_threshold",
        "min_ratio",
        "max_ratio",
        "budget_ratio",
        "lambda_low",
        "low_thr",
        "low_smooth",
        "g_min",
        "init_gamma",
    ]
    for key in keys:
        if hasattr(module, key):
            print(f"{key}: {getattr(module, key)}")