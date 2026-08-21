from ultralytics import YOLO

model = YOLO("/storage/jyx4/projects/TwoStream_Yolov8-main/runs/dronevehicle/obb/base_add/weights/best.pt")

model.val(
    data="/storage/jyx4/projects/TwoStream_Yolov8-main/data/dronevehicle_obb.yaml",
    split="val",
    imgsz=640,
    batch=16,
    device=2,
    workers=8,
    save_txt=True,
    save_conf=True,
    augment=False,
    project="./runs/diagnostic",
    name="base_best_full_val",
    iou=0.5,     # 用于 NMS
    conf=0.001,  # 建议评估时用很低的 conf（0.001/0.01），这样不会在评估前丢弃太多预测
)