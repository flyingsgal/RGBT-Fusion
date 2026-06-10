import sys
import torch

PROJECT_ROOT = "/storage/jyx4/projects/TwoStream_Yolov8-main"
sys.path.insert(0, PROJECT_ROOT)

import ultralytics
print("Using ultralytics from:", ultralytics.__file__)

from ultralytics import YOLO

yaml_path = "/storage/jyx4/projects/TwoStream_Yolov8-main/yaml/mamba/TSCIv3RawCueSS2DFusion.yaml"

model = YOLO(yaml_path)
model.model.cuda().eval()

x = torch.randn(1, 6, 640, 640).cuda()

with torch.no_grad():
    y = model.model(x)

print("forward ok")