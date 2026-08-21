# 查看模型信息

from ultralytics import YOLO
model = YOLO('/storage/jyx4/projects/TwoStream_Yolov8-main/yaml/frequency/FreqCoupledWavelet_parallel.yaml')
model.info()

# from ultralytics import YOLO

# model = YOLO("/storage/jyx4/projects/TwoStream_Yolov8-main/yaml/mamba/TSCIv4SharedWindowMambaFusion.yaml").model
# for name, module in model.named_modules():
#     if "WindowSS2D" in module.__class__.__name__:
#         print("Found WindowSS2D:", name)
#         print("has_mamba =", getattr(module, "has_mamba", None))
#         print("m_lr =", getattr(module, "m_lr", None))