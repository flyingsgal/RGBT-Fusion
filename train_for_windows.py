#检查模型输出的代码位置以及是否能build模型
# from ultralytics import YOLO
# import torch
# model = YOLO("/storage/jyx4/projects/TwoStream_Yolov8-main/yaml/yolov8-twoCSP-p4p5-align-refine2ctx-weightedadd.yaml")
# x = torch.randn(1, 6, 640, 640)

# with torch.no_grad():
#     y = model.model(x)

# print(type(model.model.model[-1]).__name__)
# print(type(y))

#检测OBB模块的代码位置
# from ultralytics.nn.modules import OBB
# import inspect

# print(inspect.getsourcefile(OBB))
# print(inspect.getsource(OBB.forward))

#检查
from ultralytics import YOLO
import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
model = YOLO("/storage/jyx4/projects/TwoStream_Yolov8-main/yaml/sod_chance/ACMp345.yaml")
model.info()
# x = torch.randn(1, 6, 640, 640)
# with torch.no_grad():
#     y = model.model(x)
# print(type(y))
# net = model.model

# net.train()
# x = torch.randn(2, 6, 640, 640)

# with torch.no_grad():
#     out = net(x)

# print("type(out):", type(out))
# print("len(out):", len(out))
# print("type(out[0]):", type(out[0]))
# print("len(out[0]):", len(out[0]) if isinstance(out[0], (list, tuple)) else None)
# print("type(out[1]):", type(out[1]))
# print("shape(out[1]):", out[1].shape if hasattr(out[1], "shape") else None)