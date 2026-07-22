from ultralytics import YOLO
from ultralytics.nn.modules import IRGuidedSelectiveOffset

MODEL_YAML = "/storage/jyx4/projects/TwoStream_Yolov8-main/yaml/offset/IRGuidedSelectiveOffset_P3.yaml"

ADD_PT = (
    "/storage/jyx4/projects/TwoStream_Yolov8-main/"
    "runs/dronevehicle/obb/base_add/weights/best.pt"
)

PROJECTION_PT = (
    "/storage/jyx4/projects/TwoStream_Yolov8-main/"
    "runs/p3_common_space_probe/first_run/projection_probe.pt"
)

# 1. 按新YAML构造模型
model = YOLO(MODEL_YAML)

# 2. 加载Add模型的所有共享权重
model.load(ADD_PT)

# 3. 找到P3偏移模块
offset_modules = [
    module
    for module in model.model.modules()
    if isinstance(module, IRGuidedSelectiveOffset)
]

assert len(offset_modules) == 1, (
    f"应当只有一个IRGuidedSelectiveOffset，实际找到{len(offset_modules)}个"
)

offset_module = offset_modules[0]

# 4. 注入公共空间projection
offset_module.load_projection_checkpoint(PROJECTION_PT)

# 5. 上界实验中保持冻结
offset_module.set_projection_trainable(False)
offset_module.set_route_mode("reliable")

print("projection loaded:", offset_module.projection_path)
print("input_dim:", offset_module.c)
print("embed_dim:", offset_module.embed_dim)
print(
    "projection trainable:",
    any(p.requires_grad for p in offset_module.rgb_proj.parameters())
    or any(p.requires_grad for p in offset_module.ir_proj.parameters())
)

import torch

checkpoint = torch.load(PROJECTION_PT, map_location="cpu")

assert torch.allclose(
    offset_module.rgb_proj.weight.detach().cpu(),
    checkpoint["matcher"]["rgb_proj.weight"],
)

assert torch.allclose(
    offset_module.ir_proj.weight.detach().cpu(),
    checkpoint["matcher"]["ir_proj.weight"],
)

print("[PASS] RGB projection matches checkpoint")
print("[PASS] IR projection matches checkpoint")
print("[PASS] projection checkpoint imported successfully")