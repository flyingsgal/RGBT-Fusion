"""Load projection_probe.pt into an already-constructed YOLO model.

The projection path is deliberately kept out of the model YAML because this
Ultralytics fork applies ast.literal_eval to string arguments.
"""

from pathlib import Path
from typing import Any, List


def find_selective_offset_modules(yolo_or_network: Any) -> List[Any]:
    network = getattr(yolo_or_network, "model", yolo_or_network)
    # YOLO.model is the task network, while task_network.model is usually the
    # Sequential layer list. Both expose modules().
    if not hasattr(network, "modules"):
        raise TypeError("Expected a YOLO wrapper or torch.nn.Module network.")
    return [
        module
        for module in network.modules()
        if module.__class__.__name__ == "IRGuidedSelectiveOffset"
    ]


def load_projection_into_yolo(
    yolo_or_network: Any,
    projection_path: str,
    freeze: bool = True,
    route_mode: str = "reliable",
) -> List[Any]:
    path = Path(projection_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Projection checkpoint does not exist: {}".format(path))
    modules = find_selective_offset_modules(yolo_or_network)
    if not modules:
        raise RuntimeError("No IRGuidedSelectiveOffset module was found.")
    for module in modules:
        module.load_projection_checkpoint(str(path))
        module.set_projection_trainable(not freeze)
        module.set_route_mode(route_mode)
    return modules
