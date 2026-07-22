"""Server-side tests for IRGuidedSelectiveOffset.

Run from this directory:
    python test_ir_guided_selective_offset.py
"""

import tempfile
from pathlib import Path

import torch

from ir_guided_selective_offset import IRGuidedSelectiveOffset


def build_identity_module(route_mode="reliable", debug=True):
    channels = 16
    module = IRGuidedSelectiveOffset(
        c=channels,
        embed_dim=channels,
        search_radius=2,
        patch_radius=1,
        margin_threshold=0.04,
        entropy_threshold=0.68,
        softmax_tau=0.05,
        route_mode=route_mode,
        freeze_projection=True,
        require_full_search=True,
        debug=debug,
    )
    with torch.no_grad():
        module.rgb_proj.weight.copy_(torch.eye(channels))
        module.ir_proj.weight.copy_(torch.eye(channels))
        module.rgb_proj.bias.zero_()
        module.ir_proj.bias.zero_()
    return module


def test_candidate_order_and_center():
    module = build_identity_module()
    assert len(module.candidate_list) == 25
    assert module.candidate_list[0] == (-2, -2)
    assert module.candidate_list[-1] == (2, 2)
    assert module.center_index == 12
    assert module.candidate_list[module.center_index] == (0, 0)
    print("[PASS] 5*5 candidate order and center index")


def test_center_mode_exact_add():
    torch.manual_seed(1)
    module = build_identity_module(route_mode="center", debug=False)
    rgb = torch.randn(2, 16, 17, 21, requires_grad=True)
    ir = torch.randn(2, 16, 17, 21, requires_grad=True)
    output = module([rgb, ir])
    assert torch.equal(output, rgb + ir)
    output.square().mean().backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert ir.grad is not None and torch.isfinite(ir.grad).all()
    print("[PASS] center mode is the exact Add baseline")


def test_shift_sign_route_and_sampling():
    torch.manual_seed(7)
    module = build_identity_module(route_mode="reliable", debug=True)
    true_dx, true_dy = 2, -1
    ir = torch.randn(1, 16, 24, 28)
    rgb = torch.roll(ir, shifts=(true_dy, true_dx), dims=(2, 3))
    output = module([rgb, ir])
    debug = module.last_debug
    assert debug is not None
    y, x = 12, 14
    assert int(debug["route"][0, y, x]) == module.ROUTE_RELIABLE
    predicted = torch.tensor(
        [debug["pred_dx"][0, y, x], debug["pred_dy"][0, y, x]]
    )
    expected = torch.tensor([float(true_dx), float(true_dy)])
    assert torch.linalg.vector_norm(predicted - expected) < 0.35, (predicted, expected)
    # A correct sample gives rgb[p+delta] == ir[p], hence output == 2*ir.
    assert torch.allclose(output[0, :, y, x], 2.0 * ir[0, :, y, x], atol=0.08)
    print("[PASS] RGB sampling sign, reliable route, and soft offset")


def test_near_center_zero_protection():
    torch.manual_seed(11)
    module = build_identity_module(route_mode="reliable", debug=True)
    ir = torch.randn(1, 16, 24, 28)
    rgb = ir.clone()
    output = module([rgb, ir])
    debug = module.last_debug
    assert debug is not None
    interior = debug["full_search_valid"]
    assert not bool(debug["active"][interior].any())
    assert torch.allclose(output, rgb + ir, atol=1e-6, rtol=1e-6)
    print("[PASS] near-center zero-offset protection")


def test_uncertain_is_rejected():
    module = build_identity_module(route_mode="reliable", debug=False)
    # Directly test the decision rule: noncenter wins, but the distribution is
    # deliberately flat enough to exceed the entropy threshold.
    scores = torch.zeros(1, 25, 1, 1)
    scores[:, module.center_index] = 0.0
    scores[:, 0] = 0.06
    valid = torch.ones_like(scores, dtype=torch.bool)
    routed = module._route_from_scores(scores, valid)
    assert int(routed["route"].item()) == module.ROUTE_UNCERTAIN
    print("[PASS] high-entropy noncenter match is rejected as uncertain")


def test_boundary_and_backward():
    torch.manual_seed(19)
    module = build_identity_module(route_mode="reliable", debug=True)
    true_dx, true_dy = 1, 1
    ir = torch.randn(2, 16, 20, 24, requires_grad=True)
    rgb_base = torch.roll(ir.detach(), shifts=(true_dy, true_dx), dims=(2, 3))
    rgb = rgb_base.clone().requires_grad_(True)
    output = module([rgb, ir])
    debug = module.last_debug
    assert debug is not None
    assert not bool(debug["active"][:, 0].any())
    assert not bool(debug["active"][:, -1].any())
    assert not bool(debug["active"][:, :, 0].any())
    assert not bool(debug["active"][:, :, -1].any())
    output.mean().backward()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert ir.grad is not None and torch.isfinite(ir.grad).all()
    print("[PASS] boundary protection and backward gradients")


def test_projection_checkpoint_loading():
    source = build_identity_module(debug=False)
    checkpoint = {
        "matcher": {
            "rgb_proj.weight": source.rgb_proj.weight.detach().clone(),
            "rgb_proj.bias": source.rgb_proj.bias.detach().clone(),
            "ir_proj.weight": source.ir_proj.weight.detach().clone(),
            "ir_proj.bias": source.ir_proj.bias.detach().clone(),
        },
        "input_dim": 16,
        "embed_dim": 16,
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "projection_probe.pt"
        torch.save(checkpoint, path)
        loaded = IRGuidedSelectiveOffset(
            c=16,
            embed_dim=16,
            projection_path=str(path),
            freeze_projection=True,
        )
    assert torch.equal(loaded.rgb_proj.weight, source.rgb_proj.weight)
    assert torch.equal(loaded.ir_proj.weight, source.ir_proj.weight)
    assert not any(parameter.requires_grad for parameter in loaded.rgb_proj.parameters())
    assert not any(parameter.requires_grad for parameter in loaded.ir_proj.parameters())
    print("[PASS] projection_probe.pt loading and freezing")


def test_cuda_amp_if_available():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA AMP test: CUDA is unavailable")
        return
    torch.manual_seed(23)
    module = build_identity_module(route_mode="reliable", debug=False).cuda()
    rgb = torch.randn(2, 16, 24, 28, device="cuda", requires_grad=True)
    ir = torch.randn(2, 16, 24, 28, device="cuda", requires_grad=True)
    with torch.cuda.amp.autocast(True):
        output = module([rgb, ir])
        loss = output.square().mean()
    loss.backward()
    assert output.is_cuda
    assert torch.isfinite(output).all()
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert ir.grad is not None and torch.isfinite(ir.grad).all()
    print("[PASS] CUDA AMP dtype and backward")


def main():
    test_candidate_order_and_center()
    test_center_mode_exact_add()
    test_shift_sign_route_and_sampling()
    test_near_center_zero_protection()
    test_uncertain_is_rejected()
    test_boundary_and_backward()
    test_projection_checkpoint_loading()
    test_cuda_amp_if_available()
    print("All IRGuidedSelectiveOffset tests passed.")


if __name__ == "__main__":
    main()

