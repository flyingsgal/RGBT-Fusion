"""Runtime checks for the reconstructed LASCIModule.

Run this file from the project root after replacing
ultralytics/nn/modules/block.py with block_lasci_reconstructed.py.
"""

import torch

from ultralytics.nn.modules.block import _lasci_extract_shifted_candidates

try:
    # Preferred name when the reconstructed implementation coexists with v1/v2.
    from ultralytics.nn.modules.block import LASCIModulev3 as LASCIModule
except ImportError:
    # Standalone reconstructed block keeps the original class name.
    from ultralytics.nn.modules.block import LASCIModule


def test_candidate_order():
    large = torch.arange(25.0).view(1, 1, 25, 1)
    candidates = _lasci_extract_shifted_candidates(large, 3, 5)
    assert candidates.shape == (1, 1, 9, 9, 1)

    grid = torch.arange(25.0).view(5, 5)
    expected = [
        grid[0:3, 0:3],
        grid[0:3, 1:4],
        grid[0:3, 2:5],
        grid[1:4, 0:3],
        grid[1:4, 1:4],
        grid[1:4, 2:5],
        grid[2:5, 0:3],
        grid[2:5, 1:4],
        grid[2:5, 2:5],
    ]
    for index, expected_patch in enumerate(expected):
        actual = candidates[0, 0, index, :, 0].view(3, 3)
        assert torch.equal(actual, expected_patch), (
            index,
            actual,
            expected_patch,
        )


def test_controlled_route():
    module = LASCIModule(
        c=4,
        embed_dim=4,
        small_win=3,
        large_win=5,
        num_heads=1,
        sim_threshold=0.05,
        min_ratio=0.0,
        max_ratio=0.35,
    )
    module.eval()

    with torch.no_grad():
        identity = torch.eye(4)
        module.ir_from_rgb.q_proj.weight.copy_(identity)
        module.ir_from_rgb.k_proj.weight.copy_(identity)

        # Nine spatial windows. IR prototypes point along channel 0.
        ir_small = torch.zeros(1, 9, 9, 4)
        ir_small[..., 0] = 1.0

        # Default RGB candidates point along channel 1.
        candidates = torch.zeros(1, 9, 9, 9, 4)
        candidates[..., 1] = 1.0

        # All center candidates match IR except the middle spatial window.
        candidates[:, :, 4, :, 0] = 1.0
        candidates[:, :, 4, :, 1] = 0.0
        candidates[:, 4, 4, :, 0] = 0.0
        candidates[:, 4, 4, :, 1] = 1.0

        # Right-shift candidate of the middle spatial window matches IR.
        candidates[:, 4, 5, :, 0] = 1.0
        candidates[:, 4, 5, :, 1] = 0.0

        route, score, *_ = module._compute_shared_route(
            ir_small,
            candidates,
            gh=3,
            gw=3,
        )

    assert route.sum().item() == 1, route
    assert route[0, 4].item(), route
    assert score[0, 4].item() > 0.9, score


def test_forward_backward():
    module = LASCIModule(
        c=16,
        embed_dim=16,
        small_win=3,
        large_win=5,
        num_heads=4,
        sim_threshold=0.05,
        min_ratio=0.0,
        max_ratio=0.35,
        lambda_low=0.5,
        debug=True,
    )

    module.eval()
    rgb = torch.randn(2, 16, 10, 13)
    ir = torch.randn(2, 16, 10, 13)
    rgb_raw = torch.rand(2, 3, 40, 52)
    with torch.no_grad():
        output = module([rgb, ir, rgb_raw])

    assert output.shape == rgb.shape
    selected_ratio = module.last_debug["selected_ratio"]
    assert (selected_ratio <= 0.35 + 1e-6).all(), selected_ratio

    module.train()
    module.debug = False
    rgb = torch.randn(1, 16, 7, 10, requires_grad=True)
    ir = torch.randn(1, 16, 7, 10, requires_grad=True)
    rgb_raw = torch.rand(1, 3, 28, 40)
    output = module([rgb, ir, rgb_raw])
    loss = output.square().mean()
    loss.backward()

    assert output.shape == rgb.shape
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert ir.grad is not None and torch.isfinite(ir.grad).all()


def test_cuda_amp_forward_backward():
    if not torch.cuda.is_available():
        print("[SKIP] CUDA AMP test (CUDA is unavailable)")
        return

    device = torch.device("cuda")
    module = LASCIModule(
        c=16,
        embed_dim=16,
        small_win=3,
        large_win=5,
        num_heads=4,
        sim_threshold=0.05,
        min_ratio=0.0,
        max_ratio=0.35,
        lambda_low=0.5,
    ).to(device)
    module.train()

    rgb = torch.randn(
        2,
        16,
        10,
        13,
        device=device,
        requires_grad=True,
    )
    ir = torch.randn(
        2,
        16,
        10,
        13,
        device=device,
        requires_grad=True,
    )
    rgb_raw = torch.rand(2, 3, 40, 52, device=device)

    with torch.cuda.amp.autocast():
        output = module([rgb, ir, rgb_raw])
        loss = output.square().mean()
    loss.backward()

    assert output.shape == rgb.shape
    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert ir.grad is not None and torch.isfinite(ir.grad).all()


if __name__ == "__main__":
    test_candidate_order()
    print("[PASS] candidate order and center index")

    test_controlled_route()
    print("[PASS] IR-reference cosine-margin route")

    test_forward_backward()
    print("[PASS] forward shape, ratio cap, and backward gradients")

    test_cuda_amp_forward_backward()
    if torch.cuda.is_available():
        print("[PASS] CUDA AMP forward and backward gradients")

    print("All reconstructed LASCIModule tests passed.")