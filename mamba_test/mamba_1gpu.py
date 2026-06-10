import torch
import traceback


def print_env():
    print("=" * 80)
    print("Environment")
    print("=" * 80)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version used by PyTorch:", torch.version.cuda)
        print("GPU name:", torch.cuda.get_device_name(0))
        print("GPU capability:", torch.cuda.get_device_capability(0))
        print("Current device:", torch.cuda.current_device())
    print()


def check_finite(name, x):
    if x is None:
        print(f"[WARN] {name}: grad is None")
        return False
    ok = torch.isfinite(x).all().item()
    print(f"{name}: finite = {ok}, shape = {tuple(x.shape)}, dtype = {x.dtype}")
    return ok


def test_selective_scan(dtype=torch.float32):
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

    print("=" * 80)
    print(f"Test selective_scan_fn, dtype = {dtype}")
    print("=" * 80)

    device = "cuda"

    batch = 2
    dim = 64
    seqlen = 128
    d_state = 16

    torch.manual_seed(0)

    # u, delta: [B, D, L]
    u = torch.randn(batch, dim, seqlen, device=device, dtype=dtype, requires_grad=True)
    delta = torch.randn(batch, dim, seqlen, device=device, dtype=dtype, requires_grad=True)

    # A: [D, N]
    # Mamba 中 A 通常是负数，保持稳定
    A = -torch.exp(torch.randn(dim, d_state, device=device, dtype=torch.float32))
    A.requires_grad_(True)

    # B, C: [D, N]
    B = torch.randn(dim, d_state, device=device, dtype=dtype, requires_grad=True)
    C = torch.randn(dim, d_state, device=device, dtype=dtype, requires_grad=True)

    # D skip: [D]
    D_skip = torch.randn(dim, device=device, dtype=torch.float32, requires_grad=True)

    # 前向：CUDA kernel
    out_cuda = selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D=D_skip,
        z=None,
        delta_bias=None,
        delta_softplus=True,
        return_last_state=False,
    )

    # 前向：参考实现
    with torch.no_grad():
        out_ref = selective_scan_ref(
            u.detach(),
            delta.detach(),
            A.detach(),
            B.detach(),
            C.detach(),
            D=D_skip.detach(),
            z=None,
            delta_bias=None,
            delta_softplus=True,
            return_last_state=False,
        )

    print("out_cuda shape:", tuple(out_cuda.shape))
    print("out_ref  shape:", tuple(out_ref.shape))
    print("out_cuda dtype:", out_cuda.dtype)

    max_abs_err = (out_cuda.float() - out_ref.float()).abs().max().item()
    mean_abs_err = (out_cuda.float() - out_ref.float()).abs().mean().item()

    print("max_abs_err :", max_abs_err)
    print("mean_abs_err:", mean_abs_err)

    # 不同精度设置不同容忍度
    if dtype == torch.float32:
        tol = 1e-4
    else:
        tol = 5e-2

    if max_abs_err < tol:
        print(f"[PASS] selective_scan_fn forward matches ref, tol = {tol}")
    else:
        print(f"[WARN] Difference is larger than tol = {tol}")
        print("       fp16 下误差略大有时正常；如果 fp32 也很大，需要重点排查。")

    # 反向传播测试
    loss = out_cuda.float().pow(2).mean()
    loss.backward()

    print()
    print("Backward grad check:")
    ok = True
    ok &= check_finite("u.grad", u.grad)
    ok &= check_finite("delta.grad", delta.grad)
    ok &= check_finite("A.grad", A.grad)
    ok &= check_finite("B.grad", B.grad)
    ok &= check_finite("C.grad", C.grad)
    ok &= check_finite("D_skip.grad", D_skip.grad)

    if ok:
        print(f"[PASS] selective_scan_fn backward works, dtype = {dtype}")
    else:
        print(f"[FAIL] selective_scan_fn backward has invalid grad, dtype = {dtype}")

    print()


def test_mamba_module(dtype=torch.float32):
    print("=" * 80)
    print(f"Test Mamba module, dtype = {dtype}")
    print("=" * 80)

    device = "cuda"

    try:
        try:
            from mamba_ssm import Mamba
        except Exception:
            from mamba_ssm.modules.mamba_simple import Mamba
    except Exception as e:
        print("[FAIL] Cannot import Mamba module.")
        print("Error:")
        traceback.print_exc()
        print()
        return

    torch.manual_seed(0)

    batch = 2
    seqlen = 128
    d_model = 64

    model = Mamba(
        d_model=d_model,
        d_state=16,
        d_conv=4,
        expand=2,
    ).to(device)

    model = model.to(dtype=dtype)

    x = torch.randn(batch, seqlen, d_model, device=device, dtype=dtype, requires_grad=True)

    y = model(x)

    print("input  shape:", tuple(x.shape))
    print("output shape:", tuple(y.shape))
    print("output dtype:", y.dtype)

    if y.shape != x.shape:
        print("[FAIL] Mamba output shape is not equal to input shape.")
        print()
        return

    loss = y.float().pow(2).mean()
    loss.backward()

    ok = True
    ok &= check_finite("x.grad", x.grad)

    param_grad_ok = True
    for name, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                print(f"[WARN] parameter {name} grad is None")
                param_grad_ok = False
            elif not torch.isfinite(p.grad).all().item():
                print(f"[FAIL] parameter {name} grad has NaN/Inf")
                param_grad_ok = False

    ok &= param_grad_ok

    if ok:
        print(f"[PASS] Mamba module forward/backward works, dtype = {dtype}")
    else:
        print(f"[FAIL] Mamba module forward/backward failed, dtype = {dtype}")

    print()


def main():
    print_env()

    if not torch.cuda.is_available():
        print("[FAIL] CUDA is not available.")
        return

    # P100 是 sm_60，不支持 bf16
    print("Note: P100 does not support bfloat16, so bf16 test is skipped.")
    print()

    # 测试 fp32
    try:
        test_selective_scan(torch.float32)
    except Exception:
        print("[FAIL] selective_scan_fn fp32 failed.")
        traceback.print_exc()
        print()

    try:
        test_mamba_module(torch.float32)
    except Exception:
        print("[FAIL] Mamba module fp32 failed.")
        traceback.print_exc()
        print()

    # 测试 fp16
    try:
        test_selective_scan(torch.float16)
    except Exception:
        print("[FAIL] selective_scan_fn fp16 failed.")
        traceback.print_exc()
        print()

    try:
        test_mamba_module(torch.float16)
    except Exception:
        print("[FAIL] Mamba module fp16 failed.")
        traceback.print_exc()
        print()

    print("=" * 80)
    print("Test finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()