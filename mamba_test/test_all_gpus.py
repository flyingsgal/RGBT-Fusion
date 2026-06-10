import torch
import traceback


def test_one_gpu(gpu_id):
    print("=" * 80)
    print(f"Testing GPU {gpu_id}")
    print("=" * 80)

    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    print("GPU name:", torch.cuda.get_device_name(gpu_id))
    print("Capability:", torch.cuda.get_device_capability(gpu_id))

    try:
        from mamba_ssm import Mamba
    except Exception:
        from mamba_ssm.modules.mamba_simple import Mamba

    for dtype in [torch.float32, torch.float16]:
        print(f"\nTesting Mamba module on GPU {gpu_id}, dtype={dtype}")

        torch.manual_seed(0)

        model = Mamba(
            d_model=64,
            d_state=16,
            d_conv=4,
            expand=2,
        ).to(device).to(dtype)

        x = torch.randn(
            2, 128, 64,
            device=device,
            dtype=dtype,
            requires_grad=True
        )

        y = model(x)

        print("input shape :", tuple(x.shape))
        print("output shape:", tuple(y.shape))
        print("output dtype:", y.dtype)

        loss = y.float().pow(2).mean()
        loss.backward()

        if not torch.isfinite(y).all():
            raise RuntimeError("Output has NaN or Inf")

        if x.grad is None:
            raise RuntimeError("x.grad is None")

        if not torch.isfinite(x.grad).all():
            raise RuntimeError("x.grad has NaN or Inf")

        print(f"[PASS] GPU {gpu_id}, dtype={dtype}")

    torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    n = torch.cuda.device_count()
    print("CUDA device count:", n)

    for gpu_id in range(n):
        try:
            test_one_gpu(gpu_id)
        except Exception:
            print(f"[FAIL] GPU {gpu_id}")
            traceback.print_exc()


if __name__ == "__main__":
    main()