#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
QRL_TEST_GPU="${1:-0}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing QRL Python environment: $PYTHON_BIN" >&2
    exit 2
fi

if [[ ! "$QRL_TEST_GPU" =~ ^[0-9]+$ ]]; then
    echo "Usage: $0 [gpu-index]" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="$QRL_TEST_GPU"

echo "Testing physical GPU index $QRL_TEST_GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version \
        --format=csv,noheader 2>&1 || true
fi
echo

exec "$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import gc
import sys
import time
import traceback

import torch
from torch import nn


def synchronize() -> None:
    torch.cuda.synchronize()


def release(*objects: object) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()


def run_test(name: str, test) -> bool:
    print(f"\n[{name}]")
    try:
        test()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=5)
        return False
    print("PASS")
    return True


print(f"PyTorch:             {torch.__version__}")
print(f"PyTorch CUDA build:  {torch.version.cuda}")
print(f"CUDA available:      {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print(
        "\nCUDA is unavailable in this process. Check CUDA_VISIBLE_DEVICES, "
        "the NVIDIA driver, and /dev/nvidia* permissions.",
        file=sys.stderr,
    )
    raise SystemExit(2)

device = torch.device("cuda:0")
properties = torch.cuda.get_device_properties(device)
capability = torch.cuda.get_device_capability(device)

print(f"Visible GPU:         {properties.name}")
print(f"Compute capability:  {capability[0]}.{capability[1]}")
print(f"GPU memory:          {properties.total_memory / 2**30:.1f} GiB")
print(f"cuDNN:               {torch.backends.cudnn.version()}")
print(f"BF16 API support:    {torch.cuda.is_bf16_supported()}")

results: dict[str, bool] = {}


def test_tf32() -> None:
    if capability < (8, 0):
        raise RuntimeError(
            f"TF32 Tensor Cores require compute capability >= 8.0, got {capability}"
        )
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    left = torch.randn(1024, 1024, device=device, requires_grad=True)
    right = torch.randn(1024, 1024, device=device)
    output = left @ right
    output.square().mean().backward()
    synchronize()

    print(f"matmul precision:    {torch.get_float32_matmul_precision()}")
    print(f"allow_tf32:          {torch.backends.cuda.matmul.allow_tf32}")
    print(f"output dtype:        {output.dtype}")
    release(left, right, output)


def test_bf16_amp() -> None:
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("torch.cuda.is_bf16_supported() returned False")

    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.LayerNorm(1024),
        nn.SiLU(),
        nn.Linear(1024, 1024),
    ).to(device)
    inputs = torch.randn(256, 1024, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(inputs)
        loss = output.float().square().mean()
    loss.backward()
    synchronize()

    print(f"parameter dtype:     {next(model.parameters()).dtype}")
    print(f"autocast output:     {output.dtype}")
    print(f"loss dtype:          {loss.dtype}")
    release(model, inputs, output, loss)


def test_fused_adamw() -> None:
    parameter = nn.Parameter(torch.randn(1024, 1024, device=device))
    optimizer = torch.optim.AdamW(
        [parameter],
        lr=1e-4,
        weight_decay=0.0,
        fused=True,
    )
    optimizer.zero_grad(set_to_none=True)
    loss = parameter.square().mean()
    loss.backward()
    optimizer.step()
    synchronize()

    print(f"optimizer fused:     {optimizer.param_groups[0].get('fused')}")
    print(f"parameter dtype:     {parameter.dtype}")
    release(parameter, optimizer, loss)


def test_torch_compile() -> None:
    if not hasattr(torch, "compile"):
        raise RuntimeError("This PyTorch build has no torch.compile API")

    model = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.LayerNorm(1024),
        nn.SiLU(),
        nn.Linear(1024, 1024),
    ).to(device)
    compiled = torch.compile(model, mode="default")
    inputs = torch.randn(256, 1024, device=device)

    start = time.perf_counter()
    first_output = compiled(inputs)
    first_output.float().square().mean().backward()
    synchronize()
    first_seconds = time.perf_counter() - start

    model.zero_grad(set_to_none=True)
    start = time.perf_counter()
    second_output = compiled(inputs)
    second_output.float().square().mean().backward()
    synchronize()
    second_seconds = time.perf_counter() - start

    print(f"first compiled step: {first_seconds:.3f}s (includes compilation)")
    print(f"cached step:         {second_seconds:.3f}s")
    release(model, compiled, inputs, first_output, second_output)


results["TF32"] = run_test("TF32", test_tf32)
results["BF16/AMP"] = run_test("BF16/AMP", test_bf16_amp)
results["fused AdamW"] = run_test("fused AdamW", test_fused_adamw)
results["torch.compile"] = run_test("torch.compile", test_torch_compile)

print("\nSummary")
print("-------")
for name, passed in results.items():
    print(f"{name:14s} {'SUPPORTED' if passed else 'FAILED'}")

if all(results.values()):
    print("\nAll four generic GPU optimization tests passed.")
    raise SystemExit(0)

print("\nOne or more tests failed. Review the errors above.", file=sys.stderr)
raise SystemExit(1)
PY
