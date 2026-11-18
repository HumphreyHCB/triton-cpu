
    
  
"""Chapter 10: balanced-arithmetic experimental variant.

Runs repeated 5-point smoothing over a one-dimensional signal using Triton-CPU.
This application contains no profiler, benchmark harness or compiler diagnostics.
It can be investigated externally, even if treated as a black box.
"""

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def smooth5_balanced_kernel(src, dst, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    interior = (i >= 2) & (i < N - 2)

    a = tl.load(src + i - 2, mask=interior, other=0.0)
    b = tl.load(src + i - 1, mask=interior, other=0.0)
    c = tl.load(src + i, mask=interior, other=0.0)
    d = tl.load(src + i + 1, mask=interior, other=0.0)
    e = tl.load(src + i + 2, mask=interior, other=0.0)

    value = ((a + 2.0 * b) + (3.0 * c + 2.0 * d) + e) / 9.0
    tl.store(dst + i, value, mask=i < N)


def main():
    parser = argparse.ArgumentParser(description="Balanced-arithmetic smoothing experiment")
    parser.add_argument("--size", type=int, default=1_048_583, help="number of signal samples")
    parser.add_argument("--steps", type=int, default=250, help="smoothing iterations")
    parser.add_argument("--output", type=str, help="optional output .pt file")
    args = parser.parse_args()
    if args.size < 5 or args.steps < 1:
        parser.error("--size must be at least 5 and --steps at least 1")

    torch.manual_seed(12)
    src = torch.rand(args.size, dtype=torch.float32, device="cpu")
    dst = torch.empty_like(src)
    grid = (triton.cdiv(args.size, 256),)

    for _ in range(args.steps):
        smooth5_balanced_kernel[grid](src, dst, args.size, 256)
        src, dst = dst, src

    print(f"samples={args.size} steps={args.steps} checksum={src.double().sum().item():.9f}")
    if args.output:
        torch.save(src, args.output)


if __name__ == "__main__":
    main()

