"""Chapter 10 experimental workload: 256-element programs, chunked internally.

This is a separate application variant, not a benchmark or profiling harness.
Compare externally with the unmodified ch10_workload.py.
"""

import argparse

import torch
import triton
import triton.language as tl


@triton.jit
def smooth5_chunked_kernel(src, dst, N: tl.constexpr, BLOCK: tl.constexpr, CHUNK: tl.constexpr):
    # Launch distribution remains BLOCK=256 elements per Triton program.
    for start in tl.static_range(0, BLOCK, CHUNK):
        i = tl.program_id(0) * BLOCK + start + tl.arange(0, CHUNK)
        interior = (i >= 2) & (i < N - 2)

        a = tl.load(src + i - 2, mask=interior, other=0.0)
        b = tl.load(src + i - 1, mask=interior, other=0.0)
        c = tl.load(src + i, mask=interior, other=0.0)
        d = tl.load(src + i + 1, mask=interior, other=0.0)
        e = tl.load(src + i + 2, mask=interior, other=0.0)

        value = (a + 2.0 * b + 3.0 * c + 2.0 * d + e) / 9.0
        tl.store(dst + i, value, mask=i < N)


def main():
    parser = argparse.ArgumentParser(description="Experimental internally chunked smoothing workload")
    parser.add_argument("--size", type=int, default=1_048_583)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--chunk", type=int, choices=[16, 32, 64, 128], default=32)
    parser.add_argument("--output", type=str, help="optional output .pt file")
    args = parser.parse_args()
    if args.size < 5 or args.steps < 1:
        parser.error("--size must be at least 5 and --steps at least 1")

    torch.manual_seed(12)
    src = torch.rand(args.size, dtype=torch.float32, device="cpu")
    dst = torch.empty_like(src)
    grid = (triton.cdiv(args.size, 256),)

    for _ in range(args.steps):
        smooth5_chunked_kernel[grid](src, dst, args.size, 256, args.chunk)
        src, dst = dst, src

    print(f"samples={args.size} steps={args.steps} checksum={src.double().sum().item():.9f} chunk={args.chunk}")
    if args.output:
        torch.save(src, args.output)


if __name__ == "__main__":
    main()
