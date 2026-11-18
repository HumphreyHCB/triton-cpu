"""Chapter 10: five-point stencil for Triton-CPU.

Run from the Triton-CPU repository root:
    TRITON_ALWAYS_COMPILE=1 python learning/ch10_stencil.py
    TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 python learning/ch10_stencil.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def five_point_stencil_kernel(
    input_ptr,
    output_ptr,
    HEIGHT: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row = idx // WIDTH
    col = idx % WIDTH

    # Every neighbour is valid only for non-boundary elements.
    interior = (row > 0) & (row < HEIGHT - 1) & (col > 0) & (col < WIDTH - 1)

    centre = tl.load(input_ptr + idx, mask=interior, other=0.0)
    north = tl.load(input_ptr + idx - WIDTH, mask=interior, other=0.0)
    south = tl.load(input_ptr + idx + WIDTH, mask=interior, other=0.0)
    west = tl.load(input_ptr + idx - 1, mask=interior, other=0.0)
    east = tl.load(input_ptr + idx + 1, mask=interior, other=0.0)

    result = (centre + north + south + west + east) / 5.0
    tl.store(output_ptr + idx, result, mask=idx < HEIGHT * WIDTH)


def reference_stencil(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    out[1:-1, 1:-1] = (
        x[1:-1, 1:-1]
        + x[:-2, 1:-1]
        + x[2:, 1:-1]
        + x[1:-1, :-2]
        + x[1:-1, 2:]
    ) / 5.0
    return out


def run_case(height: int, width: int, block_size: int = 256) -> None:
    x = torch.randn((height, width), dtype=torch.float32, device="cpu")
    out = torch.empty_like(x)

    grid = (triton.cdiv(height * width, block_size),)
    five_point_stencil_kernel[grid](x, out, height, width, block_size)

    expected = reference_stencil(x)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
    max_difference = (out - expected).abs().max().item()
    print(f"{height} x {width}: PASS, max absolute difference = {max_difference:.3g}")


if __name__ == "__main__":
    torch.manual_seed(42)
    run_case(512, 512)
    run_case(513, 517)
