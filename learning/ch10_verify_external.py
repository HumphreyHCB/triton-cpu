"""Optional external correctness check; does not alter or instrument the workload."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def reference(x, steps):
    for _ in range(steps):
        y = torch.zeros_like(x)
        y[2:-2] = (x[:-4] + 2.0*x[1:-3] + 3.0*x[2:-2]
                     + 2.0*x[3:-1] + x[4:]) / 9.0
        x = y
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--program', default='learning/ch10_workload.py')
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as d:
        output = Path(d) / 'actual.pt'
        subprocess.run([sys.executable, args.program, '--size', '4099',
                        '--steps', '4', '--output', str(output)], check=True)
        actual = torch.load(output, weights_only=True)
    torch.manual_seed(12)
    x = torch.rand(4099, dtype=torch.float32, device='cpu')
    expected = reference(x, 4)
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    print('PASS: result matches external PyTorch reference')


if __name__ == '__main__':
    main()

