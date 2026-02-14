# FusionGuard

FusionGuard is a lightweight Python library for runtime-adaptive fusion and INT8 selection in Transformer MLP blocks.

## Statement of Need

Modern inference runtimes rely on static heuristics to decide whether to apply operator fusion and quantization. These heuristics do not adapt dynamically to specific hardware conditions, batch sizes, or model dimensions. FusionGuard provides a simple runtime benchmarking-based selection mechanism to determine the optimal execution strategy.

## Features

- FP32 unfused MLP
- FP32 fused MLP
- Dynamic INT8 quantized unfused MLP
- Dynamic INT8 quantized fused MLP
- Automatic runtime benchmarking
- Hardware-aware selection (CPU or CUDA)

## Installation

Clone the repository:

git clone <your_repo_url>
cd fusionguard

Create virtual environment:

python3 -m venv venv
source venv/bin/activate

Install:

pip install -e .

Install PyTorch if needed:

pip install torch

## Usage

Run example:

python example.py

Or use programmatically:

from fusionguard import FusionGuard
import torch

guard = FusionGuard(dim=512, hidden_dim=2048)
x = torch.randn(32, 512)
y = guard(x)

## Output

Upon initialization, FusionGuard benchmarks all variants and selects the fastest configuration for the current hardware.

## Reproducibility

FusionGuard uses deterministic benchmarking loops and supports CPU and CUDA execution depending on availability.

## License

MIT
