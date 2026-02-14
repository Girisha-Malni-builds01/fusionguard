# FusionGuard

![Python](https://img.shields.io/badge/python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-research--software-orange)

FusionGuard is a lightweight research-oriented Python library for **runtime-adaptive fusion and INT8 selection in Transformer MLP inference**.  
It dynamically benchmarks multiple execution strategies and selects the fastest configuration for the current hardware environment.

Modern inference runtimes rely heavily on static heuristics to decide whether operator fusion or quantization should be applied. However, fusion profitability and quantization performance depend on hardware architecture, memory bandwidth, batch size and model dimensionality. FusionGuard introduces a minimal runtime benchmarking engine that empirically selects the optimal execution strategy for Transformer MLP blocks. The system evaluates fused vs unfused and FP32 vs dynamic INT8 variants and automatically deploys the fastest configuration.

---

# Statement of Need

Inference optimization decisions in systems such as:

- NVIDIA TensorRT
- Google XLA
- Intel oneDNN
- PyTorch Inductor

are often made at compile time using heuristic cost models.

However:

- Fusion profitability depends on memory bandwidth regime.
- Quantization speedups depend on backend engine (FBGEMM, QNNPACK).
- Small-batch inference behaves differently than large-batch.
- CPU vs GPU characteristics differ significantly.
- ARM and x86 quantization backends differ.

There is currently no minimal runtime empirical decision layer in PyTorch that:

- Benchmarks fusion vs non-fusion
- Benchmarks quantized vs non-quantized
- Selects fastest variant dynamically
- Remains lightweight and reproducible

FusionGuard fills this gap.

---

# Design Overview

FusionGuard evaluates four execution variants:

| Variant | Precision | Fusion | Description |
|----------|------------|--------|--------------|
| FP32 Unfused | float32 | No | Baseline Linear → GELU → Linear |
| FP32 Fused | float32 | Yes | Reduced intermediate overhead |
| INT8 Unfused | int8 dynamic | No | Dynamic quantized linear layers |
| INT8 Fused | int8 dynamic | Yes | Fusion + quantization |

---

FusionGuard measures average per-iteration latency:
```
t = (T_end - T_start) / N
```
Procedure:

1. Warm-up runs (10 iterations)
2. Timed execution loop (default 50 iterations)
3. CUDA synchronization if GPU present
4. Average latency computation
5. Fastest variant selection

---

# Installation

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/fusionguard.git
cd fusionguard

python3.10 -m venv venv
source venv/bin/activate

pip install torch
pip install -e .
```

# Programmatic Use
```
from fusionguard import FusionGuard
import torch

guard = FusionGuard(dim=768, hidden_dim=3072)
x = torch.randn(16, 768)
y = guard(x)
```
After initialization, guard(x) automatically uses the optimal execution path.

## Hardware Adaptation

FusionGuard supports:
* CPU-only environments
* CUDA-enabled systems
* ARM and x86 architectures
* Automatic quantization backend selection
If a quantization backend is unavailable, FP32 variants remain functional.

## Reproducibility
* Reproducibility is ensured via:
* Deterministic iteration counts
* Explicit CUDA synchronization
* Fixed input tensor shapes
* No stochastic graph rewriting

## Limitations

*Dynamic INT8 primarily benefits CPU inference.
*CUDA INT8 dynamic quantization is limited.
*Fusion is implemented at the PyTorch module level (not kernel-level fusion).
*No persistent caching across sessions.
*Limited to Transformer MLP blocks (no attention yet).

## Performance Interpretation

Fusion is beneficial when:

* Kernel launch overhead dominates
* Intermediate activation writes are costly
* Memory traffic is a bottleneck

Quantization is beneficial when:

* Compute-bound regime dominates
* INT8 backend is optimized
* Memory bandwidth is constrained

These behaviors align with the Roofline performance model framework.


# Citation

If you use FusionGuard in research, please cite:
```
@software{fusionguard2026,
  title = {FusionGuard: Runtime Adaptive Fusion and INT8 Selection for Transformer Inference},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/YOUR_USERNAME/fusionguard}
}
```

