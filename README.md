# EdgeProfiler

EdgeProfiler is a lightweight, analytical cost‐model framework for estimating per‐token inference performance (latency and energy) of transformer‐style language models on edge devices. It calculates closed‐form estimates for compute, memory, storage, host‐to‐device, and network latencies—without ever downloading real weight files or launching any kernels. Using only model and hardware configuration parameters, EdgeProfiler generates a “roofline‐style” breakdown of bottlenecks so you can rapidly compare different models, precisions, and hardware targets.

---

## Table of Contents

1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [citation](#citation)  

---

## Features

- **Closed‐Form Cost Model**  
  - Computes parameter count, FLOPs per token, and memory footprint from first principles.  
  - Estimates compute‐bound and memory‐bound latencies, plus I/O, host‐to‐device, and network overheads.

- **Precision‐Aware**  
  - Supports FP32, FP16, and INT8.  
  - Adjusts both weight footprint and arithmetic intensity accordingly.

- **Hardware Configurable**  
  - Works with any device once you specify peak FLOPs, memory bandwidth, storage bandwidth, PCIe/NVLink bandwidth, and network bandwidth.  
  - Models utilization factors (compute, memory, storage, H2D, network).

- **Rapid Sweeps**  
  - Profile dozens of (model, precision, hardware) combinations in microseconds each—no real weight downloads or GPU kernels required.

- **Roofline Insights**  
  - Reports arithmetic intensity (FLOPs per byte) to identify compute‐ vs memory‐bound regimes.  
  - Provides energy estimates per token (Joules) based on simple energy‐per‐FLOP and energy‐per‐byte coefficients.

---

## Prerequisites

- Python 3.7+  

If you’re running Python 3.6 or earlier, install the required package via pip:

```bash
pip install dataclasses
```
## Installation

No special packaging is required—EdgeProfiler is a self‐contained Python module. Simply clone this repository (or copy the source file) into your project:

```bash
git clone https://github.com/ShakyaJayakody/EdgeProfiler.git
cd EdgeProfiler
python EdgeProfiler.py
```
## Citation
```bash
@article{pinnock2025edgeprofiler,
  title={EdgeProfiler: A Fast Profiling Framework for Lightweight LLMs on Edge Using Analytical Model},
  author={Pinnock, Alyssa and Jayakody, Shakya and Roxy, Kawsher A and Ahmed, Md Rubel},
  journal={arXiv preprint arXiv:2506.09061},
  year={2025}
}
```

