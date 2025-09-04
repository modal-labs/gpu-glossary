---
title: What is pipe utilization?
---

Pipe utilization measures how effectively a [kernel](https://modal.com/gpu-glossary/device-software/kernel) uses the execution resources within each [Streaming Multiprocessor (SM)](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor).

Each [SM](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor) contains multiple independent execution pipes optimized for different instruction types - [CUDA Cores](https://modal.com/gpu-glossary/device-hardware/cuda-core) for general floating-point arithmetic, [Tensor Cores](https://modal.com/gpu-glossary/device-hardware/tensor-core#gpu-glossary) for tensor contractions, [load/store units](https://modal.com/gpu-glossary/device-hardware/load-store-unit) for memory access, and control flow units for branching. Pipe utilization shows what percentage of each pipeline's [peak rate](/gpu-glossary/perf/peak-rate) is being achieved when that pipe is actively executing at least one [warp](https://modal.com/gpu-glossary/device-software/warp), averaged across all active [SMs](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor).

Before debugging application performance at the level of pipe utilization, GPU programmers should first consider [GPU kernel utilization](https://modal.com/blog/gpu-utilization-guide) and [SM utilization](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21).

Pipe utilization is available in the the `sm__inst_executed_pipe_*.avg.pct_of_peak_sustained_active` metrics from [NSight Compute](https://developer.nvidia.com/nsight-compute) (`ncu`), where the asterisk represents specific pipelines like [`fma`](https://modal.com/gpu-glossary/device-hardware/cuda-core), [`tensor`](https://modal.com/gpu-glossary/device-hardware/tensor-core), [`lsu`](https://modal.com/gpu-glossary/device-hardware/load-store-unit), or `adu` (address).
