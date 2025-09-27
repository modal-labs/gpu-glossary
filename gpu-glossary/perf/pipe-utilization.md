---
title: What is pipe utilization?
---

Pipe utilization measures how effectively a
[kernel](/gpu-glossary/device-software/kernel) uses the execution resources
within each
[Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor).

Each [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) contains
multiple independent execution pipes optimized for different instruction types -
[CUDA Cores](/gpu-glossary/device-hardware/cuda-core) for general floating-point
arithmetic, [Tensor Cores](/gpu-glossary/device-hardware/tensor-core) for tensor
contractions, [load/store units](/gpu-glossary/device-hardware/load-store-unit)
for memory access, and control flow units for branching. Pipe utilization shows
what percentage of each pipeline's [peak rate](/gpu-glossary/perf/peak-rate) is
being achieved when that pipe is actively executing at least one
[warp](/gpu-glossary/device-software/warp), averaged across all active
[SMs](/gpu-glossary/device-hardware/streaming-multiprocessor).

Before debugging application performance at the level of pipe utilization, GPU
programmers should first consider
[GPU kernel utilization](https://modal.com/blog/gpu-utilization-guide) and
[SM utilization](/gpu-glossary/perf/streaming-multiprocessor-utilization).

Pipe utilization is available in the
`sm__inst_executed_pipe_*.avg.pct_of_peak_sustained_active` metrics from
[NSight Compute](https://developer.nvidia.com/nsight-compute) (`ncu`), where the
asterisk represents specific pipelines like
[`fma`](/gpu-glossary/device-hardware/cuda-core),
[`tensor`](/gpu-glossary/device-hardware/tensor-core),
[`lsu`](/gpu-glossary/device-hardware/load-store-unit), or `adu` (address).
