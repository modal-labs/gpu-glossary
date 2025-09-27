---
title: What is a Warp?
---

A warp is a group of [threads](/gpu-glossary/device-software/thread) that are
scheduled together and execute in parallel. All
[threads](/gpu-glossary/device-software/thread) in a warp are scheduled onto a
single
[Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor).
A single [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) typically
executes multiple warps, at the very least all warps from the same
[Cooperative Thread Array](/gpu-glossary/device-software/cooperative-thread-array),
aka [thread block](/gpu-glossary/device-software/thread-block).

Warps are the typical unit of execution on a GPU. In normal execution, all
[threads](/gpu-glossary/device-software/thread) of a warp execute the same
instruction in parallel — the so-called "Single-Instruction, Multiple Thread" or
SIMT model. When the [threads](/gpu-glossary/device-software/thread) in a warp
split from one another to execute different instructions, also known as
[warp divergence](/gpu-glossary/perf/warp-divergence), performance generally
drops precipitously.

Warp size is technically a machine-dependent constant, but in practice (and
elsewhere in this glossary) it is 32.

When a warp is issued an instruction, the results are generally not available
within a single clock cycle, and so dependent instructions cannot be issued.
While this is most obviously true for fetches from
[global memory](/gpu-glossary/device-software/global-memory), which generally
[go off-chip](/gpu-glossary/device-hardware/gpu-ram), it is also true for some
arithmetic instructions (see
[the CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions)
for a table of results per clock cycle for specific instructions).

A warp whose next instruction is delayed by missing operands is said to be
[stalled](/gpu-glossary/perf/warp-execution-state).

Instead of waiting for an instruction's results to return, when multiple warps
are scheduled onto a single
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor), the
[Warp Scheduler](/gpu-glossary/device-hardware/warp-scheduler) will select
another warp to execute. This
[latency-hiding](/gpu-glossary/perf/latency-hiding) is how GPUs achieve high
throughput and ensure work is always available for all of their cores during
execution. For this reason, it is often beneficial to maximize the number of
warps scheduled onto each
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor), ensuring there is
always an [eligible](/gpu-glossary/perf/warp-execution-state) warp for the
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor) to run. The
fraction of cycles on which a warp was issued an instruction is known as the
[issue efficiency](/gpu-glossary/perf/issue-efficiency). The degree of
concurrency in warp scheduling is known as
[occupancy](/gpu-glossary/perf/occupancy).

Warps are not actually part of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model)'s
[thread hierarchy](/gpu-glossary/device-software/thread-hierarchy). Instead,
they are an implementation detail of the implementation of that model on NVIDIA
GPUs. In that way, they are somewhat akin to
[cache lines](https://www.nic.uoregon.edu/~khuck/ts/acumem-report/manual_html/ch03s02.html)
in CPUs: a feature of the hardware that you don't directly control and don't
need to consider for program correctness, but which is important for achieving
[maximum performance](/gpu-glossary/perf).

Warps are named in reference to weaving, "the first parallel thread technology",
according to
[Lindholm et al., 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf).
The equivalent of warps in other GPU programming models include
[subgroups](https://github.com/gpuweb/gpuweb/pull/4368) in WebGPU,
[waves](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html)
in DirectX, and
[simdgroups](https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups#2928931)
in Metal.
