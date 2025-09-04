---
title: What is a Cooperative Thread Array?
---

![Cooperative thread arrays correspond to the [thread block](/gpu-glossary/device-software/thread-block) level of the thread block hierarchy in the [CUDA programming model](/gpu-glossary/device-software/cuda-programming-model). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](themed-image://cuda-programming-model.svg)

A cooperative thread array (CTA) is a collection of threads scheduled onto the
same
[Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor).
CTAs are the
[PTX](/gpu-glossary/device-software/parallel-thread-execution)/[SASS](/gpu-glossary/device-software/streaming-assembler)
implementation of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model)'s
[thread blocks](/gpu-glossary/device-software/thread-block). CTAs are composed
of one or more [warps](/gpu-glossary/device-software/warp).

Programmers can direct [threads](/gpu-glossary/device-software/thread) within a
CTA to coordinate with each other. The programmer-managed
[shared memory](/gpu-glossary/device-software/shared-memory), in the
[L1 data cache](/gpu-glossary/device-hardware/l1-data-cache) of the
[SMs](/gpu-glossary/device-hardware/streaming-multiprocessor), makes this
coordination fast. Threads in different CTAs cannot coordinate with each other
via barriers, unlike threads within a CTA, and instead must coordinate via
[global memory](/gpu-glossary/device-software/global-memory), e.g. via atomic
update instructions. Due to driver control over the scheduling of CTAs at
runtime, CTA execution order is indeterminate and blocking a CTA on another CTA
can easily lead to deadlock.

The number of CTAs that can be scheduled onto a single
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor)
sets the [achievable occupancy](/gpu-glossary/perf/occupancy)
and depends on a number
of factors. Fundamentally, the
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor) has a limited set
of resources — lines in the
[register file](/gpu-glossary/device-hardware/register-file), "slots" for
[warps](/gpu-glossary/device-software/warp), bytes of
[shared memory](/gpu-glossary/device-software/shared-memory) in the
[L1 data cache](/gpu-glossary/device-hardware/l1-data-cache) — and each CTA uses
a certain amount of those resources (as calculated at
[compile](/gpu-glossary/host-software/nvcc) time) when scheduled onto an
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor).
