---
title: What is a CUDA Thread Block?
---

![Thread blocks are an intermediate level of the thread group hierarchy of the [CUDA programming model](/gpu-glossary/device-software/cuda-programming-model) (left). A thread block executes on a single [Streaming Multiprocessor](/gpu-glossary/device-hardware/streaming-multiprocessor) (right, middle). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](themed-image://cuda-programming-model.svg)

A thread block is a level of the
[CUDA programming model's](/gpu-glossary/device-software/cuda-programming-model)
[thread hierarchy](/gpu-glossary/device-software/thread-hierarchy) below a
[grid](/gpu-glossary/device-software/thread-block-grid) but above a
[thread](/gpu-glossary/device-software/thread). It is the
[CUDA programming model's](/gpu-glossary/device-software/cuda-programming-model)
abstract equivalent of the concrete
[cooperative thread arrays](/gpu-glossary/device-software/cooperative-thread-array)
in
[PTX](/gpu-glossary/device-software/parallel-thread-execution)/[SASS](/gpu-glossary/device-software/streaming-assembler).

Blocks are the smallest unit of thread coordination exposed to programmers in
the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model).
Blocks must execute independently, so that any execution order for blocks is
valid, from fully serial in any order to all interleavings.

A single CUDA [kernel](/gpu-glossary/device-software/kernel) launch produces one
or more thread blocks (in the form of a
[thread block grid](/gpu-glossary/device-software/thread-block-grid)), each of
which contains one or more [warps](/gpu-glossary/device-software/warp). Blocks
can be arbitrarily sized, up to a limit of 1024 on current devices,
but they are typically multiples of the
[warp](/gpu-glossary/device-software/warp) size (32 on current devices).
