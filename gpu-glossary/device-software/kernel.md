---
title: What is a Kernel?
---

![A single kernel launch corresponds to a [thread block grid](/gpu-glossary/device-software/thread-block-grid) in the [CUDA programming model](/gpu-glossary/device-software/cuda-programming-model). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](themed-image://cuda-programming-model.svg)

A kernel is the unit of CUDA code that programmers typically write and compose,
akin to a procedure or function in typical languages targeting CPUs.

Unlike procedures, a kernel is called ("launched") once and returns once, but is
executed many times, once each by a number of
[threads](/gpu-glossary/device-software/thread). These executions are generally
concurrent (their execution order is non-deterministic) and parallel (they occur
simultaneously on different execution units).

The collection of all threads executing a kernel is organized as a kernel grid —
aka a [thread block grid](/gpu-glossary/device-software/thread-block-grid), the
highest level of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model)'s
thread hierarchy. A kernel grid executes across multiple
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor)
and so operates at the scale of the entire GPU. The matching level of the
[memory hierarchy](/gpu-glossary/device-software/memory-hierarchy) is the
[global memory](/gpu-glossary/device-software/global-memory).

In [CUDA C++](/gpu-glossary/host-software/cuda-c), kernels are passed pointers
to [global memory](/gpu-glossary/device-software/global-memory) on the device
when they are invoked by the host and return nothing — they just mutate memory.
