---
title: What is the L1 Data Cache?
---

The L1 data cache is the private memory of the
[Streaming Multiprocessor](/gpu-glossary/device-hardware/streaming-multiprocessor)
(SM).

![The internal architecture of an H100 SM. The L1 data cache is depicted in light blue. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](themed-image://gh100-sm.svg)

Each SM partitions that memory among
[groups of threads](/gpu-glossary/device-software/thread-block) scheduled onto
it.

The L1 data cache is co-located with and only about an order of magnitude slower
than the components that effect computations (e.g. the
[CUDA Cores](/gpu-glossary/device-hardware/cuda-core)).

It is implemented with SRAM, the same basic semiconductor cell used in CPU
caches and registers and in the
[memory subsystem of Groq LPUs](https://groq.com/wp-content/uploads/2023/05/GroqISCAPaper2022_ASoftwareDefinedTensorStreamingMultiprocessorForLargeScaleMachineLearning-1.pdf).
The L1 data cache is accessed by the
[Load/Store Units](/gpu-glossary/device-hardware/load-store-unit) of the
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor).

CPUs also maintain an L1 cache. In CPUs, that cache is fully hardware-managed.
In GPUs that cache is mostly programmer-managed, even in high-level languages
like [CUDA C](/gpu-glossary/host-software/cuda-c).

Each L1 data cache in each of an H100's SMs can store 256 KiB (2,097,152 bits).
Across the 132 SMs in an H100 SXM 5, that's 33 MiB (242,221,056 bits) of cache
space.
