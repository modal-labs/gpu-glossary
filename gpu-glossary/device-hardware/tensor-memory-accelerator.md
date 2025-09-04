---
title: What is a Tensor Memory Accelerator?
abbreviation: TMA
---

Tensor Memory Accelerators are specialized hardware in Hopper and Blackwell
[architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)
GPUs designed to accelerate access to multi-dimensional arrays in
[GPU RAM](/gpu-glossary/device-hardware/gpu-ram).

![The internal architecture of an H100 [Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor). Note the Tensor Memory Accelerator at the bottom of the [SM](/gpu-glossary/device-hardware/streaming-multiprocessor), shared between the four sub-units. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](themed-image://gh100-sm.svg)

The TMA loads data from
[global memory](/gpu-glossary/device-software/global-memory)/[GPU RAM](/gpu-glossary/device-hardware/gpu-ram)
to
[shared memory](/gpu-glossary/device-software/shared-memory)/[L1 data cache](/gpu-glossary/device-hardware/l1-data-cache),
bypassing the
[registers](/gpu-glossary/device-software/registers)/[register file](/gpu-glossary/device-hardware/register-file)
entirely.

The first advantage of the TMA comes from reducing the use of other compute and
memory resources. The TMA hardware calculates addresses for bulk affine memory
accesses, i.e. accesses of the form `addr = width * base + offset` for many
bases and offsets concurrently, which are the most common accesses for arrays.
Offloading this work to the TMA saves space in the
[register file](/gpu-glossary/device-hardware/register-file), reducing
"[register pressure](/gpu-glossary/perf/register-pressure)", and reduces demand
on the [arithmetic bandwidth](/gpu-glossary/perf/arithmetic-bandwidth) provided
by the [CUDA Cores](/gpu-glossary/device-hardware/cuda-core). The savings are
more pronounced for large (KB-scale) accesses to arrays with two or more
dimensions.

The second advantage comes from the asynchronous execution model of TMA copies.
A single [CUDA thread](/gpu-glossary/device-software/thread) can trigger a large
copy and then rejoin its [warp](/gpu-glossary/device-software/warp) to perform
other work. Those [threads](/gpu-glossary/device-software/thread) and others in
the same [thread block](/gpu-glossary/device-software/thread-block) can then
asynchronously detect the completion of the TMA copy after it finishes and
operate on the results (as in a producer-consumer model).

For details, see the TMA sections of
[Luo et al.'s Hopper micro-benchmarking paper](https://arxiv.org/abs/2501.12084v1).
and the
[NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator).

Note that, despite the name, the Tensor Memory Accelerator does not accelerate
operations using [Tensor Memory](/gpu-glossary/device-hardware/tensor-memory).
