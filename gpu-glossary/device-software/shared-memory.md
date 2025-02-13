---
title: What is Shared Memory?
---

![Shared memory is the abstract memory associated with the [thread block](/gpu-glossary/device-software/thread-block) level (left, center) of the CUDA thread group hierarchy (left). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](themed-image://cuda-programming-model.svg)

Shared memory is the level of the
[memory hierarchy](/gpu-glossary/device-software/memory-hierarchy) corresponding
to the [thread block](/gpu-glossary/device-software/thread-block) level of the
thread group hierarchy in the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model).
It is generally expected to be much smaller but much faster (in throughput and
latency) than the [global memory](/gpu-glossary/device-software/global-memory).

A fairly typical [kernel](/gpu-glossary/device-software/kernel) therefore looks
something like this:

- load data from [global memory](/gpu-glossary/device-software/global-memory)
  into shared memory
- perform a number of arithmetic operations on that data via the
  [CUDA Cores](/gpu-glossary/device-hardware/cuda-core) and
  [Tensor Cores](/gpu-glossary/device-hardware/tensor-core)
- optionally, synchronize [threads](/gpu-glossary/device-software/thread) within
  a [thread block](/gpu-glossary/device-software/thread-block) by means of
  barriers while performing those operations
- write data back into
  [global memory](/gpu-glossary/device-software/global-memory), optionally
  preventing races across
  [thread blocks](/gpu-glossary/device-software/thread-block) by means of
  atomics

Shared memory is stored in the
[L1 data cache](/gpu-glossary/device-hardware/l1-data-cache) of the GPU's
[Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor).
