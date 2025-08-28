---
title: What is the CUDA Memory Hierarchy?
---

![[Shared memory](/gpu-glossary/device-software/shared-memory) and [global memory](/gpu-glossary/device-software/global-memory) are two levels of the memory hierarchy in the [CUDA programming model](/gpu-glossary/device-software/cuda-programming-model) (left), mapping onto the [L1 data cache](/gpu-glossary/device-hardware/l1-data-cache) and [GPU RAM](/gpu-glossary/device-hardware/gpu-ram), respectively. Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](themed-image://cuda-programming-model.svg)

As part of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model),
each level of the
[thread hierarchy](/gpu-glossary/device-software/thread-hierarchy) has access to
a distinct block of memory shared by all
[threads](/gpu-glossary/device-software/thread) in a group at that level: a
"memory hierarchy". This memory can be used for coordination and communication
and is managed by the programmer (not the hardware or a runtime).

For a [thread block grid](/gpu-glossary/device-software/thread-block-grid), that
shared memory is in the [GPU's RAM](/gpu-glossary/device-hardware/gpu-ram) and
is known as the [global memory](/gpu-glossary/device-software/global-memory).
Access to this memory can be coordinated with atomic operations and barriers,
but execution order across
[thread blocks](/gpu-glossary/device-software/thread-block) is indeterminate.

For a single [thread](/gpu-glossary/device-software/thread), the memory is a
chunk of the
[Streaming Multiprocessor's (SM's)](/gpu-glossary/device-hardware/streaming-multiprocessor)
[register file](/gpu-glossary/device-hardware/register-file). According to the
original semantics of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model),
this memory is private to a [thread](/gpu-glossary/device-software/thread), but
certain instructions added to
[PTX](/gpu-glossary/device-software/parallel-thread-execution) and
[SASS](/gpu-glossary/device-software/streaming-assembler) to target matrix
multiplication on [Tensor Cores](/gpu-glossary/device-hardware/tensor-core)
share inputs and outputs across [threads](/gpu-glossary/device-software/thread).

In between, the [shared memory](/gpu-glossary/device-software/shared-memory) for
the [thread block](/gpu-glossary/device-software/thread-block) level of the
thread hierarchy is stored in the
[L1 data cache](/gpu-glossary/device-hardware/l1-data-cache) of each
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor). Careful management
of this cache — e.g. loading data into it to support the maximum number of
arithmetic operations before new data is loaded — is key to the art of designing
high-performance CUDA [kernels](/gpu-glossary/device-software/kernel).
