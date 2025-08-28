---
title: What is Global Memory?
---

![Global memory is the highest level of the [memory hierarchy](/gpu-glossary/device-software/memory-hierarchy) in the [CUDA programming model](/gpu-glossary/device-software/cuda-programming-model). It is stored in the [GPU RAM](/gpu-glossary/device-hardware/gpu-ram). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](themed-image://cuda-programming-model.svg)

As part of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model),
each level of the
[thread hierarchy](/gpu-glossary/device-software/thread-hierarchy) has access to
matching memory from the
[memory hierarchy](/gpu-glossary/device-software/memory-hierarchy). This memory
can be used for coordination and communication and is managed by the programmer
(not the hardware or a runtime).

The highest level of that memory hierarchy is the global memory. Global memory
is global in its scope and its lifetime. That is, it is accessible by every
[thread](/gpu-glossary/device-software/thread) in a
[thread block grid](/gpu-glossary/device-software/thread-block-grid) and its
lifetime is as long as the execution of the program.

Access to data structures in the global memory can be synchronized across all
accessors using atomic instructions, as with CPU memory. Within a
[cooperative thread array](/gpu-glossary/device-software/cooperative-thread-array),
access can be more tightly synchronized, e.g. with barriers.

This level of the
[memory hierarchy](/gpu-glossary/device-software/memory-hierarchy) is typically
implemented in the [GPU's RAM](/gpu-glossary/device-hardware/gpu-ram) and
allocated from the host using a memory allocator provided by the
[CUDA Driver API](/gpu-glossary/host-software/cuda-driver-api) or the
[CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api).

The terminology "global" unfortunately collides with the `__global__` keyword in
[CUDA C/C++](/gpu-glossary/host-software/cuda-c), which annotates functions that
are launched on the host but run on the device
([kernels](/gpu-glossary/device-software/kernel)), whereas global memory is only
on the device. Early CUDA architect Nicholas Wilt wrily notes that this choice
was made "for maximum developer confusion" in his
[_CUDA Handbook_](https://www.cudahandbook.com/).
