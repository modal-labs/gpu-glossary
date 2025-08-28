---
title: What is the CUDA Thread Hierarchy?
---

![The thread hierarchy of the [CUDA programming model](/gpu-glossary/device-software/cuda-programming-model) spans from individual [threads](/gpu-glossary/device-software/thread) to [thread blocks](/gpu-glossary/device-software/thread-block) to [thread block grids](/gpu-glossary/device-software/thread-block-grid) (left), mapping onto the hardware from [CUDA Cores](/gpu-glossary/device-hardware/cuda-core) to [Streaming Multiprocessors](/gpu-glossary/device-hardware/streaming-multiprocessor) to the entire GPU (right). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](themed-image://cuda-programming-model.svg)

The thread hierarchy is a key abstraction of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model),
alongside the
[memory hierarchy](/gpu-glossary/device-software/memory-hierarchy). It organizes
the execution of parallel programs across multiple levels, from individual
threads up to entire GPU devices.

At the lowest level are individual
[threads](/gpu-glossary/device-software/thread). Like a thread of execution on a
CPU, each [CUDA thread](/gpu-glossary/device-software/thread) executes a stream
of instructions. The hardware resources that effect arithmetic and logic
instructions are called [cores](/gpu-glossary/device-hardware/core) or sometimes
"pipes". Threads are selected for execution by the
[Warp Scheduler](/gpu-glossary/device-hardware/warp-scheduler).

The intermediate level consists of
[thread blocks](/gpu-glossary/device-software/thread-block), which are also
known as
[cooperative thread arrays](/gpu-glossary/device-software/cooperative-thread-array)
in [PTX](/gpu-glossary/device-software/parallel-thread-execution) and
[SASS](/gpu-glossary/device-software/streaming-assembler). Each
[thread](/gpu-glossary/device-software/thread) has a unique identifier within
its [thread block](/gpu-glossary/device-software/thread-block). These thread
identifiers are index-based, to support easy assignment of work to threads based
on indices into input or output arrays. All threads within a block are scheduled
simultaneously onto the same
[Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor).
They can coordinate through
[shared memory](/gpu-glossary/device-software/shared-memory) and synchronize
with barriers.

At the highest level, multiple
[thread blocks](/gpu-glossary/device-software/thread-block) are organized into a
[thread block grid](/gpu-glossary/device-software/thread-block-grid) that spans
the entire GPU. [Thread blocks](/gpu-glossary/device-software/thread-block) are
strictly limited in their coordination and communication. Blocks within a grid
execute concurrently with respect to each other, with no guaranteed execution
order. [CUDA programs](/gpu-glossary/device-software/cuda-programming-model)
must be written so that any interleaving of blocks is valid, from fully serial
to fully parallel. That means
[thread blocks](/gpu-glossary/device-software/thread-block) cannot, for
instance, synchronize with barriers. Like
[threads](/gpu-glossary/device-software/thread), each
[thread block](/gpu-glossary/device-software/thread-block) has a unique,
index-based identifier to support assignment of work based on array index.

This hierarchy maps directly onto the
[GPU hardware](/gpu-glossary/device-hardware):
[threads](/gpu-glossary/device-software/thread) execute on individual
[cores](/gpu-glossary/device-hardware/core),
[thread blocks](/gpu-glossary/device-software/thread-block) are scheduled onto
[SMs](/gpu-glossary/device-hardware/streaming-multiprocessor), and
[grids](/gpu-glossary/device-software/thread-block-grid) utilize all available
[SMs](/gpu-glossary/device-hardware/streaming-multiprocessor) on the device.
