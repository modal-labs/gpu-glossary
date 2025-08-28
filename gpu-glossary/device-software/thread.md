---
title: What is a CUDA Thread?
---

![Threads are the lowest level of the thread group hierarchy (top, left) and are mapped onto the [cores](/gpu-glossary/device-hardware/core) of a [Streaming Multiprocessor](/gpu-glossary/device-hardware/streaming-multiprocessor). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](themed-image://cuda-programming-model.svg)

A _thread of execution_ (or "thread" for short) is the lowest unit of
programming for GPUs, the base and atom of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model)'s
[thread hierarchy](/gpu-glossary/device-software/thread-hierarchy). A thread has
its own [registers](/gpu-glossary/device-software/registers), but little else.

Both [SASS](/gpu-glossary/device-software/streaming-assembler) and
[PTX](/gpu-glossary/device-software/parallel-thread-execution) programs target
threads. Compare this to a typical C program in a POSIX environment, which
targets a process, itself a collection of one or more threads. Unlike POSIX
threads, [CUDA](/gpu-glossary/device-software/cuda-programming-model) threads
are not used to make syscalls.

Like a thread on a CPU, a GPU thread can have a private instruction
pointer/program counter. However, for performance reasons, GPU programs are
generally written so that all the threads in a
[warp](/gpu-glossary/device-software/warp) share the same instruction pointer,
executing instructions in lock-step (see also
[Warp Scheduler](/gpu-glossary/device-hardware/warp-scheduler)).

Also like threads on CPUs, GPU threads have stacks in
[global memory](/gpu-glossary/device-hardware/gpu-ram) for storing spilled
registers and a function call stack, but high-performance
[kernels](/gpu-glossary/device-software/kernel) generally limit use of both.

A single [CUDA Core](/gpu-glossary/device-hardware/cuda-core) executes
instructions from a single thread.
