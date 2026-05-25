---
title: "What is the CUDA Tile programming model?"
---

The CUDA Tile programming model is a tile-based programming model targeting
NVIDIA GPUs.

The traditional
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model)
exposes a [hierarchy of threads](/gpu-glossary/device-software/thread-hierarchy)
and a [hierarchy of memories](/gpu-glossary/device-software/memory-hierarchy) to
user programs that receive pointers and execute concurrently to mutate memory
relative to those pointers. The same instructions are issued to multiple
[threads](/gpu-glossary/device-software/thread) in parallel, and so this
programming model is a "single-instruction, multiple thread" (SIMT) programming
model. This is the programming model used in, for instance,
[CUDA C/C++](/gpu-glossary/host-software/cuda-c) and the
[PTX](/gpu-glossary/device-software/parallel-thread-execution) IR used by
pre-CUDA-Tile programs targeting NVIDIA GPUs.

This programming model is defined for a
["unified" hardware substrate](/gpu-glossary/device-hardware/cuda-device-architecture)
-- the "U" in "CUDA". That is, homogenous
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor)
with homogenous [CUDA Cores](/gpu-glossary/device-hardware/cuda-core) implement
the majority of operations, rather than the device comprising specialized cores,
programmed heterogenously, as was generically the case in graphics programming
before CUDA.

This programming model is a poor fit for GPUs of the latest
[SM architectures](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture),
where the vast majority of
[arithmetic bandwidth](/gpu-glossary/perf/arithmetic-bandwidth) is in the
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core). The Tensor Cores can
only perform matrix multiplications and must be programmed with
[thread](/gpu-glossary/device-software/thread)-level instructions and
asynchrony, rather than the [warp](/gpu-glossary/device-software/warp)-level
asynchrony used to program the rest of the hardware.

In the CUDA Tile programming model, programs are expressed at the level of
_tile-kernels_, which are instances of the program that run concurrently across
a grid of _tile blocks_, each of which is a single thread of execution.
Tile-kernels operate, in the happy path, on _structured pointers_, which combine
a pointer with information about an array: its total extent (shape) and its
access patterns (stride). Note the similarity to the
[CuTe](/gpu-glossary/host-software/cute) type system for `Layout`s and
`Tensor`s.

As with traditional "CUDA SIMT" in CUDA C/C++ and PTX IR, this programming model
is shared between high-level languages and an intermediate representation --
here,
[Tile IR](https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html).

At time of writing in mid-2026, the CUDA Tile programming model is new, and to
what extent it will replace the existing "CUDA SIMT" programming model is as yet
unclear. The CUDA Tile programming model is currently available via
[cuTile Python](https://docs.nvidia.com/cuda/cutile-python/quickstart.html). It
is also available, albeit in experimental form, via
[cuTile BASIC](/gpu-glossary/host-software/cutile-basic) and
[cuTile Rust](https://github.com/nvlabs/cutile-rs).
