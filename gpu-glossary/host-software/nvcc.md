---
title: What is the NVIDIA CUDA Compiler Driver?
abbreviation: nvcc
---

The NVIDIA CUDA Compiler Driver is a toolchain for compiling
[CUDA C/C++](/gpu-glossary/host-software/cuda-c) programs. It outputs binary
executables that conform to the host ABI and include
[PTX](/gpu-glossary/device-software/parallel-thread-execution) and/or
[SASS](/gpu-glossary/device-software/streaming-assembler) to be executed on the
GPU — a so-called "fat binary". These binaries are inspectable with the same
tools used for other binaries, like `readelf` on Linux, but can be additionally
manipulated with the specialized
[CUDA Binary Utilities](/gpu-glossary/host-software/cuda-binary-utilities).

The included [PTX](/gpu-glossary/device-software/parallel-thread-execution) code
is versioned by
[Compute Capability](/gpu-glossary/device-software/compute-capability),
configured by passing `compute_XYz` values to the `--gpu-architecture` or
`--gpu-code` options.

The included [SASS](/gpu-glossary/device-software/streaming-assembler) code is
versioned by
[SM architecture version](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture),
configured by passing `sm_XYz` values to the `--gpu-architecture` or
`--gpu-code` options. Passing `compute_XYz` to `--gpu-code` will also trigger
the generation of [SASS](/gpu-glossary/device-software/streaming-assembler) code
with the same version as the
[PTX](/gpu-glossary/device-software/parallel-thread-execution).

Compilation of host/CPU code is done using the host system's compiler driver,
e.g. the `gcc` compiler driver. Note that compiler drivers are not to be
confused with hardware drivers, like the
[NVIDIA GPU Drivers](/gpu-glossary/host-software/nvidia-gpu-drivers).

The documentation for `nvcc` can be found
[here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/).
