---
title: What is the NVIDIA Runtime Compiler?
abbreviation: nvrtc
---

The NVIDIA Runtime Compiler (`nvrtc`) is a runtime compilation library for CUDA
C. It compiles [CUDA C++](/gpu-glossary/host-software/cuda-c) to
[PTX](/gpu-glossary/device-software/parallel-thread-execution) without requiring
a separate launch of the
[NVIDIA CUDA Compiler Driver](/gpu-glossary/host-software/nvcc) (`nvcc`) in
another process. It is used by some libraries or frameworks to, for example, map
generated C/C++ code to
[PTX](/gpu-glossary/device-software/parallel-thread-execution) code that can run
on a GPU.

Note that this [PTX](/gpu-glossary/device-software/parallel-thread-execution) is
then further JIT-compiled from the
[PTX](/gpu-glossary/device-software/parallel-thread-execution) IR to the
[SASS assembly](/gpu-glossary/device-software/streaming-assembler). This is done
by the [NVIDIA GPU drivers](/gpu-glossary/host-software/nvidia-gpu-drivers) and
is distinct from the compilation done by NVRTC. CUDA binaries that contain
[PTX](/gpu-glossary/device-software/parallel-thread-execution), as required for
forward compatibility, also pass through this compilation step.

NVRTC is closed source. You can find its documentation
[here](https://docs.nvidia.com/cuda/nvrtc/index.html).
