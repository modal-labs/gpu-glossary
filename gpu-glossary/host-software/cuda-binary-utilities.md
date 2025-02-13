---
title: What are the CUDA Binary Utilities?
---

The CUDA Binary Utilities are a collection of tools for examining the contents
of binaries like those output by `nvcc`, the
[NVIDIA CUDA Compiler driver](/gpu-glossary/host-software/nvcc).

One tool, `cuobjdump`, can be used to examine and manipulate the contents of
entire host binaries or of the CUDA-specific `cubin` files that are normally
embedded within those binaries.

Another, `nvidisasm`, is intended for manipulating `cubin` files. It can extract
[SASS assembler](/gpu-glossary/device-software/streaming-assembler) and
manipulate it, e.g. constructing control flow graphs and mapping assembly
instructions to lines in CUDA program files.

You can find their documentation
[here](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html).
