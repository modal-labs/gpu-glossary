---
title: What is a Tensor Core?
---

Tensor Cores are GPU [cores](/gpu-glossary/device-hardware/core) that operate on
entire matrices with each instruction.

![The internal architecture of an H100 SM. Note the larger size and lower number of Tensor Cores. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](themed-image://gh100-sm.svg)

For example, the `mma`
[PTX](/gpu-glossary/device-software/parallel-thread-execution) instructions
(documented
[here](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensors))
calculate D = AB + C for matrices A, B, C, and D. Operating on more data for a
single instruction fetch dramatically reduces power requirements (see
[this talk](https://youtu.be/kLiwvnr4L80?t=868) by Bill Dally, Chief Scientist
at NVIDIA).

Tensor Cores are much larger and less numerous than CUDA Cores. An H100 SXM5 has
only four Tensor Cores per
[Streaming Multiprocessor](/gpu-glossary/device-hardware/streaming-multiprocessor),
to compared to hundreds of
[CUDA Cores](/gpu-glossary/device-hardware/cuda-core).

Tensor Cores were introduced in the V100 GPU, which represented a major
improvement in the suitability of NVIDIA GPUs for large neural network worloads.
For more, see
[the NVIDIA white paper introducing the V100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf).
