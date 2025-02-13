---
title: What is a GPU Core?
---

The cores are the primary compute units that make up the
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor).

![The internal architecture of an H100 GPU's Streaming Multiprocessors. CUDA and Tensor Cores are shown in green. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](themed-image://gh100-sm.svg)

Examples of GPU core types include
[CUDA Cores](/gpu-glossary/device-hardware/cuda-core) and
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core).

Though GPU cores are comparable to CPU cores in that they are the component that
effects actual computations, this analogy can be misleading. The
[SMs](/gpu-glossary/device-hardware/streaming-multiprocessor) are closer to
being the equivalent of CPU cores.
