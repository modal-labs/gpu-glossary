---
title: What is a GPU Core?
---

The cores are the primary compute units that make up the
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor).

![The internal architecture of an H100 GPU's Streaming Multiprocessors. CUDA and Tensor Cores are shown in green. Modified from NVIDIA's [H100 white paper](https://modal-cdn.com/gpu-glossary/gtc22-whitepaper-hopper.pdf).](themed-image://gh100-sm.svg)

Examples of GPU core types include
[CUDA Cores](/gpu-glossary/device-hardware/cuda-core) and
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core).

Though GPU cores are comparable to CPU cores in that they are the component that
effects actual computations, this analogy can be quite misleading. Instead, it
is perhaps more helpful to take the viewpoint of the
[quantitative computer architect](https://archive.org/details/computerarchitectureaquantitativeapproach6thedition)
and think of them as "pipes" into which data goes in and out of which
transformed data is returned. These pipes are associated in turn with specific
[instructions](/gpu-glossary/device-software/streaming-assembler) from the
hardware's perspective and with different fundamental affordances of throughput
from the programmers' (e.g. floating point matrix multiplication arithmetic
throughput in the case of the
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core)).

The [SMs](/gpu-glossary/device-hardware/streaming-multiprocessor) are closer to
being the equivalent of CPU cores, in that they have
[register memory](/gpu-glossary/device-hardware/register-file) to store
information, cores to transform it, and an
[instruction scheduler](/gpu-glossary/device-hardware/warp-scheduler) to specify
and command transformations.
