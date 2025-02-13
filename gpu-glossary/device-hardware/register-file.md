---
title: What is a Register File?
---

The register file of the
[Streaming Multiprocessor](/gpu-glossary/device-hardware/streaming-multiprocessor)
stores bits in between their manipulation by the
[cores](/gpu-glossary/device-hardware/core).

![The internal architecture of an H100 SM. The register file is depicted in blue. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](themed-image://gh100-sm.svg)

The register file is split into 32 bit registers that can be dynamically
reallocated between different data types, like 32 bit integers, 64 bit floating
point numbers, and (pairs of) 16 bit floating point numbers.

Allocation of registers in a
[Streaming Multiprocessor](/gpu-glossary/device-hardware/streaming-multiprocessor)
to [threads](/gpu-glossary/device-software/thread) is therefore generally
managed by a compiler like [nvcc](/gpu-glossary/host-software/nvcc), which
optimizes register usage by
[thread blocks](/gpu-glossary/device-software/thread-block).
