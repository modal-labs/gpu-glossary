---
title: What is a Special Function Unit?
abbreviation: SFU
---

The Special Function Units (SFUs) in
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor)
accelerate certain arithmetic operations.

![The internal architecture of an H100 SM. Special Function Units are shown in maroon, along with the [Load/Store Units](/gpu-glossary/device-hardware/load-store-unit). Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](themed-image://gh100-sm.svg)

Notable for neural network workloads are transcendental mathematical operations,
like `exp`, `sin`, and `cos`.

The
[Streaming Assembler (SASS)](/gpu-glossary/device-software/streaming-assembler)
instructions associated with the SFUs generally begin with `MUFU`: `MUFU.SQRT`,
`MUFU.EX2`. See [this Godbolt link](https://godbolt.org/z/WGh3rPe83) for sample
assembly using the `MUFU.EX2` instruction to implement the `expf` intrinsic in
[CUDA C++](/gpu-glossary/host-software/cuda-c).
