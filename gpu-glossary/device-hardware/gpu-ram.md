---
title: What is GPU RAM?
---

![In state-of-the-art GPUs like the H100, RAM is located on a die directly next to the processor's. Adapted from the Wikipedia page for [high-bandwidth memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory).](themed-image://hbm-schematic.svg)

The global memory of the GPU is a large (many megabytes to gigabytes) memory
store that is addressable by all of the GPU's
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor).

It is also known as GPU RAM (random access memory) or video RAM (VRAM). It uses
Dynamic RAM (DRAM) cells, which are slower but smaller than the Static RAM
(SRAM) used in registers and shared memory. For details on DRAM and SRAM, we
recommend Ulrich Drepper's 2007 article
["What Every Programmer Should Know About Memory"](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf).

It is generally not on the same die as the
[SMs](/gpu-glossary/device-hardware/streaming-multiprocessor), though in the
latest data center-grade GPUs like the H100, it is located on a shared
[interposer](https://en.wikipedia.org/wiki/Interposer) for decreased latency and
increased bandwidth (aka
"[high-bandwidth memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)").

RAM is used to implement the
[global memory](/gpu-glossary/device-software/global-memory) of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model)
and to store [register](/gpu-glossary/device-software/registers) data that
spills from the [register file](/gpu-glossary/device-hardware/register-file).

An H100 can store 80 GiB (687,194,767,360 bits) in its RAM.
