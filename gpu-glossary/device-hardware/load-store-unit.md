---
title: What is a Load/Store Unit?
abbreviation: LSU
---

The Load/Store Units (LSUs) dispatch requests to load or store data to the
memory subsystems of the GPU.

![The internal architecture of an H100 SM. Load/Store Units are shown in pink, along with the [Special Function Units](/gpu-glossary/device-hardware/special-function-unit). Modified from NVIDIA's [H100 white paper](https://modal-cdn.com/gpu-glossary/gtc22-whitepaper-hopper.pdf).](themed-image://gh100-sm.svg)

Most importantly for
[CUDA programmers](/gpu-glossary/host-software/cuda-software-platform), they
interact directly with the
[Streaming Multiprocessor](/gpu-glossary/device-hardware/streaming-multiprocessor)'s
on-chip SRAM [L1 data cache](/gpu-glossary/device-hardware/l1-data-cache) and
indirectly with the off-chip, on-device
[global RAM](/gpu-glossary/device-hardware/gpu-ram) that respectively implement
the lowest and highest levels of the
[memory hierarchy](/gpu-glossary/device-software/memory-hierarchy) in the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model).
