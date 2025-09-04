---
title: What is the NVIDIA CUDA Profiling Tools Interface?
abbreviation: CUPTI
---

The NVIDIA CUDA Profiling Tools Interface (CUPTI) provides a set of APIs for
profiling execution of [CUDA C++](/gpu-glossary/host-software/cuda-c),
[PTX](/gpu-glossary/device-software/parallel-thread-execution), and
[SASS](/gpu-glossary/device-software/streaming-assembler) code on GPUs.
Critically, it synchronizes timestamps across the CPU host and the GPU device.

CUPTI's interfaces are consumed by, for example, the
[NSight Systems Profiler](/gpu-glossary/host-software/nsight-systems) and the
[PyTorch Profiler](https://modal.com/docs/examples/torch_profiling).

You can find its documentation [here](https://docs.nvidia.com/cupti/).

For details on using profiling tools for GPU applications running on Modal, see
[this example from our documentation](/docs/examples/torch_profiling).
