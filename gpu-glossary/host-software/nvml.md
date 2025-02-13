---
title: What is the NVIDIA Management Library?
abbreviation: NVML
---

The NVIDIA Management Library (NVML) is used for monitoring and managing the
state of NVIDIA GPUs. It exposes, for example, the power draw and temperature of
the GPU, the allocated memory, and the device's power limit and power limiting
state.

The function of NVML are frequently accessed via the
[nvidia-smi](/gpu-glossary/host-software/nvidia-smi) command line utility, but
are also accessible to programs via wrappers, like
[pynvml in Python](https://pypi.org/project/pynvml/) and
[nvml_wrapper in Rust](https://docs.rs/nvml-wrapper/latest/nvml_wrapper/).
