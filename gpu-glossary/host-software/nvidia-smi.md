---
title: What is nvidia-smi?
---

This command line utility is used to query and manage the state of the GPU
exposed by the [NVML](/gpu-glossary/host-software/nvml) management libraries.
Its outputs, a sample of which appears below, are familiar to users of NVIDIA
GPUs to the point of being a
[meme](https://x.com/boborado/status/1752724223934578760).

`nvidia-smi` reports GPU identification (model, UUID, PCI ID), utilization 
metrics (GPU, memory, encoder/decoder), memory usage (FB, BAR1), power draw, 
temperature, and performance state (P-state). It can also list processes 
currently using the GPU (`-q`, `--query`, `pmon`).
Common management tasks include setting persistence mode (`-pm`), compute 
mode (`-c`), power limits (`-pl`), application/locked clocks 
(`-ac`, `-lgc`, `-lmc`), and performing GPU resets (`-r`). Output can be 
formatted as human-readable text or XML (`-x`).
While `nvidia-smi`'s text output format is not guaranteed backward compatible, 
the underlying NVML C library and its Python bindings offer a stable API for 
tool development.
The documentation for `nvidia-smi` can be found 
[here](https://docs.nvidia.com/deploy/nvidia-smi/),
and the official python bindings also exist 
[here](http://pypi.python.org/pypi/nvidia-ml-py/).

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.90.07              Driver Version: 550.90.07      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:53:00.0 Off |                    0 |
| N/A   25C    P0             92W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA H100 80GB HBM3          On  |   00000000:64:00.0 Off |                    0 |
| N/A   27C    P0             93W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA H100 80GB HBM3          On  |   00000000:75:00.0 Off |                    0 |
| N/A   26C    P0             96W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA H100 80GB HBM3          On  |   00000000:86:00.0 Off |                    0 |
| N/A   27C    P0             93W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA H100 80GB HBM3          On  |   00000000:97:00.0 Off |                    0 |
| N/A   27C    P0             95W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA H100 80GB HBM3          On  |   00000000:A8:00.0 Off |                    0 |
| N/A   25C    P0             91W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA H100 80GB HBM3          On  |   00000000:B9:00.0 Off |                    0 |
| N/A   26C    P0             91W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA H100 80GB HBM3          On  |   00000000:CA:00.0 Off |                    0 |
| N/A   24C    P0             91W /  700W |       1MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
```
