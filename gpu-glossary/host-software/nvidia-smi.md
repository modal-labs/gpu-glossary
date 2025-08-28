---
title: What is nvidia-smi?
---

This command line utility is used to query and manage the state of the GPU
exposed by the [NVML](/gpu-glossary/host-software/nvml) management libraries.
Its outputs, a sample of which appears below, are familiar to users of NVIDIA
GPUs to the point of being a
[meme](https://x.com/boborado/status/1752724223934578760).

`nvidia-smi` reports the following:

- GPU identity information like the card's model name, a UUID, and the PCI ID
- live utilization metrics for kernel execution time and memory allocation
- live power and thermal information

For details on these metrics, including how to interpret power and thermal
readings, see
[this page on the Modal docs](https://modal.com/docs/guide/gpu-metrics).

`nvidia-smi` can also list processes currently using the GPU (`-q`, `--query`,
`pmon`). Common management tasks include setting persistence mode (`-pm`),
compute mode (`-c`), power limits (`-pl`), application/locked clocks (`-ac`,
`-lgc`, `-lmc`), and performing GPU resets (`-r`).

Output can be formatted as human-readable text or XML (`-x`). While
`nvidia-smi`'s text output format is not guaranteed to be stable, the underlying
[NVML C library](/gpu-glossary/host-software/nvml) offers a stable API for tool
development.

The documentation for `nvidia-smi` can be found
[here](https://docs.nvidia.com/deploy/nvidia-smi/), and the official Python
bindings can be found [here](http://pypi.python.org/pypi/nvidia-ml-py/).

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA B200                    On  |   00000000:51:00.0 Off |                    0 |
| N/A   27C    P0            136W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA B200                    On  |   00000000:52:00.0 Off |                    0 |
| N/A   25C    P0            140W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA B200                    On  |   00000000:62:00.0 Off |                    0 |
| N/A   27C    P0            138W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA B200                    On  |   00000000:63:00.0 Off |                    0 |
| N/A   26C    P0            138W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA B200                    On  |   00000000:75:00.0 Off |                    0 |
| N/A   27C    P0            139W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA B200                    On  |   00000000:76:00.0 Off |                    0 |
| N/A   25C    P0            140W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA B200                    On  |   00000000:86:00.0 Off |                    0 |
| N/A   27C    P0            142W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA B200                    On  |   00000000:87:00.0 Off |                    0 |
| N/A   26C    P0            138W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
```
