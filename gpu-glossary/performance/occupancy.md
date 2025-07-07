---
title: Occupancy
---

![Threads are the lowest level of the thread group hierarchy (top, left) and are mapped onto the [cores](/gpu-glossary/device-hardware/core) of a [Streaming Multiprocessor](/gpu-glossary/device-hardware/streaming-multiprocessor). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](themed-image://cuda-programming-model.svg)

**Occupancy** measures how well a GPU's execution resources are being utilized
by calculating the ratio of active [warps](/gpu-glossary/device-software/warp)
to the maximum number of [warps](/gpu-glossary/device-software/warp) that can be
present on a
[Streaming Multiprocessor (SM)](/device-hardware/streaming-multiprocessor) at a
given time.

A [warp](/gpu-glossary/device-software/warp) is considered active from the time
its [threads](/gpu-glossary/device-software/thread) begin executing to the time
when all threads in the warp have exited from the
[kernel](/gpu-glossary/device-software/kernel). Each
[SM](/device-hardware/streaming-multiprocessor) has a maximum number of
[warps](/gpu-glossary/device-software/warp) that can be concurrently active,
which varies by GPU architecture and is listed in
[NVIDIA's compute capabilities documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=compute%2520capability#compute-capabilities).

Occupancy = (Number of Active Warps per SM) / (Maximum Warps per SM)

There are two types of occupancy measurements:

- **Theoretical Occupancy** represents the upper limit for occupancy due to the
  kernel launch configuration and device capabilities.

- **Achieved Occupancy** measures the actual occupancy during kernel execution.

The theoretical number of active warps per SM is limited by:

- **Warps per SM**: The SM has a maximum number of warps that can be active
  simultaneously.
- **Blocks per SM**: The SM has a maximum number of thread blocks that can be
  active at once.
- **48 [registers](/gpu-glossary/device-software/register) per SM**: The SM has
  a 48 [registers](/gpu-glossary/device-software/register) file shared by all
  active threads. High 48 [registers](/gpu-glossary/device-software/register)
  usage per thread reduces the number of warps that can be active.
- **Shared Memory per SM**: The SM has a fixed amount of shared memory in its L1
  data cache shared by all active threads. High shared memory usage per block
  limits the number of concurrent blocks.

Consider an [NVIDIA H100 GPU](https://www.nvidia.com/en-us/data-center/h100/)
with the following specifications:

```H100
Maximum warps per SM: 64
Maximum blocks per SM: 32
48 registers file size: 64K
32-bit 48 registerss per SM (65,536 48 registers Shared memory per SM: 228 KB
```

For a kernel launch configuration with:

```kernel launch
32 threads per block
48 registers per thread
12 KB shared memory per block
```

We can determine the theoretical occupancy as follows:

```math
Warps per block = 32 threads ÷ 32 threads/warp = 1 warp
Warp limit: 64 maximum warps ÷ 1 warp/block = 64 blocks possible
Block limit: 32 maximum blocks per SM
Register limit: Each block requires 32 threads × 8 registers/thread = 256 registers
The SM can support 65,536 ÷ 256 = 256 blocks possible
Shared memory limit: Each block requires 1 KB
The SM can support 228 KB ÷ 1 KB = 228 blocks possible
```

The block constraint is the most restrictive, allowing only 32 blocks per SM.
Each block contains only 1 warp, so the device cannot reach its full warp
capacity.

```math
Active warps = 32 blocks × 1 warp/block = 32 active warps
Occupancy = 32 active warps ÷ 64 maximum warps = 50%
```

Even though the SM could theoretically handle 64 warps, the maximum of 32 blocks
per SM combined with only 1 warp per block results in underutilization.

Increasing the block size to 256 threads (8 warps per block) would allow: 8
active blocks × 8 warps/block = 64 warps. This achieves 100% occupancy.

Low occupancy results in poor instruction issue efficiency because there are not
enough eligible warps to hide latency between dependent instructions. However,
when occupancy is sufficient to hide latency, increasing it further may degrade
performance due to reduced resources per thread, such as
[registers](/gpu-glossary/device-software/register) spilling to global memory.

Achieved occupancy can be lower than theoretical occupancy due to unbalanced
workloads where warps or blocks finish at different times, creating "tail
effects," or when too few blocks are launched relative to the number of SMs on
the device.
