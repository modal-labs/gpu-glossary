---
title: What is occupancy?
---

Occupancy is the ratio of the [active warps](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) to the maximum number of [active warps](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) on a device.

![There are four warp slots per cycle on each of four clock cycles and so there are 16=4*4 total warp slots, and there are active warps in 15 of them, for an occupancy of ~94%. Diagram inspired by the [CUDA Techniques to Maximize Compute and Instruction Throughput](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/) talk at GTC 2025.](GPU%20Performance%20Glossary%202251e7f1694980bd93e4f67a75c6e489/terminal-cycles(2)%202.png)

There are two types of occupancy measurements:

- *Theoretical Occupancy* represents the upper limit for occupancy due to the
kernel launch configuration and device capabilities.
- *Achieved Occupancy* measures the actual occupancy during [kernel](https://modal.com/gpu-glossary/device-software/kernel) execution, aka on [active cycles](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21).

As part of the [CUDA programming model](https://modal.com/gpu-glossary/device-software/cuda-programming-model), all the [threads](https://modal.com/gpu-glossary/device-software/thread) in a [thread block](https://modal.com/gpu-glossary/device-software/thread-block) are scheduled onto the same [Streaming Multiprocessor (SM)](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor). Each [SM](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor) has resources (like space in [shared memory](https://modal.com/gpu-glossary/device-software/shared-memory)) that must be partitioned across [thread blocks](https://modal.com/gpu-glossary/device-software/thread-block) and so limit the number of [thread blocks](https://modal.com/gpu-glossary/device-software/thread-block) that can be scheduled on the [SM](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor).

Let’s work through an example. Consider an NVIDIA H100 GPU, which has these specifications:

```
Maximum warps/SM: 64
Maximum blocks/SM: 32
(32 bit) Registers: 65536
Shared memory (smem): 228 KB
```

For a [kernel](https://modal.com/gpu-glossary/device-software/kernel) using 32 [threads](https://modal.com/gpu-glossary/device-software/thread) per [thread block](https://modal.com/gpu-glossary/device-software/thread-block), 8 [registers](https://modal.com/gpu-glossary/device-software/registers) per [thread](https://modal.com/gpu-glossary/device-software/thread), and 12 KB [shared memory](https://modal.com/gpu-glossary/device-software/shared-memory) per [thread block](https://modal.com/gpu-glossary/device-software/thread-block), we end up limited by [shared memory](https://modal.com/gpu-glossary/device-software/shared-memory):

```
64 > 1   = warps/block = 32 threads/block ÷ 32 threads/warp
32 < 256 = blocks/register-file = 65,536 registers/register-file ÷ (32 threads/block × 8 registers/thread)
32       = blocks/SM
19       = blocks/smem = 228 KB/smem ÷ 12 KB/block
```

Even though our [register file](https://modal.com/gpu-glossary/device-hardware/register-file) is big enough to support 256 [thread blocks](https://modal.com/gpu-glossary/device-software/thread-block) concurrently, our [shared memory](https://modal.com/gpu-glossary/device-software/shared-memory) is not, and so we can only run 19 [thread blocks](https://modal.com/gpu-glossary/device-software/thread-block) per [SM](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor), corresponding to 19 [warps](https://modal.com/gpu-glossary/device-software/warp). This is the common case where the size of program intermediates stored in [registers](https://modal.com/gpu-glossary/device-software/registers) is much smaller than elements of the program’s [working set](https://en.wikipedia.org/wiki/Working_set) that need to stay in [shared memory](https://modal.com/gpu-glossary/device-software/shared-memory).

Low occupancy can hurt performance when there aren't enough [eligible warps](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) to [hide the latency](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) of instructions, which shows up as low instruction [issue efficiency](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) and [under-utilized pipes](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21). However, once occupancy is sufficient for [latency hiding](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21), increasing it further may actually degrade performance. Higher occupancy reduces resources per [thread](https://modal.com/gpu-glossary/device-software/thread), potentially [bottlenecking the kernel on registers](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) or reducing the [arithmetic intensity](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) that modern GPU architectures are designed to exploit.

More generally, occupancy measures what fraction of its maximum parallel tasks the GPU is handling simultaneously, which is not inherently a target of optimization in most kernels. Instead, we want to maximize the [utilization](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) of compute resources if we are [compute-bound](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) or memory resources if we are [memory-bound](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21).

In particular, high-performance GEMM kernels on Hopper and Blackwell [architecture](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) GPUs often run at single-digit occupancy percentages because they don't need many [warps](https://modal.com/gpu-glossary/device-software/warp) to fully saturate the [Tensor Cores](https://modal.com/gpu-glossary/device-hardware/tensor-core#gpu-glossary).
