---
title: What is an active cycle?
---

An active cycle is a clock cycle in which a [Streaming Multiprocessor](/gpu-glossary/device-hardware/streaming-multiprocessor) has at least one [active warp](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) resident. The [warp](/gpu-glossary/device-software/warp) may be [eligible](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) or [stalled](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21).

![All cycles depicted in this diagram are active cycles. Diagram inspired by the [CUDA Techniques to Maximize Compute and Instruction Throughput](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/) talk at GTC 2025.](GPU%20Performance%20Glossary%202251e7f1694980bd93e4f67a75c6e489/terminal-cycles(2)%201.png)
