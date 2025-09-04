---
title: What is issue efficiency?
---

Issue efficiency measures how effectively the [warp scheduler](https://modal.com/gpu-glossary/device-hardware/warp-scheduler) keeps execution pipes busy by issuing instructions from [eligible warps](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21).

![Of the four clock cycles in this diagram, instructions were issued on three, for an issue efficiency of 75%. Diagram inspired by the [CUDA Techniques to Maximize Compute and Instruction Throughput](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/) talk at GTC 2025.
](GPU%20Performance%20Glossary%202251e7f1694980bd93e4f67a75c6e489/terminal-cycles(2)%203.png)

An issue efficiency of 100% means every [scheduler](https://modal.com/gpu-glossary/device-hardware/warp-scheduler) issued an instruction on every cycle, indicating at least one [eligible warp](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) on each cycle. Values below 100 % signal that, during some cycles, all [active warps](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) were [stalled](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) - waiting on data, resources, or dependencies - so the [scheduler](https://modal.com/gpu-glossary/device-hardware/warp-scheduler) sat idle and overall instruction throughput fell.
