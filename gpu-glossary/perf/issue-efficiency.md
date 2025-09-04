---
title: What is issue efficiency?
---

Issue efficiency measures how effectively the
[warp scheduler](/gpu-glossary/device-hardware/warp-scheduler) keeps execution
pipes busy by issuing instructions from
[eligible warps](/gpu-glossary/perf/warp-execution-state).

![Of the four clock cycles in this diagram, instructions were issued on three, for an issue efficiency of 75%. Diagram inspired by the [*CUDA Techniques to Maximize Compute and Instruction Throughput*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/) talk at GTC 2025.](themed-image://cycles.svg)

An issue efficiency of 100% means every
[scheduler](/gpu-glossary/device-hardware/warp-scheduler) issued an instruction
on every cycle, indicating at least one
[eligible warp](/gpu-glossary/perf/warp-execution-state) on each cycle. Values
below 100 % signal that, during some cycles, all
[active warps](/gpu-glossary/perf/warp-execution-state) were
[stalled](/gpu-glossary/perf/warp-execution-state) - waiting on data, resources,
or dependencies - so the
[scheduler](/gpu-glossary/device-hardware/warp-scheduler) sat idle and overall
instruction throughput fell.
