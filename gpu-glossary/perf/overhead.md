---
title: What is overhead?
---

Overhead latency is the time spent with no useful work being done.

Unlike time spent [bottlenecked](/gpu-glossary/perf/performance-bottleneck) on
[compute](/gpu-glossary/perf/compute-bound) or
[memory](/gpu-glossary/perf/memory-bound), during which the GPU is working as
fast as possible, latency from overhead represents time where the GPU is instead
waiting to receive work.

Overhead often comes from CPU-side bottlenecks that prevent the GPU from
receiving work fast enough. For example, CUDA API call overhead adds on the
order of 10 μs per kernel launch. Moreover, frameworks like PyTorch or
TensorFlow spend time deciding which
[kernel](/gpu-glossary/device-software/kernel) to launch, which can take many
microseconds. We generally use the term "host overhead" here, though it's not
entirely standardized.
[CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/), which collect a
number of device-side [kernels](/gpu-glossary/device-software/kernel) together
into a single host-side launch, are a common solution to these overheads. For
more, see the
[_CUDA Techniques to Maximize Concurrency and System Utilization_ talk at GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72686/).

"Memory overhead" or "communications overhead" is overhead latency incurred
moving data back and forth from the CPU to the GPU or from one GPU to another.
But when communication bandwidth is the limiting factor, it's often better to
think of it as a form of [memory-boundedness](/gpu-glossary/perf/memory-bound)
where the "memory" is distributed across machines.
