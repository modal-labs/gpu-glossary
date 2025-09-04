---
title: What is Little's Law?
---

Little's Law establishes the amount of concurrency required to fully [hide latency](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) with throughput.

```
concurrency (ops) = latency (s) * throughput (ops/s)
```

Little's Law is described as "the most important of the fundamental laws" of analysis in [the classic quantitative systems textbook by Lazowska and others](https://homes.cs.washington.edu/~lazowska/qsp/Images/Chap_03.pdf).

Little's Law determines how many instructions must be "in flight" for GPUs to [hide latency](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) through [warp](/gpu-glossary/device-software/warp) switching by [warp schedulers](/gpu-glossary/device-hardware/warp-scheduler) (aka fine-grained thread-level parallelism, like [simultaneous multi-threading](https://en.wikipedia.org/wiki/Simultaneous_multithreading) in CPUs).

If a GPU has a peak throughput of 1 instruction per cycle and a [memory access latency](/gpu-glossary/performance/warp-latency) of 400 cycles, then 400 concurrent memory operations are needed across all [active warps](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) in a program. If the throughput goes up to 10 instructions per cycle, then the program needs 4000 concurrent memory operations to properly take advantage of the increase. For more detail, see the article on [latency hiding](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21).

For a non-trivial application of Little's Law, consider the following observation, from Section 4.3 of [Vasily Volkov's PhD thesis](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf) on [latency hiding](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21): the number of warps required to hide pure memory access latency is not much higher than that required to hide pure arithmetic latency (30 vs 24, in his experiment). Intuitively, the longer latency of memory accesses would seem to require more concurrency. But the concurrency is determined not just by latency but also by throughput. And because [memory bandwidth](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) is so much lower than [arithmetic bandwidth](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21), the required concurrency turns out to be roughly the same — a useful form of balance for a [latency hiding](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21)-oriented system that will mix arithmetic and memory operations.
