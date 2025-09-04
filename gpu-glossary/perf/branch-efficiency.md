---
title: What is branch efficiency?
---

Branch efficiency measures how often all
[threads](/gpu-glossary/device-software/thread) in a
[warp](/gpu-glossary/device-software/warp) take the same execution path when
encountering conditional statements.

Branch efficiency is calculated as the ratio of uniform control flow decisions
to total branch instructions executed. Control flow uniformity is measured at
the level of [warps](/gpu-glossary/device-software/warp), and so branch
efficiency indicates the absence of
[warp divergence](/gpu-glossary/perf/warp-divergence).

Not all conditionals reduce branch efficiency. The common "bounds-check"
fragment that appears in most [kernels](https://godbolt.org/z/d1PsYYPnW)

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
```

will generally have very high branch efficiency, since most
[warps](/gpu-glossary/device-software/warp) will be composed of
[threads](/gpu-glossary/device-software/thread) that all have the same value for
the conditional, save for a single [warp](/gpu-glossary/device-software/warp)
whose [threads](/gpu-glossary/device-software/thread)' indices are above and
below `n`.

While CPUs also care about the uniformity of branching behavior, they tend to
care primarily about uniformity of branch behavior over time, as part of
hardware-controlled branch prediction and speculative execution. That is, as
circuits within the CPU accumulate data about a branch as it is encountered
multiple times during program execution, the performance should improve.

GPUs instead care about uniformity in space. That is, uniformity is measured
within [warps](/gpu-glossary/device-software/warp), whose
[threads](/gpu-glossary/device-software/thread) execute concurrently in time but
are mapped onto distinct data, and performance improves if those
[threads](/gpu-glossary/device-software/thread) branch uniformly.
