---
title: What is a Warp Scheduler?
---

The Warp Scheduler of the
[Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor)
decides which group of [threads](/gpu-glossary/device-software/thread) to
execute on each clock cycle.

![The internal architecture of an H100 SM. The Warp Scheduler and Dispatch Unit are shown in orange. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](themed-image://gh100-sm.svg)

These groups of [threads](/gpu-glossary/device-software/thread), known as
[warps](/gpu-glossary/device-software/warp), are switched out on a per clock
cycle basis — roughly one nanosecond - much like the fine-grained thread-level
parallelism of simultaneous multi-threading ("hyper-threading") in CPUs, but at
a much larger scale. The ability of the Warp Schedulers to switch rapidly
between a large number of concurrent tasks as soon as their instructions'
operands are available is key to the
[latency hiding](/gpu-glossary/perf/latency-hiding) capabilities of GPUs.

Full CPU thread context switches take a few hundred to a few thousand clock
cycles (more like a microsecond than a nanosecond) due to the need to save the
context of one thread and restore the context of another. Additionally, context
switches on CPUs lead to reduced locality, further reducing performance by
increasing cache miss rates (see
[Mogul and Borg, 1991](https://www.researchgate.net/publication/220938995_The_Effect_of_Context_Switches_on_Cache_Performance)).

Because each [thread](/gpu-glossary/device-software/thread) has its own private
[registers](/gpu-glossary/device-software/registers) allocated from the
[register file](/gpu-glossary/device-hardware/register-file) of the
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor), context switches
on the GPU do not require any data movement to save or restore contexts.

And because the [L1 caches](/gpu-glossary/device-hardware/l1-data-cache) on GPUs
can be entirely programmer-managed and are
[shared](/gpu-glossary/device-software/shared-memory) between the
[warps](/gpu-glossary/device-software/warp) scheduled together onto an
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor) (see
[cooperative thread array](/gpu-glossary/device-software/cooperative-thread-array)),
context switches on the GPU have much less impact on cache hit rates. For
details on the interaction between programmer-managed caches and
hardware-managed caches in GPUs, see
[the "Maximize Memory Throughput" section of the CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-memory-throughput).

The Warp Schedulers also manage the
[execution state of warps](/gpu-glossary/perf/warp-execution-state).
