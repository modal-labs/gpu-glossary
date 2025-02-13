---
title: What is a Streaming Multiprocessor?
abbreviation: SM
---

When we [program GPUs](/gpu-glossary/host-software/cuda-software-platform), we
produce
[sequences of instructions](/gpu-glossary/device-software/streaming-assembler)
for its Streaming Multiprocessors to carry out.

![A diagram of the internal architecture of an H100 GPU's Streaming Multiprocessors. GPU cores appear in green, other compute units in maroon, scheduling units in orange, and memory in blue. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](themed-image://gh100-sm.svg)

Streaming Multiprocessors (SMs) of NVIDIA GPUs are roughly analogous to the
cores of CPUs. That is, SMs both execute computations and store state available
for computation in registers, with associated caches. Compared to CPU cores, GPU
SMs are simple, weak processors. Execution in SMs is pipelined within an
instruction (as in almost all CPUs since the 1990s) but there is no speculative
execution or instruction pointer prediction (unlike all contemporary
high-performance CPUs).

However, GPU SMs can execute more
[threads](/gpu-glossary/device-software/thread) in parallel.

For comparison: an
[AMD EPYC 9965](https://www.techpowerup.com/cpu-specs/epyc-9965.c3904) CPU draws
at most 500 W and has 192 cores, each of which can execute instructions for at
most two threads at a time, for a total of 384 threads in parallel, running at
about 1.25 W per thread.

An H100 SXM GPU draws at most 700 W and has 132 SMs, each of which has four
[Warp Schedulers](/gpu-glossary/device-hardware/warp-scheduler) that can each
issue instructions to 32 threads (aka a
[warp](/gpu-glossary/device-software/warp)) in parallel per clock cycle, for a
total of 128 × 132 > 16,000 parallel threads running at about 5 cW apiece. Note
that this is truly parallel: each of the 16,000 threads can make progress with
each clock cycle.

GPU SMs also support a large number of _concurrent_ threads -- threads of
execution whose instructions are interleaved.

A single SM on an H100 can concurrently execute up to 2048 threads split across
64 thread groups of 32 threads each. With 132 SMs, that's a total of over
250,000 concurrent threads.

CPUs can also run many threads concurrently. But switches between
[warps](/gpu-glossary/device-software/warp) happen at the speed of a single
clock cycle (over 1000x faster than context switches on a CPU), again powered by
the SM's [Warp Schedulers](/gpu-glossary/device-hardware/warp-scheduler). The
volume of available [warps](/gpu-glossary/device-software/warp) and the speed of
warp switches help hide latency caused by memory reads, thread synchronization,
or other expensive instructions, ensuring that the compute resources (especially
the [CUDA Cores](/gpu-glossary/device-hardware/cuda-core) and
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core)) are well utilized.

This latency-hiding is the secret to GPUs' strengths. CPUs seek to hide latency
from end-users and programmers by maintaining large, hardware-managed caches and
sophisticated instruction prediction. This extra hardware limits the fraction of
their silicon area, power, and heat budgets that CPUs can allocate to
computation.

![GPUs dedicate more of their area to compute (green), and less to control and caching (orange and blue), than do CPUs. Modified from a diagram in [Fabien Sanglard's blog](https://fabiensanglard.net/cuda), itself likely modifed from a diagram in [the CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).](themed-image://cpu-vs-gpu.svg)

For programs or functions like neural network inference or sequential database
scans for which it is relatively straightforward for programmers to
[express](/gpu-glossary/device-software/cuda-programming-model) the behavior of
[caches](/gpu-glossary/device-hardware/l1-data-cache) — e.g. store a chunk of
each input matrix and keep it in cache for long enough to compute the related
outputs — the result is much higher throughput.
