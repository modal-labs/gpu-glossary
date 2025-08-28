---
title: What is a Texture Processing Cluster?
abbreviation: TPC
---

A Texture Processing Cluster (TPC) is a pair of adjacent
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor).

Before the Blackwell
[SM architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture),
TPCs were not mapped onto any level of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model)'s
[memory hierarchy](/gpu-glossary/device-software/memory-hierarchy) or
[thread hierarchy](/gpu-glossary/device-software/thread-hierarchy).

The fifth-generation [Tensor Cores](/gpu-glossary/device-hardware/tensor-core)
in the Blackwell
[SM architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)
added the "CTA pair" level of the
[Parallel Thread eXecution (PTX)](/gpu-glossary/device-software/parallel-thread-execution)
[thread hierarchy](/gpu-glossary/device-software/thread-hierarchy), which maps
onto TPCs. Many `tcgen05`
[PTX](/gpu-glossary/device-software/parallel-thread-execution) instructions
include a `.cta_group` field that can use a single
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor) (`.cta_group::1`)
or a pair of [SMs](/gpu-glossary/device-hardware/streaming-multiprocessor) in a
TPC (`::2`), which are mapped to `1SM` and `2SM` variants of
[Streaming Assembler (SASS)](/gpu-glossary/device-software/streaming-assembler)
instructions like `MMA`.
