---
title: What is register pressure?
---

Register pressure is a colorful term used when the
[register file](/gpu-glossary/device-hardware/register-file) is a
[bottleneck](/gpu-glossary/perf/performance-bottleneck).

[Registers](/gpu-glossary/device-software/registers) in the
[Parallel Thread eXecution (PTX)](/gpu-glossary/device-software/parallel-thread-execution)
language are virtual and unlimited, but the
[register files](/gpu-glossary/device-hardware/register-file) of the
[Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor)
are physical and so limited.

The amount of space in the
[register file](/gpu-glossary/device-hardware/register-file) consumed by a
[thread](/gpu-glossary/device-software/thread) is determined by the
[Streaming ASSembler (SASS)](/gpu-glossary/device-software/streaming-assembler)
code for the [kernel](/gpu-glossary/device-software/kernel), and since all
[threads](/gpu-glossary/device-software/thread) in a
[thread block](/gpu-glossary/device-software/thread-block) are scheduled onto
the same [SM](/gpu-glossary/device-hardware/streaming-multiprocessor), the total
space required by a [thread block](/gpu-glossary/device-software/thread-block)
is determined also by the [kernel](/gpu-glossary/device-software/kernel) launch
configuration. As the space allocated per
[thread block](/gpu-glossary/device-software/thread-block) increases, fewer
[thread blocks](/gpu-glossary/device-software/thread-block) can be scheduled
onto the same [SM](/gpu-glossary/device-hardware/streaming-multiprocessor),
reducing [occupancy](/gpu-glossary/perf/occupancy) and making it more difficult
to [hide latency](/gpu-glossary/perf/latency-hiding).

See
[this excellent article by SemiAnalysis](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)
for an account of the relationship between register pressure and key features
added in recent
[Streaming Multiprocessor architectures](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture),
like asynchronous copies (added in Ampere), the
[Tensor Memory Accelerator](/gpu-glossary/device-hardware/tensor-memory-accelerator)
(TMA, added in Hopper), and
[tensor memory](/gpu-glossary/device-hardware/tensor-memory) (added in
Blackwell).

Register pressure also occurs in CPUs, where similar register
[bottlenecks](/gpu-glossary/perf/performance-bottleneck) limit the degree to
which loops can be
[strip-mined during auto-vectorization](https://hogback.atmos.colostate.edu/rr/old/tidbits/intel/macintel/doc_files/source/extfile/optaps_for/common/optaps_vec_mine.htm).
