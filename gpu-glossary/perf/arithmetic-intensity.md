---
title: What is arithmetic intensity?
---

Arithmetic intensity is the ratio of arithmetic operations to memory operations
in a [kernel](/gpu-glossary/device-software/kernel).

![In the [roofline model](/gpu-glossary/perf/roofline-model), operational/arithmetic intensity is plotted on the horizontal axis. Diagram adapted from [Williams, Waterman, and Patterson (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf).](themed-image://roofline-model.svg)

A high arithmetic intensity indicates that a
[kernel](/gpu-glossary/device-software/kernel) performs many arithmetic
operations per byte loaded. Due to the high ratio between
[arithmetic bandwidth](/gpu-glossary/perf/arithmetic-bandwidth) and
[memory bandwidth](/gpu-glossary/perf/memory-bandwidth) in modern GPUs, the most
efficient kernels have high arithmetic intensity. That means that when elevating
a memory [bottleneck](/gpu-glossary/perf/performance-bottleneck), we can often
shift work from the memory subsystem to the compute subsystem, saving on
[memory bandwidth](/gpu-glossary/perf/memory-bandwidth) but adding to the load
on the arithmetic units.

For example, compressing data in
[global memory](/gpu-glossary/device-software/global-memory) reduces memory
traffic since fewer bytes need to be transferred, but the compute units must
perform additional decompression operations. If we were previously
[bottlenecked](/gpu-glossary/perf/performance-bottleneck) by memory, this can
improve performance. It also increases the ratio of FLOPs to bytes moved,
increasing the arithmetic intensity.

As another example, the
[backpropagation algorithm](https://www.nature.com/articles/323533a0) creates
long-lived intermediates (activation values) that generally must be stored in
[global memory](/gpu-glossary/device-software/global-memory) during a forward
pass and then retrieved during a backwards pass. In some cases, it is faster to
store only a fraction of these intermediates and then recompute the remainder (a
technique known as [gradient checkpointing](https://arxiv.org/abs/1604.06174)),
which increases arithmetic intensity.

Because different algorithms inherently have different operational and memory
complexities, they inherently scale differently in arithmetic intensity. An
algorithm with O(1) operational complexity and O(N) memory complexity has O(1/N)
arithmetic intensity scaling, while one with O(N) operational complexity and
O(1) memory complexity has O(N) arithmetic intensity scaling.

| **Kernel**                |    **FLOPs** | **Bytes Moved** | **Arithmetic Intensity** | **Arithmetic Intensity Scaling** |
| :------------------------ | -----------: | --------------: | -----------------------: | -------------------------------: |
| SAXPY y = ax + y          |           2N |             12N |                      1/6 |                             O(1) |
| Single-Precision Real FFT | 5/2 N log(N) |             16N |              5/32 log(N) |                        O(log(N)) |
| SGEMM C = A @ B + C       |         2N^3 |           16N^2 |                     N/8  |                             O(N) |

Notably, matrix multiplication scales linearly, i.e. is O(N), in arithmetic
intensity — it is O(N^3) in operational complexity and O(N^2) in memory
complexity. This favorable scaling makes it easy to map applications of matrix
multiplication onto arithmetic-intensity-oriented hardware (see discussion in
the [article on roofline modeling](/gpu-glossary/perf/roofline-model)). It is a
key secret to the success of machine learning algorithms based on matrix
multiplication, like neural networks, in the past few decades.

For a discussion of arithmetic intensity as applied to Bahdanau attention, used
in Transformer neural networks, see
[this paper](https://arxiv.org/abs/2505.21487) by Zadouri, Strauss, and Dao.

The minimum arithmetic intensity required for work to be
[compute-bound](/gpu-glossary/perf/compute-bound) (that is, to be past the ridge
point of the [roofline model](/gpu-glossary/perf/roofline-model)) is a fixed
parameter of a system and so only needs to be derived once. Ridge point
arithmetic intensities for recent NVIDIA data center GPUs appear in the table
below. Notice that the highest ridge point has increased going from the Ampere
to Hopper to Blackwell
[Streaming Multiprocessor architectures](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).

| **System (Compute / Memory)**                                                                                                                               | **[Arithmetic Bandwidth](/gpu-glossary/perf/arithmetic-bandwidth) (TFLOPs/s)** | **[Memory Bandwidth](/gpu-glossary/perf/memory-bandwidth) (TB/s)** | **[Ridge Point](/gpu-glossary/perf/roofline-model) (FLOPs/byte)** |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------- | -----------------------------------------------------------------------------: | -----------------------------------------------------------------: | ----------------------------------------------------------------: |
| [A100 80GB SXM BF16 TC / HBM2e](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) |                                                                            312 |                                                                  2 |                                                               156 |
| [H100 SXM BF16 TC / HBM3](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)                                                            |                                                                            989 |                                                               3.35 |                                                               295 |
| [B200 BF16 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                   |                                                                           2250 |                                                                  8 |                                                               281 |
| [H100 SXM FP8 TC / HBM3](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)                                                             |                                                                           1979 |                                                               3.35 |                                                               592 |
| [B200 FP8 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                    |                                                                           4500 |                                                                  8 |                                                               562 |
| [B200 FP4 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                    |                                                                           9000 |                                                                  8 |                                                              1125 |
