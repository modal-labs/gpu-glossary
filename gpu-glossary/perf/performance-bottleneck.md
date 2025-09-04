---
title: What is a performance bottleneck?
---

The literal neck of a bottle limits the rate at which liquid can be poured; a
metaphorical performance bottleneck in a system limits the rate at which tasks
can be completed.

![[Roofline diagrams](/gpu-glossary/perf/roofline-model) like this one are used to quickly identify performance bottlenecks in throughput-oriented systems. Adapted from [Williams, Waterman, and Patterson (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf).](themed-image://roofline-model.svg)

Bottlenecks are the target of performance optimization. The textbook approach to
optimization is to

- determine the bottleneck,
- elevate the bottleneck until it is no longer such, and
- repeat on the new bottleneck.

This approach is formalized in, for instance, the
["Theory of Constraints" by Eliyahu Goldratt](https://en.wikipedia.org/wiki/Theory_of_constraints)
that helped
[transmit the Toyota approach to manufacturing to manufacturers worldwide](https://www.leanproduction.com/theory-of-constraints/),
[thence to software engineering and operations](https://youtu.be/1jU7iUr-0xE).

In [this talk for Jane Street](https://youtu.be/139UPjoq7Kw?t=1229), Horace He
broke down the work done by the [kernels](/gpu-glossary/device-software/kernel)
of programs run on GPUs into three categories:

- Compute (running floating point operations on
  [CUDA Cores](/gpu-glossary/device-hardware/cuda-core) or
  [Tensor Cores](/gpu-glossary/device-hardware/tensor-core))
- Memory (moving data in the system's
  [memory hierarchy](/gpu-glossary/device-software/memory-hierarchy))
- Overhead (everything else)

And so for GPU [kernels](/gpu-glossary/device-software/kernel), performance
bottlenecks fall into three main\* categories:

- [compute-bound](/gpu-glossary/perf/compute-bound)
  [kernels](/gpu-glossary/device-software/kernel), bottlenecked by the
  [arithmetic bandwidth](/gpu-glossary/perf/arithmetic-bandwidth) of compute
  units, like large matrix-matrix multiplication,
- [memory-bound](/gpu-glossary/perf/memory-bound)
  [kernels](/gpu-glossary/device-software/kernel), bottlenecked by the
  [bandwidth of memory subsystems](/gpu-glossary/perf/memory-bandwidth), like
  large vector-vector multiplication, and
- [overhead-bound](/gpu-glossary/perf/overhead)
  [kernels](/gpu-glossary/device-software/kernel) bottlenecked by latency, like
  small array operations.

[Roofline model](/gpu-glossary/perf/roofline-model) analysis helps quickly
identify whether a program's performance is bottlenecked by
compute/[arithmetic bandwidth](/gpu-glossary/perf/arithmetic-bandwidth) or
[memory bandwidth](/gpu-glossary/perf/memory-bandwidth).

<small>Of course, _any_ resource can become a bottleneck. For instance, power
ingress and heat egress can and does bottleneck some GPUs below their
theoretical maximum performance. See
[this article from NVIDIA](https://developer.nvidia.com/blog/nvidia-sets-new-generative-ai-performance-and-scale-records-in-mlperf-training-v4-0/)
explaining a 4% end-to-end performance improvement by redirecting power from the
L2 cache to the
[Streaming Multiprocessors](/gpu-glossary/device-hardware/streaming-multiprocessor)
or
[this article from Horace He](https://www.thonking.ai/p/strangely-matrix-multiplications)
indicating that matrix multiplication performance varies depending on the input
data via the amount of power demanded by transistor switching. But compute and
memory are the most important resources and the most common bottlenecks.</small>
