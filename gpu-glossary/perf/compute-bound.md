---
title: What does it mean to be compute-bound?
---

[Kernels](/gpu-glossary/device-software/kernel) that are compute-bound are
limited by the [arithmetic bandwidth](/gpu-glossary/perf/arithmetic-bandwidth)
of the [CUDA Cores](/gpu-glossary/device-hardware/cuda-core) or
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core).

![In the [roofline diagram](/gpu-glossary/perf/roofline-model) above, [kernels](/gpu-glossary/device-software/kernel) underneath the blue line are compute-bound. Diagram adapted from [Williams, Waterman, and Patterson (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf).](themed-image://roofline-model.svg)

Compute-bound kernels are characterized by high
[arithmetic intensity](/gpu-glossary/perf/arithmetic-intensity) (many arithmetic
operations per byte of memory loaded or stored).
[Utilization of arithmetic pipes](/gpu-glossary/perf/pipe-utilization) is the
limiting factor for a compute-bound kernel.

Technically, compute-boundedness is only defined for a single
[kernel](/gpu-glossary/device-software/kernel), as part of the
[roofline model](/gpu-glossary/perf/roofline-model), but with a bit of squinting
it can be generalized to cover the multiple
[kernels](/gpu-glossary/device-software/kernel) that make up a typical workload.

Large diffusion model inference workloads are generally compute-bound.
Contemporary large language model inference workloads are often compute-bound
during batch prefill/prompt processing, when each weight can be loaded into
[shared memory](/gpu-glossary/device-software/shared-memory) once and then used
across many tokens.

Let's do a simple estimation, inspired by
[kipperrii](https://twitter.com/kipperrii)'s
[Transformer inference arithmetic](https://kipp.ly/transformer-inference-arithmetic)
framework, of the minimum latency between tokens (inter-token latency or time
per output token) for compute-bound Transformer language model inference. Assume
the model has 500B parameters, stored in 16-bit precision, for a total of 1 TB.
This model will perform roughly one trillion floating point operations (one
multiply and one accumulate per parameter) per batch element. Run on a GPU with
one petaFLOP/s of
[arithmetic bandwidth](/gpu-glossary/perf/arithmetic-bandwidth) for 16-bit
matrix math, the minimum latency between tokens, assuming compute-boundedness,
is one millisecond per batch element.

Note that for this GPU to be compute-bound at batch size one, it would need a
[memory bandwidth](/gpu-glossary/perf/memory-bandwidth) of 1 PB/s (so that it
can load all 1 TB of weights in one ms). Contemporary
[memory bandwidths](/gpu-glossary/perf/memory-bandwidth) are in the TB/s range,
and so batches of hundreds of inputs are required to provide sufficient
[arithmetic intensity](/gpu-glossary/perf/arithmetic-intensity) for execution to
be compute-bound.

For more on LLM inference, see our
[LLM Engineer's Almanac](https://modal.com/llm-almanac/summary).
