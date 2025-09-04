---
title: What does it mean to be memory-bound?
---

[Kernels](/gpu-glossary/device-software/kernel) that are memory-bound are limited by the [memory bandwidth](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) of the GPU.

![Roofline diagrams, like the one above, help identify whether a program's performance is bottlenecked by compute power, memory bandwidth, or something else Diagram adapted from [Williams, Waterman, and Patterson (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf).](GPU%20Performance%20Glossary%202251e7f1694980bd93e4f67a75c6e489/terminal-roofline-model(1)%203.png)

Specifically, they are limited by [the bandwidth](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) between the [GPU RAM](https://modal.com/gpu-glossary/device-hardware/gpu-ram) and the [local cache](https://modal.com/gpu-glossary/device-hardware/l1-data-cache) of the [Streaming Multiprocessors](https://modal.com/gpu-glossary/device-hardware/streaming-multiprocessor), because the problems of interest for GPU performance generally have [working set sizes](https://en.wikipedia.org/wiki/Working_set_size) much larger than any higher level of the [memory hierarchy](https://modal.com/gpu-glossary/device-software/memory-hierarchy).

Memory-bound kernels have a lower [arithmetic intensity](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) (fewer operations per byte moved), relative to the ridge point of their [roofline model](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21).

Technically, memory-boundedness is only defined for a single [kernel](/gpu-glossary/device-software/kernel),
as part of the [roofline model](/gpu-glossary/perf/roofline-model),
but with a bit of squinting it can be generalized to cover the multiple [kernels](/gpu-glossary/device-software/kernel)
that make up a typical workload.

Contemporary large language model inference workloads are often memory-bound during the decode/output generation stage, when the weights must be loaded once in each forward pass.
That happens once per output token, unless multi-token prediction or speculative decoding are used,
which makes it easy to calculate the minimum latency between tokens (intertoken latency or time per output token) for memory-bound Transformer large language model inference.

Assume the model has 500B parameters, stored in 16-bit precision, for a total of 1 TB.
If we run inference on a single GPU with a [memory bandwidth](/gpu-glossary/perf/memory-bandwidth) of 10 TB/s,
we can load the weights once every 100 ms, and that puts a lower bound on our intertoken latency.
By batching multiple inputs together, we can linearly increase the number of floating point operations done per parameter loaded
(the [arithmetic intensity](/gpu-glossary/perf/arithmetic-intensity)),
in principle up the point of [compute-boundedness](/gpu-glossary/perf/compute-bound),
without incurring any additional latency, which implies that the throughput improves linearly in the batch size.

For more on LLM inference, see our [LLM Engineer's Almanac](https://modal.com/llm-almanac/summary).
