---
title: What is SM utilization?
---

SM utilization measures the percentage of time that
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor)
are executing instructions.

SM utilization is akin to the more familiar
[kernel utilization](https://modal.com/blog/gpu-utilization-guide) reported by
[`nvidia-smi`](/gpu-glossary/host-software/nvidia-smi), but more fine-grained.
Instead of reporting the fraction of time that a
[kernel](/gpu-glossary/device-software/kernel) is executing anywhere on the GPU,
it reports the fraction of time all
[SMs](/gpu-glossary/device-hardware/streaming-multiprocessor) spend executing
[kernels](/gpu-glossary/device-software/kernel). If a
[kernel](/gpu-glossary/device-software/kernel) uses only one
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor), e.g. because it
only has one [thread block](/gpu-glossary/device-software/thread-block), then it
will achieve 100% GPU utilization while it is active, but the SM utilization
will be at most one over the number of
[SMs](/gpu-glossary/device-hardware/streaming-multiprocessor) — under 1% in an
H100 GPU.

[As with GPU utilization but unlike CPU utilization](https://modal.com/blog/gpu-utilization-guide),
SM utilization should be high, even up to 100%.

But even though SM utilization is finer-grained than GPU utilization, it still
isn't fine-grained enough to capture how well the GPU's compute resources are
being used. If SM utilization is high, but performance is still inadequate,
programmers should check
[pipe utilization](/gpu-glossary/perf/pipe-utilization), which measures how
effectively each SM uses its internal functional units. High SM utilization with
low [pipe utilization](/gpu-glossary/perf/pipe-utilization) indicates that your
[kernel](/gpu-glossary/device-software/kernel) is running on many SMs but not
fully utilizing the computational resources within each one.
