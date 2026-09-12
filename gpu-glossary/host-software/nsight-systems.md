---
title: What is NVIDIA Nsight Systems?
---

NVIDIA Nsight Systems is a performance debugging tool for
[CUDA C++](/gpu-glossary/host-software/cuda-c) programs. It combines profiling,
tracing, and expert systems analysis in a GUI.

No one wakes up and says "today I want to write a program that runs on a hard to
use, expensive piece of hardware using a proprietary software stack". Instead,
GPUs are selected when normal computing hardware doesn't perform well enough to
solve a computing problem. So
[almost all GPU programs are performance-sensitive](/gpu-glossary/perf), and the
performance debugging workflows supported by Nsight Systems or other tools built
on top of the
[CUDA Profiling Tools Interface](/gpu-glossary/host-software/cupti) are
mission-critical.

Nsight Systems profiles an entire _system_: both the CPU host and the GPU
device, and, if you like, many processes and many GPUs across many nodes. Its
central artifact is a timeline. On it you'll find

- [CUDA Runtime](/gpu-glossary/host-software/cuda-runtime-api) and
  [CUDA Driver](/gpu-glossary/host-software/cuda-driver-api) API calls made by
  each host thread, like [kernel](/gpu-glossary/device-software/kernel) launches
  and memory copies,
- the corresponding device-side activity — which
  [kernel](/gpu-glossary/device-software/kernel) was resident on which GPU when,
  and for how long,
- calls into libraries like [cuBLAS](/gpu-glossary/host-software/cublas),
  [cuDNN](/gpu-glossary/host-software/cudnn), NCCL, and MPI,
- user-defined ranges, via the NVIDIA Tools Extension (NVTX) API, which is how
  framework-level operations from e.g. PyTorch get onto the timeline, and
- CPU-side samples, with backtraces, plus operating system calls.

Because it consumes the
[CUDA Profiling Tools Interface](/gpu-glossary/host-software/cupti), which
synchronizes timestamps across host and device, these tracks line up: you can
see a host thread call into [`libcudart`](/gpu-glossary/host-software/libcudart)
and watch the [kernel](/gpu-glossary/device-software/kernel) it launched appear
on the device some microseconds later.

Critically, Nsight Systems does not look inside those
[kernels](/gpu-glossary/device-software/kernel). A
[kernel](/gpu-glossary/device-software/kernel) is an opaque block on the
timeline, annotated with its name, its duration, and the dimensions of its
[thread block grid](/gpu-glossary/device-software/thread-block-grid) — not with
anything about how its [warps](/gpu-glossary/device-software/warp) fared inside
an [SM](/gpu-glossary/device-hardware/streaming-multiprocessor). That is
[Nsight Compute](/gpu-glossary/host-software/nsight-compute)'s job.

So the questions Nsight Systems answers are the ones _between_ and _around_
[kernels](/gpu-glossary/device-software/kernel). Is the GPU doing anything at
all, or is it waiting on the host — that is, are we bound by
[overhead](/gpu-glossary/perf/overhead)? Are there gaps between
[kernels](/gpu-glossary/device-software/kernel) that
[CUDA Graphs](/gpu-glossary/host-software/cuda-graph) could close? Do copies to
the device overlap with computation, or serialize against it? Which
[kernel](/gpu-glossary/device-software/kernel) actually dominates the run and is
therefore worth optimizing at all?

That last question is why Nsight Systems comes first in a performance
engineering workflow. It traces rather than replays, and it neither serializes
[kernel](/gpu-glossary/device-software/kernel) launches nor locks clocks, so its
overhead is low and the durations it reports are close to the ones your users
experience. Once it has identified a
[kernel](/gpu-glossary/device-software/kernel) on the critical path that isn't
meeting expectations, you hand that
[kernel](/gpu-glossary/device-software/kernel) to
[Nsight Compute](/gpu-glossary/host-software/nsight-compute) — in the GUI,
literally by right-clicking the launch on the timeline.

You can find its documentation
[here](https://docs.nvidia.com/nsight-systems/index.html), but
[watching someone use the tool](https://www.youtube.com/watch?v=dUDGO66IadU) is
usually more helpful. For details on how to profile GPU applications on Modal,
see [our documentation](https://modal.com/docs/examples/torch_profiling).
