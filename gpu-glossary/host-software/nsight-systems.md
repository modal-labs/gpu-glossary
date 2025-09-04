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

You can find its documentation
[here](https://docs.nvidia.com/nsight-systems/index.html), but
[watching someone use the tool](https://www.youtube.com/watch?v=dUDGO66IadU) is
usually more helpful. For details on how to profile GPU applications on Modal,
see [our documentation](https://modal.com/docs/examples/nsys).
