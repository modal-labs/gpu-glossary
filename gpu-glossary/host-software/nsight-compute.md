---
title: What is NVIDIA Nsight Compute?
---

NVIDIA Nsight Compute is a profiler for individual
[CUDA kernels](/gpu-glossary/device-software/kernel). Where
[Nsight Systems](/gpu-glossary/host-software/nsight-systems) profiles an entire
system and treats each [kernel](/gpu-glossary/device-software/kernel) as an
opaque block on a timeline, Nsight Compute opens that block up: it reads the
hardware performance counters of the
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor)
to report what happened to a single
[kernel](/gpu-glossary/device-software/kernel) launch, cycle by cycle and
instruction by instruction.

Put another way, [Nsight Systems](/gpu-glossary/host-software/nsight-systems)
tells you _which_ [kernel](/gpu-glossary/device-software/kernel) to optimize.
Nsight Compute tells you _why_ that
[kernel](/gpu-glossary/device-software/kernel) is slow. It is usually a mistake
to reach for the second tool before the first, since a perfectly-tuned
[kernel](/gpu-glossary/device-software/kernel) that accounts for 2% of your
runtime is worth nothing.

Its reports are organized into _sections_, and the sections are a good map of
the [performance](/gpu-glossary/perf) section of this glossary:

- **GPU Speed of Light Throughput** compares achieved throughput against the
  [peak rates](/gpu-glossary/perf/peak-rate) of the compute and memory
  subsystems, including a [roofline chart](/gpu-glossary/perf/roofline-model),
  which is the fastest way to tell whether a
  [kernel](/gpu-glossary/device-software/kernel) is
  [compute-bound](/gpu-glossary/perf/compute-bound) or
  [memory-bound](/gpu-glossary/perf/memory-bound).
- **Compute Workload Analysis** reports instructions per clock and the
  [utilization of each pipe](/gpu-glossary/perf/pipe-utilization).
- **Memory Workload Analysis** charts traffic between the
  [SMs](/gpu-glossary/device-hardware/streaming-multiprocessor), the
  [caches](/gpu-glossary/device-hardware/l1-data-cache), and
  [GPU RAM](/gpu-glossary/device-hardware/gpu-ram), which is where
  [uncoalesced accesses](/gpu-glossary/perf/memory-coalescing) and
  [shared memory bank conflicts](/gpu-glossary/perf/bank-conflict) show up.
- **Scheduler Statistics** and **Warp State Statistics** break down the
  [states of the kernel's warps](/gpu-glossary/perf/warp-execution-state) — how
  many were active, how many were eligible, how often an instruction was
  actually [issued](/gpu-glossary/perf/issue-efficiency), and what the
  [stalled](/gpu-glossary/perf/scoreboard-stall) ones were waiting on.
- **Launch Statistics** summarizes the configuration the
  [kernel](/gpu-glossary/device-software/kernel) was launched with — the
  dimensions of its [grid](/gpu-glossary/device-software/thread-block-grid) and
  [blocks](/gpu-glossary/device-software/thread-block) — along with the
  [registers](/gpu-glossary/device-software/registers) and
  [shared memory](/gpu-glossary/device-software/shared-memory) each
  [block](/gpu-glossary/device-software/thread-block) consumed, and
  **Occupancy** turns those into
  [theoretical and achieved occupancy](/gpu-glossary/perf/occupancy).
- **Source Counters** correlates metrics like
  [branch efficiency](/gpu-glossary/perf/branch-efficiency) and sampled
  [warp](/gpu-glossary/device-software/warp) stall reasons back to individual
  [SASS](/gpu-glossary/device-software/streaming-assembler) instructions and, if
  you compiled with line information, to lines of your
  [CUDA C++](/gpu-glossary/host-software/cuda-c).

Each section can carry _rules_, which are the "expert system" part of the tool:
they inspect the collected metrics and emit prose recommendations, like a note
that your [occupancy](/gpu-glossary/perf/occupancy) is limited by
[shared memory](/gpu-glossary/device-software/shared-memory) rather than by
[registers](/gpu-glossary/device-software/registers).

All of this is expensive, in ways worth understanding. A GPU can only count so
many things at once, so collecting a full set of metrics requires more than one
pass over the [kernel](/gpu-glossary/device-software/kernel). Nsight Compute's
default strategy is _kernel replay_: it saves the
[global memory](/gpu-glossary/device-software/global-memory) the
[kernel](/gpu-glossary/device-software/kernel) can reach, runs the launch again
and again, and restores whatever the
[kernel](/gpu-glossary/device-software/kernel) wrote between passes so that each
pass sees identical inputs. It also serializes
[kernel](/gpu-glossary/device-software/kernel) launches — only one
[kernel](/gpu-glossary/device-software/kernel) at a time, and only one process
profiling a given device at a time — and it locks
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor) clocks by default,
for reproducibility.

The consequences: profiling a single
[kernel](/gpu-glossary/device-software/kernel) can take orders of magnitude
longer than running it, wall-clock timings taken with host-side timers under the
profiler are meaningless, and any performance effect that depends on
[kernels](/gpu-glossary/device-software/kernel) running concurrently is
invisible by construction. So you filter: profile a handful of launches of one
[kernel](/gpu-glossary/device-software/kernel), not every launch in your
program.

Reading hardware counters is also a privileged operation. Nsight Compute must
reserve the driver's performance monitor, which means administrator access on
most systems — or a driver configured to permit non-admin profiling — and which
it cannot share with other consumers of the same counters, like DCGM or a client
of [CUPTI](/gpu-glossary/host-software/cupti)'s profiling API. The resulting
`ERR_NVGPUCTRPERM` is the single most common thing standing between a programmer
and their first profile.

Nsight Compute ships as a GUI and as a command-line tool, `ncu`. It replaced
`nvprof` and the NVIDIA Visual Profiler, which lose metric collection on GPUs of
[compute capability](/gpu-glossary/device-software/compute-capability) 7.5 and
don't support 8.0 or higher at all.

You can find its documentation [here](https://docs.nvidia.com/nsight-compute/),
in particular the
[Kernel Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html),
which doubles as a reference for the metrics it collects. For details on
profiling GPU applications on Modal, see
[our documentation](https://modal.com/docs/examples/torch_profiling).
