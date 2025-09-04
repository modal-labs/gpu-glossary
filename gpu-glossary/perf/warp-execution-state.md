---
title: What is warp execution state?
---

The state of the [warps](/gpu-glossary/device-software/warp) running a [kernel](/gpu-glossary/device-software/kernel) is described with a number of non-exclusive adjectives: active, stalled, eligible, and selected.

![Warp execution states are indicated by color and transparency. Diagram inspired by the [*CUDA Techniques to Maximize Compute and Instruction Throughput*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/) talk at GTC 2025.](themed-image://cycles.svg)

A [warp](/gpu-glossary/device-software/warp) is considered *active* from the time its [threads](/gpu-glossary/device-software/thread) begin executing to the time when all [threads](/gpu-glossary/device-software/thread) in the [warp](/gpu-glossary/device-software/warp) have exited from the [kernel](/gpu-glossary/device-software/kernel). Active [warps](/gpu-glossary/device-software/warp) form the pool from which [warp schedulers](/gpu-glossary/device-hardware/warp-scheduler) select candidates for instruction issue each cycle (i.e. to be put in one of the issue slots).

The maximum number of active [warps](/gpu-glossary/device-software/warp) per [Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor) varies by [architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture) and is listed in [NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=compute%2520capability#compute-capabilities) for [Compute Capability](/gpu-glossary/device-software/compute-capability). For instance, on an H100 SXM GPU with [Compute Capability](/gpu-glossary/device-software/compute-capability) 9.0, there can be up to 64 active [warps](/gpu-glossary/device-software/warp) per [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) (2048 threads). Note that active [warps](/gpu-glossary/device-software/warp) are not necessarily executing instructions. There are active [warps](/gpu-glossary/device-software/warp) in all but one slot+cycle in the diagram above — a high [occupancy](/gpu-glossary/perf/occupancy).

An *eligible* [warp](/gpu-glossary/device-software/warp) is an active [warp](/gpu-glossary/device-software/warp) that is ready to issue its next instruction. For a [warp](/gpu-glossary/device-software/warp) to be eligible, the following must be true:

- the next instruction has been fetched,
- the required execution unit is available,
- all instruction dependencies have been resolved, and
- no synchronization barriers block execution.

Eligible [warps](/gpu-glossary/device-software/warp) represent the immediate candidates for instruction issue by the [warp scheduler](/gpu-glossary/device-hardware/warp-scheduler). Eligible [warps](/gpu-glossary/device-software/warp) appear on all cycles but cycle n + 2 in the diagram above. Having no eligible [warps](/gpu-glossary/device-software/warp) on many cycles can be bad for performance, especially if you are primarily using lower latency arithmetic units like [CUDA Cores](/gpu-glossary/device-hardware/cuda-core).

A *stalled* [warp](/gpu-glossary/device-software/warp) is an active [warp](/gpu-glossary/device-software/warp) that cannot issue its next instruction due to unresolved dependencies or resource conflicts. [Warps](/gpu-glossary/device-software/warp) become stalled for various reasons including:

- execution dependencies, i.e. they must wait for results from previous arithmetic instructions,
- memory dependencies, i.e. they must wait for results from previous memory operations,
- pipeline conflicts, i.e. the execution resources are currently occupied.

When warps are stalled on accesses to shared memory or on long-running arithmetic instructions, they are said to be stalled on the "short scoreboard". When warps are stalled on accesses to GPU RAM, they are said to be stalled on the "long scoreboard". These are hardware units inside the [warp scheduler](/gpu-glossary/device-hardware/warp-scheduler). [Scoreboarding](https://www.cs.umd.edu/~meesh/411/website/projects/dynamic/scoreboard.html) is a technique for dependency tracking in dynamic instruction scheduling that dates back to the "first supercomputer", the [Control Data Corporation 6600](https://en.wikipedia.org/wiki/CDC_6600), one of which [disproved Euler's sum of powers conjecture](https://www.ams.org/journals/bull/1966-72-06/S0002-9904-1966-11654-3/S0002-9904-1966-11654-3.pdf) in 1966. Unlike in CPUs, scoreboarding isn't used for out-of-order execution within [threads](/gpu-glossary/device-software/thread) (instruction-level parallelism), only across them (thread-level parallelism); see [this NVIDIA patent](https://patents.google.com/patent/US7676657).

Stalled [warps](/gpu-glossary/device-software/warp) appear in multiple slots in each cycle in the diagram above. Stalled [warps](/gpu-glossary/device-software/warp) are not inherently bad — a large collection of concurrently stalled [warps](/gpu-glossary/device-software/warp) might be necessary to [hide latency](/gpu-glossary/perf/latency-hiding) from long-running instructions, like memory loads or [Tensor Core](/gpu-glossary/device-hardware/tensor-core) instructions like `HMMA`, which [can run for dozens of cycles](https://arxiv.org/abs/2206.02874).

A *selected* [warp](/gpu-glossary/device-software/warp) is an eligible [warp](/gpu-glossary/device-software/warp) chosen by the [warp scheduler](/gpu-glossary/device-hardware/warp-scheduler) to receive an instruction during the current cycle. Each cycle, [warp schedulers](/gpu-glossary/device-hardware/warp-scheduler) look at their pool of eligible [warps](/gpu-glossary/device-software/warp), select one if there are any, and issue it an instruction. There is a selected [warp](/gpu-glossary/device-software/warp) on each cycle with an eligible [warp](/gpu-glossary/device-software/warp). The fraction of [active cycles](/gpu-glossary/perf/active-cycles) on which a [warp](/gpu-glossary/device-software/warp) is selected and an instruction is issued is the [issue efficiency](/gpu-glossary/perf/issue-efficiency).
