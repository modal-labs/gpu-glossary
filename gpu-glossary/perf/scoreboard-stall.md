---
title: What is a scoreboard stall?
---

A scoreboard is a hardware structure that tracks which registers are waiting to be written to by an in-flight instruction.
When an instruction cannot be issued due to a dependency on the result of a prior instruction, this is known as a
scoreboard stall. A scoreboard stall prevents a [warp](/gpu-glossary/device-software/warp) from making progress. 

Scoreboard stalls can be classified into two types: short scoreboard stalls and long scoreboard
stalls.

A short scoreboard stall occurs when an instruction is waiting on the result of a variable latency instruction which
does not leave the [Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor), such as slow
math instructions on the [Special Function Unit](/gpu-glossary/device-hardware/special-function-unit) like `MUFU.EX2` and `MUFU.SQRT`, or [shared memory](/gpu-glossary/device-software/shared-memory) operations like `LDS` and `STS`.

A long scoreboard stall occurs when an instruction is waiting on the result of a memory operation that leaves the
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor), such as global memory loads (`LDG`) or stores (`STG`).

A good mental model for a scoreboard is a bit array with 1 bit for each register. 
- If a bit is not set: the register has valid data
- If a bit is set: the register has stale data

In practice, the implementation is much more complex for performance reasons, for a deep dive into scoreboard
implementation on GPUs, see [Professor Matthew D. Sinclair's slides](https://pages.cs.wisc.edu/~sinclair/courses/cs758/fall2019/handouts/lecture/cs758-fall19-gpu_uarch2.pdf).
