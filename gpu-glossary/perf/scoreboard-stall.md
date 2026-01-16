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
Long scoreboard stalls typically indicate [memory-bound](/gpu-glossary/perf/memory-bound) code.

A warp has 6 scoreboards which the compiler can use to track data dependencies between instructions. For example:

```nasm
[B------:R-:W2:-:S04]  /*00f0*/  LDG.E.SYS R0, [R2] ;   # Sets scoreboard 2
[B------:R-:W2:-:S01]  /*0100*/  LDG.E.SYS R5, [R4] ;   # `ptxas` intelligently reuses scoreboard 2 
...
[B--2---:R-:W-:Y:S08]  /*0150*/  IMAD R0, R0, c[0x0][0x160], R5 ;  # Waits on scoreboard 2
```

We can see that our `IMAD` instruction has a barrier (`B------`) on scoreboard 2, indicating that it requires that bit flag to
be cleared before it can issue. Both `LDG` instructions increment (`W-` write) scoreboard 2 when they are issued
so that our `IMAD` instruction will have the correct values in registers `R0` and `R5` before it executes. 

There may be multiple scoreboards to barrier, such as `B01--4-` which means wait until scoreboards 0,1,4 are all cleared.

Hardware automatically clears the scoreboard when the data arrives. 

For more details about scoreboard implementation on GPUs, see [Professor Matthew D. Sinclair's slides](https://pages.cs.wisc.edu/~sinclair/courses/cs758/fall2019/handouts/lecture/cs758-fall19-gpu_uarch2.pdf).
