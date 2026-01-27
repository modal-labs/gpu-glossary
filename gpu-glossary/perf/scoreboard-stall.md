---
title: What is a scoreboard stall?
---

A scoreboard stall occurs when an instruction cannot be issued due to a
dependency on the result of a prior instruction.

A scoreboard is a hardware structure that tracks which
[registers](/gpu-glossary/device-software/registers) are waiting to be written
to by an in-flight instruction. A [warp](/gpu-glossary/device-software/warp)
cannot progress when it is in the
[stalled state](/gpu-glossary/perf/warp-execution-state).

Scoreboard stalls can be classified into two types: short scoreboard stalls and
long scoreboard stalls.

A short scoreboard stall occurs when an instruction is waiting on the result of
a variable latency instruction which does not leave the
[Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor).
This includes slow math instructions on the
[Special Function Unit](/gpu-glossary/device-hardware/special-function-unit)
like `MUFU.EX2` and `MUFU.SQRT` and matrix multiplications on the
[Tensor Core](/gpu-glossary/device-hardware/tensor-core) like `MMA`. It also
includes [shared memory](/gpu-glossary/device-software/shared-memory) operations
like `LDS` and `STS`.

A long scoreboard stall occurs when an instruction is waiting on the result of a
memory operation that leaves the
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor), such as global
memory loads (`LDG`) or stores (`STG`). Long scoreboard stalls dominate
[memory-bound](/gpu-glossary/perf/memory-bound) code.

A [warp](/gpu-glossary/device-software/warp) has 6 scoreboards which the
compiler uses to track data dependencies between instructions.

Some scoreboard information is legible in
[Streaming Assembler (SASS)](/gpu-glossary/device-software/streaming-assembler).
For example, below is what you might see from a `cuobjdump` with the
`--dump-sass` flag:

```nasm
[barrier:  :  :  :  ]  /*line*/  INSTRUCTION Ri, [Rj] ; # format: scoreboard info, line number, instruction, operands
[B------:R-:W2:-:S04]  /*00f0*/  LDG.E.SYS R0, [R2] ;   # Sets scoreboard 2
[B------:R-:W2:-:S01]  /*0100*/  LDG.E.SYS R5, [R4] ;   # `ptxas` intelligently reuses scoreboard 2
...
[B--2---:R-:W-:Y:S08]  /*0150*/  IMAD R0, R0, c[0x0][0x160], R5 ;  # Waits on scoreboard 2
```

We can see that our `IMAD` instruction has a barrier (`B--2---`) on scoreboard
2, indicating that it requires that bit flag to be cleared before it can issue.
Both `LDG` instructions increment (`W2` write) scoreboard 2 when they are issued
so that our `IMAD` instruction will have the correct values in registers `R0`
and `R5` before it executes.

There may be multiple scoreboards to barrier, such as `B01--4-` which means wait
until scoreboards 0,1,4 are all cleared. When the data dependency has been
satisfied, the respective scoreboard is decremented.

Scoreboard reuse can mean that the stall classification from Nsight Compute is
incorrect, as a long and short scoreboard stall may be conflated if they use the
same scoreboard.

For more details about scoreboard implementation on GPUs, see
[Professor Matthew D. Sinclair's slides](https://pages.cs.wisc.edu/~sinclair/courses/cs758/fall2019/handouts/lecture/cs758-fall19-gpu_uarch2.pdf).

[Scoreboarding](https://www.cs.umd.edu/~meesh/411/website/projects/dynamic/scoreboard.html)
for dependency tracking in dynamic instruction scheduling dates back to the
"first supercomputer", the
[Control Data Corporation 6600](https://en.wikipedia.org/wiki/CDC_6600), one of
which
[disproved Euler's sum of powers conjecture](https://www.ams.org/journals/bull/1966-72-06/S0002-9904-1966-11654-3/S0002-9904-1966-11654-3.pdf)
in 1966. Unlike in CPUs, scoreboarding in GPUs isn't used for out-of-order
execution within [threads](/gpu-glossary/device-software/thread)
(instruction-level parallelism), only across them (thread-level parallelism);
see [this NVIDIA patent](https://patents.google.com/patent/US7676657).
