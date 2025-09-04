---
title: What is warp divergence?
---

Warp divergence occurs when threads within a
[warp](/gpu-glossary/device-software/warp) take different execution paths due to
control flow statements.

For example, consider this [kernel](/gpu-glossary/device-software/kernel):

```cpp
__global__ void divergent_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (data[idx] > 0.5f) {
		    // A
            data[idx] = data[idx] * 4.0f;
        } else {
		    // B
            data[idx] = data[idx] + 2.0f;
        }
        data[idx] = data[idx] * data[idx];
    }
}
```

When the [threads](/gpu-glossary/device-software/thread) within a
[warp](/gpu-glossary/device-software/warp) encounter the data-dependent
conditional, some [threads](/gpu-glossary/device-software/thread) must execute
block A while others must execute block B, depending on the value at
`data[idx]`. Because of this data-dependency and the structural constraints of
the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model)
and its implementation in the
[PTX machine model](/gpu-glossary/device-software/parallel-thread-execution),
there is no way for a programmer or a compiler to avoid this split in control
flow inside of the [warp](/gpu-glossary/device-software/warp).

Instead, the [warp scheduler](/gpu-glossary/device-hardware/warp-scheduler) must
handle concurrent execution of these divergent code paths, which it achieves by
"masking" some [threads](/gpu-glossary/device-software/thread) so that they
don't execute the instruction. This is achieved using predicate
[registers](/gpu-glossary/device-software/registers).

Let's examine the generated
[SASS](/gpu-glossary/device-software/streaming-assembler)
([Godbolt link](https://godbolt.org/z/EGWKb5oWr)) to understand the execution
flow:

```nasm
LDG.E.SYS R4, [R2]                       // L1 load data[idx]
FSETP.GT.AND P0, PT, R4.reuse, 0.5, PT   // L2 set P0 to data[idx] > 0.5
FADD R0, R4, 2                           // L3 store 2 + data[idx] in R0
@P0 FMUL R0, R4, 4                       // L4 in some threads, store 4 * data[idx] in R0
FMUL R5, R0, R0                          // L5 store R0 * R0 in R5
STG.E.SYS [R2], R5                       // L6 store R5 in data[idx]
```

After loading the data into `R4` (`L1`), all 32
[threads](/gpu-glossary/device-software/thread) in the
[warp](/gpu-glossary/device-software/warp) execute `FSETP.GT.AND` concurrently
(`L2`), and each [thread](/gpu-glossary/device-software/thread) gets its own
`P0` value based on the `data` value in `R4`. Then, we have a bit of
[compiler](/gpu-glossary/host-software/nvcc) cleverness: in `L3` _all_
[threads](/gpu-glossary/device-software/thread) execute the code in A, writing
to `R0`. Only those for whom `P0` is true then execute the code in B (`L4`),
over-writing the value written to `R0` in `L3`. On this instruction, the
[warp](/gpu-glossary/device-software/warp) is said to be "divergent". On `L5`,
all [threads](/gpu-glossary/device-software/thread) are back to executing the
same code. Once the
[warp scheduler](/gpu-glossary/device-hardware/warp-scheduler) brings them back
into alignment by issuing the same instruction on the same clock cycle, the warp
has "converged".

This is presumably more efficient than the naïve encoding of the branch into
[SASS](/gpu-glossary/device-software/streaming-assembler), which would instead
predicate both lines `L3` and `L4` — "presumably" in that we can trust the
[compiler](/gpu-glossary/host-software/nvcc) and in that, heuristically, we are
trading use of cheap, plentiful
[CUDA Core](/gpu-glossary/device-hardware/cuda-core) computation for more
expensive flow control. As often in GPU programming, it's better to waste
compute (an unnecessary `FADD` for every execution of `L4`) than to add
complexity, even if it's just a simple predication!

One reason compilers might aggressively avoid divergence is that in early
(pre-Volta) GPUs, divergent [warps](/gpu-glossary/device-software/warp) were
always fully serialized. While warp divergence still reduces efficiency, modern
GPUs with independent thread scheduling don't necessarily experience the full
serialization penalties.
