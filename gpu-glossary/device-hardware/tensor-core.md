---
title: What is a Tensor Core?
---

Tensor Cores are GPU [cores](/gpu-glossary/device-hardware/core) that operate on
entire matrices with each instruction.

![The internal architecture of an H100 SM. Note the larger size and lower number of Tensor Cores. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](themed-image://gh100-sm.svg)

Operating on more data for a single instruction fetch dramatically reduces power
requirements, which unlocks increased performance (see
[this talk](https://youtu.be/kLiwvnr4L80?t=868) by Bill Dally, Chief Scientist
at NVIDIA). Since their introduction in the Volta
[Streaming Multiprocessor (SM) Architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)
generation, they have been the only way to achieve the highest arithmetic
throughput on NVIDIA GPUs.

As an example, the `HMMA16.16816.F32`
[SASS](/gpu-glossary/device-software/streaming-assembler) instruction calculates
D = AB + C for matrices A, B, C, and D (where C is often the same physical
matrix as D). The `MMA` stands for "Matrix Multiply and Accumulate". `HMMA16`
indicates that the inputs are half-precision (`16` bits) and the `F32` indicates
that the outputs are accumulated into `32` bit (aka single-precision) floats.

`16816` is not single number larger than 16,000. Instead, the string of numbers
`16`, `8`, `16` denote the dimensions of the matrices. These dimensions are
generally named `m`, `k`, and `n` by NVIDIA, for example in
[PTX](/gpu-glossary/device-software/parallel-thread-execution) instructions. The
outer dimensions of A and B, aka `m` and `n`, come first and last, respectively,
and the shared inner dimension for the accumulation, `k`, is in the middle.
Multiplying these out, we see that the `HMMA16.16816.32` instruction performs 16
× 8 × 16 = 2,048 multiply-accumulate (MAC) operations.

Note that a single instruction in a single
[thread](/gpu-glossary/device-software/thread) does not produce the entire
matrix multiplication. Instead, the 32 threads of a
[warp](/gpu-glossary/device-software/warp) cooperatively produce the result by
executing the instruction together. Most of the per-instruction power overhead
is in decoding, which is shared across a
[warp](/gpu-glossary/device-software/warp) thanks to the
[warp scheduler](/gpu-glossary/device-hardware/warp-scheduler). But even spread
across those 32 threads, that's 64 = 2,048 ÷ 32 MACs per instruction.

For this reason, it is helpful to think of Tensor Cores, and similar hardware
like the systolic arrays in Google Tensor Processing Units (TPUs), as a form of
[complex instruction set computer (CISC)](https://www.omgwiki.org/ddsf/doku.php?id=ddsf:public:guidebook:06_append:glossary:c:cisc)
hardware. For more on this perspective, applied to TPUs, see
[this talk by computer architect David Patterson](https://youtu.be/fhHAArxwzvQ?t=2072),
who also
[coined the terms CISC and RISC](https://www.semanticscholar.org/paper/4d3a941a5749dbf0dd39554f12597c449c3c07ff).

That assembler-level instruction might be produced by a compiler to implement
[PTX-level](/gpu-glossary/device-software/parallel-thread-execution)
matrix-multiply-and-accumlate instructions like `wmma` (documented
[here](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-instructions)).
Those instructions also calculate D = AB + C for matrices A, B, C, and D, but
are generally compiled into many individual
[SASS](/gpu-glossary/device-software/streaming-assembler) Tensor Core
instructions that operate on smaller matrices.

These instructions from the
[PTX](/gpu-glossary/device-software/parallel-thread-execution) instruction set
architecture are exposed in the high-level
[CUDA C++ programming language](/gpu-glossary/host-software/cuda-c) as
intrinsics.

In reverse order, a line of [CUDA C++](/gpu-glossary/host-software/cuda-c)
coding a matrix multiplication `C = A @ B`, of two 16 by 16 matrices, like

```cpp
wmma::mma_sync(c, a, b, c);
```

where `c` is initialized to all zeros, and the first appearance indicates it is
also the output, might be compiled by [`nvcc`](/gpu-glossary/host-software/nvcc)
to the [PTX](/gpu-glossary/device-software/parallel-thread-execution)
intermediate representation as

```ptx
wmma.mma.sync.aligned.col.row.m16n16k16.f32.f32 {%f2, %f3, %f4, %f5, %f6, %f7, %f8, %f9}, {%r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9}, {%r10, %r11, %r12, %r13, %r14, %r15, %r16, %r17}, {%f1, %f1, %f1, %f1, %f1, %f1, %f1, %f1};
```

and then finally compiled by `ptxas` to
[SASS](/gpu-glossary/device-software/streaming-assembler) as

```sass
HMMA.1688.F32 R20, R12, R11, RZ   // 1
HMMA.1688.F32 R24, R12, R17, RZ   // 2
HMMA.1688.F32 R20, R14, R16, R20  // 3
HMMA.1688.F32 R24, R14, R18, R24  // 4
```

The operands to each `HMMA` instruction can be read, in order, as
`D = A @ B + C`. For example, instruction 3 uses
[register](/gpu-glossary/device-hardware/register-file) 20 for its output `D`,
registers 14 and 16 for its inputs `A` and `B`, respectively, and re-uses
register 20 for its input `C`, effecting the computation `C += A @ B`.

This program partitions the full 16 by 16 square matrix multiplication into four
separate instructions, each itself a matrix multiplication of a 16 by 8 matrix
with an 8 by 8 matrix. Similarly, programs running large-scale matrix
multiplications must break their work down into smaller matrix multiplications,
like the 16 by 16 square matrix multiplication performed by the `mma_sync` call
we are dissecting. We walk through this program below.

![Register usage in a Tensor Core MMA for C = A @ B. The R11, R17, R16, and R18 registers are used in instructions 1, 2, 3, and 4, respectively. See surrounding text for details.](themed-image://tensor-core-mma.svg)

The first two instructions compute the matrix multiplication of the first eight
columns of the input `a`, from `R12`, with the first eight rows of the input
`b`, from `R11` and `R17`, producing a 16 by 16 matrix, which is stored in `R20`
and `R24`. This is a sort of "outer product": a tall and skinny matrix
mutliplied by a short and wide matrix. (`RZ` is a special-purpose "register"
that contains the value `Z`ero).

The second two instructions compute a similar "outer product" for the second
eight columns of `a` and second eight rows of `b`, accumulating with the output
of the first two instructions to produce the final value in `c`.

Put another way: within a block of eight rows out of eight columns in B and
within an entire column of A, a number of multiplications and additions occur
inside the Tensor Core concurrently, with respect to the instruction, to
implement a matrix multiplication. Each instruction handles all `m` rows of A
for the given block of rows and columns from B. Together, they handle the full
matrix multiplication.

Explore [this compiler output on Godbolt](https://godbolt.org/z/e6cqn8491) if
you want to dive deeper. Note that this is far from a
[utilization-maximizing](https://modal.com/blog/gpu-utilization-guide) matrix
multiplication using Tensor Cores! For that, see
[this worklog by Pranjal Shandkar](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog).

Programming Hopper and Blackwell Tensor Cores for maximum performance cannot be
done in pure CUDA C++, requiring instead
[PTX](/gpu-glossary/device-software/parallel-thread-execution) intrinsics for
both computation and memory. It is generally recommended to instead use existing
kernels from kernel libraries like
[cuBLAS (CUDA Basic Linear Algebra Subroutines)](https://docs.nvidia.com/cuda/cublas/)
or higher-level kernel programming interfaces like
[CUTLASS (CUDA Templates for Linear Algebra Subroutines)](https://github.com/NVIDIA/cutlass).
For an introduction to CUTLASS, see
[this blog post series by Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/).

Tensor Cores are much larger and less numerous than CUDA Cores. An H100 SXM5 has
only four Tensor Cores per
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor), i.e. one per
[Warp Scheduler](/gpu-glossary/device-hardware/warp-scheduler), compared to
hundreds of [CUDA Cores](/gpu-glossary/device-hardware/cuda-core).

Tensor Cores were introduced in the V100 GPU, which represented a major
improvement in the suitability of NVIDIA GPUs for large neural network worloads.
For more, see
[the NVIDIA white paper introducing the V100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf).

The internals of Tensor Cores are unknown, and likely differ from
[SM Architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)
to
[SM Architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).
They are commonly assumed to be systolic arrays, like TPUs, but there is no
consensus in the microbenchmarking literature.
