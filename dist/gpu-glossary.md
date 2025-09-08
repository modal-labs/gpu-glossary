# GPU Glossary

## README

<pre class="text-xs md:text-base font-mono whitespace-pre">
 ██████╗ ██████╗ ██╗   ██╗
██╔════╝ ██╔══██╗██║   ██║
██║  ███╗██████╔╝██║   ██║
██║   ██║██╔═══╝ ██║   ██║
╚██████╔╝██║     ╚██████╔╝
 ╚═════╝ ╚═╝      ╚═════╝
 ██████╗ ██╗      ██████╗ ███████╗███████╗ █████╗ ██████╗ ██╗   ██╗
██╔════╝ ██║     ██╔═══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗╚██╗ ██╔╝
██║  ███╗██║     ██║   ██║███████╗███████╗███████║██████╔╝ ╚████╔╝
██║   ██║██║     ██║   ██║╚════██║╚════██║██╔══██║██╔══██╗  ╚██╔╝
╚██████╔╝███████╗╚██████╔╝███████║███████║██║  ██║██║  ██║   ██║
 ╚═════╝ ╚══════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝
 </pre>

We wrote this glossary to solve a problem we ran into working with GPUs here at
[Modal](/): the documentation is fragmented, making it difficult to connect
concepts at different levels of the stack, like
[Streaming Multiprocessor Architecture](#streaming-multiprocessor-architecture),
[Compute Capability](#compute-capability), and
[nvcc compiler flags](#host-software).

So we've read the
[PDFs from NVIDIA](https://docs.nvidia.com/cuda/pdf/PTX_Writers_Guide_To_Interoperability.pdf),
lurked in the [good Discords](https://discord.gg/gpumode), and even bought
[dead-tree textbooks](https://www.amazon.com/Professional-CUDA-Programming-John-Cheng/dp/1118739329)
to put together a glossary that spans the whole stack in one place.

This glossary, unlike a PDF or a Discord or a book, is a _hypertext document_ --
all pages are inter-linked with one another, so you can jump down to read about
the [Warp Scheduler](#warp-scheduler) so you can
better understand the [threads](#thread) that you
came across in the article on the
[CUDA programming model](#cuda-c).

You can also read it linearly. To navigate between pages, use the arrow keys,
the arrows at the bottom of each page, or the table of contents (in the sidebar
on desktop or in the hamburger menu on mobile).

The source for the glossary is available
[on GitHub](https://github.com/modal-labs/gpu-glossary).

## Device Hardware

These terms and technologies are physical components of the GPU — the "device"
in NVIDIA's lingo.

### CUDA (Device Architecture)

CUDA stands for _Compute Unified Device Architecture_. Depending on the context,
"CUDA" can refer to multiple distinct things: a high-level device architecture,
a
[parallel programming model](#cuda-programming-model)
for architectures with that design, or a
[software platform](#cuda-software-platform) that
extends high-level languages like C to add that programming model.

The vision for CUDA is laid out in the
[Lindholm et al., 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf)
white paper. We highly recommend this paper, which is the original source for
many claims, diagrams, and even specific turns of phrase in NVIDIA's
documentation.

Here, we focus on the _device architecture_ part of CUDA. The core feature of a
"compute unified device architecture" is simplicity, relative to preceding GPU
architectures.

Prior to the GeForce 8800 and the Tesla data center GPUs it spawned, NVIDIA GPUs
were designed with a complex pipeline shader architecture that mapped software
shader stages onto heterogeneous, specialized hardware units. This architecture
was challenging for the software and hardware sides alike: it required software
engineers to map programs onto a fixed pipeline and forced hardware engineers to
guess the load ratios between pipeline steps.

![A diagram of a fixed-pipeline device architecture (G71). Note the presence of a separate group of processors for handling fragment and vertex shading. Adapted from [Fabien Sanglard's blog](https://fabiensanglard.net/cuda/).](dist/diagrams/light-fixed-pipeline-g71.png)

GPU devices with a unified architecture are much simpler: the hardware units are
entirely uniform, each capable of a wide array of computations. These units are
known as
[Streaming Multiprocessors (SMs)](#streaming-multiprocessor)
and their main subcomponents are the
[CUDA Cores](#cuda-core) and (for recent GPUs)
[Tensor Cores](#tensor-core).

![A diagram of a compute unified device architecture (G80). Note the absence of distinct processor types — all meaningful computation occurs in the identical [Streaming Multiprocessors](#streaming-multiprocessor) in the center of the diagram, fed with instructions for vertex, geometry, and pixel threads. Modified from [Peter Glazkowsky's 2009 white paper on the Fermi Architecture](https://www.nvidia.com/content/pdf/fermi_white_papers/p.glaskowsky_nvidia%27s_fermi-the_first_complete_gpu_architecture.pdf).](/dist/diagrams/light-cuda-g80.png)

For an accessible introduction to the history and design of CUDA hardware
architectures, see [this blog post](https://fabiensanglard.net/cuda/) by Fabien
Sanglard. That blog post cites its (high-quality) sources, like NVIDIA's
[Fermi Compute Architecture white paper](https://www.nvidia.com/content/pdf/fermi_white_papers/nvidia_fermi_compute_architecture_whitepaper.pdf).
The white paper by
[Lindholm et al. in 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf)
introducing the Tesla architecture is both well-written and thorough. The
[NVIDIA whitepaper for the Tesla P100](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf)
is less scholarly but documents the introduction of a number of features that
are critical for today's large-scale neural network workloads, like NVLink and
[on-package high-bandwidth memory](#gpu-ram).

### Streaming Multiprocessor

When we [program GPUs](#cuda-software-platform), we
produce
[sequences of instructions](#streaming-assembler)
for its Streaming Multiprocessors to carry out.

![A diagram of the internal architecture of an H100 GPU's Streaming Multiprocessors. GPU cores appear in green, other compute units in maroon, scheduling units in orange, and memory in blue. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](/dist/diagrams/light-gh100-sm.png)

Streaming Multiprocessors (SMs) of NVIDIA GPUs are roughly analogous to the
cores of CPUs. That is, SMs both execute computations and store state available
for computation in registers, with associated caches. Compared to CPU cores, GPU
SMs are simple, weak processors. Execution in SMs is pipelined within an
instruction (as in almost all CPUs since the 1990s) but there is no speculative
execution or instruction pointer prediction (unlike all contemporary
high-performance CPUs).

However, GPU SMs can execute more
[threads](#thread) in parallel.

For comparison: an
[AMD EPYC 9965](https://www.techpowerup.com/cpu-specs/epyc-9965.c3904) CPU draws
at most 500 W and has 192 cores, each of which can execute instructions for at
most two threads at a time, for a total of 384 threads in parallel, running at
about 1.25 W per thread.

An H100 SXM GPU draws at most 700 W and has 132 SMs, each of which has four
[Warp Schedulers](#warp-scheduler) that can each
issue instructions to 32 threads (aka a
[warp](#warp)) in parallel per clock cycle, for a
total of 128 × 132 > 16,000 parallel threads running at about 5 cW apiece. Note
that this is truly parallel: each of the 16,000 threads can make progress with
each clock cycle.

GPU SMs also support a large number of _concurrent_ threads -- threads of
execution whose instructions are interleaved.

A single SM on an H100 can concurrently execute up to 2048 threads split across
64 thread groups of 32 threads each. With 132 SMs, that's a total of over
250,000 concurrent threads.

CPUs can also run many threads concurrently. But switches between
[warps](#warp) happen at the speed of a single
clock cycle (over 1000x faster than context switches on a CPU), again powered by
the SM's [Warp Schedulers](#warp-scheduler). The
volume of available [warps](#warp) and the speed of
[warp switches](#warp-scheduler) help
[hide latency](#latency-hiding) caused by memory reads, thread
synchronization, or other expensive instructions, ensuring that the
[arithmetic bandwidth](#arithmetic-bandwidth) provided by the
[CUDA Cores](#cuda-core) and
[Tensor Cores](#tensor-core) is well utilized.

This [latency-hiding](#latency-hiding) is the secret to GPUs'
strengths. CPUs seek to hide latency from end-users and programmers by
maintaining large, hardware-managed caches and sophisticated instruction
prediction. This extra hardware limits the fraction of their silicon area,
power, and heat budgets that CPUs can allocate to computation.

![GPUs dedicate more of their area to compute (green), and less to control and caching (orange and blue), than do CPUs. Modified from a diagram in [Fabien Sanglard's blog](https://fabiensanglard.net/cuda), itself likely modified from a diagram in [the CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).](dist/diagrams/light-cpu-vs-gpu.png)

For programs or functions like neural network inference or sequential database
scans for which it is relatively straightforward for programmers to
[express](#cuda-programming-model) the behavior of
[caches](#l1-data-cache) — e.g. store a chunk of
each input matrix and keep it in cache for long enough to compute the related
outputs — the result is much higher throughput.

### Core

The cores are the primary compute units that make up the
[Streaming Multiprocessors (SMs)](#streaming-multiprocessor).

![The internal architecture of an H100 GPU's Streaming Multiprocessors. CUDA and Tensor Cores are shown in green. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](dist/diagrams/light-gh100-sm.png)

Examples of GPU core types include
[CUDA Cores](#cuda-core) and
[Tensor Cores](#tensor-core).

Though GPU cores are comparable to CPU cores in that they are the component that
effects actual computations, this analogy can be quite misleading. Instead, it
is perhaps more helpful to take the viewpoint of the
[quantitative computer architect](https://archive.org/details/computerarchitectureaquantitativeapproach6thedition)
and think of them as "pipes" into which data goes in and out of which
transformed data is returned. These pipes are associated in turn with specific
[instructions](#streaming-assembler) from the
hardware's perspective and with different fundamental affordances of throughput
from the programmers' (e.g. floating point matrix multiplication arithmetic
throughput in the case of the
[Tensor Cores](#tensor-core)).

The [SMs](#streaming-multiprocessor) are closer to
being the equivalent of CPU cores, in that they have
[register memory](#register-file) to store
information, cores to transform it, and an
[instruction scheduler](#warp-scheduler) to specify
and command transformations.

### Special Function Unit

The Special Function Units (SFUs) in
[Streaming Multiprocessors (SMs)](#streaming-multiprocessor)
accelerate certain arithmetic operations.

![The internal architecture of an H100 SM. Special Function Units are shown in maroon, along with the [Load/Store Units](#load-store-unit). Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](dist/diagrams/light-gh100-sm.png)

Notable for neural network workloads are transcendental mathematical operations,
like `exp`, `sin`, and `cos`.

The
[Streaming Assembler (SASS)](#streaming-assembler)
instructions associated with the SFUs generally begin with `MUFU`: `MUFU.SQRT`,
`MUFU.EX2`. See [this Godbolt link](https://godbolt.org/z/WGh3rPe83) for sample
assembly using the `MUFU.EX2` instruction to implement the `expf` intrinsic in
[CUDA C++](#cuda-c).

### Load/Store Unit

The Load/Store Units (LSUs) dispatch requests to load or store data to the
memory subsystems of the GPU.

![The internal architecture of an H100 SM. Load/Store Units are shown in pink, along with the [Special Function Units](#special-function-unit). Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](dist/diagrams/light-gh100-sm.png)

Most importantly for
[CUDA programmers](#cuda-software-platform), they
interact directly with the
[Streaming Multiprocessor](#streaming-multiprocessor)'s
on-chip SRAM [L1 data cache](#l1-data-cache) and
indirectly with the off-chip, on-device
[global RAM](#gpu-ram) that respectively implement
the lowest and highest levels of the
[memory hierarchy](#memory-hierarchy) in the
[CUDA programming model](#cuda-programming-model).

### Warp Scheduler

The Warp Scheduler of the
[Streaming Multiprocessor (SM)](#streaming-multiprocessor)
decides which group of [threads](#thread) to
execute on each clock cycle.

![The internal architecture of an H100 SM. The Warp Scheduler and Dispatch Unit are shown in orange. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](dist/diagrams/light-gh100-sm.png)

These groups of [threads](#thread), known as
[warps](#warp), are switched out on a per clock
cycle basis — roughly one nanosecond - much like the fine-grained thread-level
parallelism of simultaneous multi-threading ("hyper-threading") in CPUs, but at
a much larger scale. The ability of the Warp Schedulers to switch rapidly
between a large number of concurrent tasks as soon as their instructions'
operands are available is key to the
[latency hiding](#latency-hiding) capabilities of GPUs.

Full CPU thread context switches take a few hundred to a few thousand clock
cycles (more like a microsecond than a nanosecond) due to the need to save the
context of one thread and restore the context of another. Additionally, context
switches on CPUs lead to reduced locality, further reducing performance by
increasing cache miss rates (see
[Mogul and Borg, 1991](https://www.researchgate.net/publication/220938995_The_Effect_of_Context_Switches_on_Cache_Performance)).

Because each [thread](#thread) has its own private
[registers](#registers) allocated from the
[register file](#register-file) of the
[SM](#streaming-multiprocessor), context switches
on the GPU do not require any data movement to save or restore contexts.

And because the [L1 caches](#l1-data-cache) on GPUs
can be entirely programmer-managed and are
[shared](#shared-memory) between the
[warps](#warp) scheduled together onto an
[SM](#streaming-multiprocessor) (see
[cooperative thread array](#cooperative-thread-array)),
context switches on the GPU have much less impact on cache hit rates. For
details on the interaction between programmer-managed caches and
hardware-managed caches in GPUs, see
[the "Maximize Memory Throughput" section of the CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#maximize-memory-throughput).

The Warp Schedulers also manage the
[execution state of warps](#warp-execution-state).

### CUDA Core

The CUDA Cores are GPU [cores](#core) that execute
scalar arithmetic instructions.

![The internal architecture of an H100 SM. The CUDA Cores and Tensor Cores are depicted in green. Note the larger size and lower number of Tensor Cores. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](dist/diagrams/light-gh100-sm.png)

They are to be contrasted with the
[Tensor Cores](#tensor-core), which execute matrix
operations.

Unlike CPU cores, instructions issued to CUDA Cores are not generally
independently scheduled. Instead, groups of cores are issued the same
instruction simultaneously by the
[Warp Scheduler](#warp-scheduler) but apply that
instruction to different [registers](#registers).
Commonly, these groups are of size 32, the size of a
[warp](#warp), but for contemporary GPUs groups can
contain as little as one thread, at a cost to performance.

The term "CUDA Core" is slightly slippery: in different
[Streaming Multiprocessor architectures](#streaming-multiprocessor-architecture)
CUDA Cores can consist of different units -- a different mixture of 32 bit
integer and 32 bit and 64 bit floating point units. They are perhaps best
thought of in contrast to early GPUs, which contained a variety of much more
specialized compute units mapped onto shader pipelines (see
[CUDA Device Architecture](#cuda-device-architecture)).

So, for example, the
[H100 whitepaper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) indicates that
an H100 GPU's
[Streaming Multiprocessors (SMs)](#streaming-multiprocessor)
each have 128 "FP32 CUDA Cores", which accurately counts the number of 32 bit
floating point units but is double the number of 32 bit integer or 64 bit
floating point units (as evidenced by the diagram above). For estimating
performance, it's best to look directly at the number of hardware units for a
given operation.

### Tensor Core

Tensor Cores are GPU [cores](#core) that operate on
entire matrices with each instruction.

![The internal architecture of an H100 SM. Note the larger size and lower number of Tensor Cores. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](dist/diagrams/light-gh100-sm.png)

Operating on more data for a single instruction fetch dramatically reduces power
requirements, which unlocks increased performance (see
[this talk](https://youtu.be/kLiwvnr4L80?t=868) by Bill Dally, Chief Scientist
at NVIDIA). Since their introduction in the Volta
[Streaming Multiprocessor (SM) Architecture](#streaming-multiprocessor-architecture)
generation, they have been the only way to achieve the highest
[arithmetic throughput](#arithmetic-bandwidth) on NVIDIA GPUs
-- providing 100x more floating point operations per second than
[CUDA Cores](#cuda-core).

As an example, the `HMMA16.16816.F32`
[SASS](#streaming-assembler) instruction calculates
D = AB + C for matrices A, B, C, and D (where C is often the same physical
matrix as D). The `MMA` stands for "Matrix Multiply and Accumulate". `HMMA16`
indicates that the inputs are half-precision (`16` bits) and the `F32` indicates
that the outputs are accumulated into `32` bit (aka single-precision) floats.

`16816` is not a single number larger than 16,000. Instead, the string of
numbers `16`, `8`, `16` denote the dimensions of the matrices. These dimensions
are generally named `m`, `k`, and `n` by NVIDIA, for example in
[PTX](#parallel-thread-execution) instructions. The
outer dimensions of A and B, aka `m` and `n`, come first and last, respectively,
and the shared inner dimension for the accumulation, `k`, is in the middle.
Multiplying these out, we see that the `HMMA16.16816.32` instruction performs 16
× 8 × 16 = 2,048 multiply-accumulate (MAC) operations.

Note that a single instruction in a single
[thread](#thread) does not produce the entire
matrix multiplication. Instead, the 32 threads of a
[warp](#warp) cooperatively produce the result by
executing the instruction together. Most of the per-instruction power overhead
is in decoding, which is shared across a
[warp](#warp) thanks to the
[warp scheduler](#warp-scheduler). But even spread
across those 32 threads, that's 64 = 2,048 ÷ 32 MACs per instruction.

For this reason, it is helpful to think of Tensor Cores, and similar hardware
like the systolic arrays in Google Tensor Processing Units (TPUs), as a form of
[complex instruction set computer (CISC)](https://www.omgwiki.org/ddsf/doku.php?id=ddsf:public:guidebook:06_append:glossary:c:cisc)
hardware. For more on this perspective, applied to TPUs, see
[this talk by computer architect David Patterson](https://youtu.be/fhHAArxwzvQ?t=2072),
who also
[coined the terms CISC and RISC](https://www.semanticscholar.org/paper/4d3a941a5749dbf0dd39554f12597c449c3c07ff).

That assembler-level instruction might be produced by a compiler to implement
[PTX-level](#parallel-thread-execution)
matrix-multiply-and-accumulate instructions like `wmma` (documented
[here](https://docs.nvidia.com/cuda/archive/12.8.0/parallel-thread-execution/index.html#warp-level-matrix-instructions)).
Those instructions also calculate D = AB + C for matrices A, B, C, and D, but
are generally compiled into many individual
[SASS](#streaming-assembler) Tensor Core
instructions that operate on smaller matrices.

These instructions from the
[PTX](#parallel-thread-execution) instruction set
architecture are exposed in the high-level
[CUDA C++ programming language](#cuda-c) as
intrinsics.

In reverse order, a line of [CUDA C++](#cuda-c)
coding a matrix multiplication `C = A @ B`, of two 16 by 16 matrices, like

```cpp
wmma::mma_sync(c, a, b, c);
```

where `c` is initialized to all zeros, and the first appearance indicates it is
also the output, might be compiled by [`nvcc`](#nvcc)
to the [PTX](#parallel-thread-execution)
intermediate representation as

```ptx
wmma.mma.sync.aligned.col.row.m16n16k16.f32.f32 {%f2, %f3, %f4, %f5, %f6, %f7, %f8, %f9}, {%r2, %r3, %r4, %r5, %r6, %r7, %r8, %r9}, {%r10, %r11, %r12, %r13, %r14, %r15, %r16, %r17}, {%f1, %f1, %f1, %f1, %f1, %f1, %f1, %f1};
```

and then finally compiled by `ptxas` to
[SASS](#streaming-assembler) as

```sass
HMMA.1688.F32 R20, R12, R11, RZ   // 1
HMMA.1688.F32 R24, R12, R17, RZ   // 2
HMMA.1688.F32 R20, R14, R16, R20  // 3
HMMA.1688.F32 R24, R14, R18, R24  // 4
```

The operands to each `HMMA` instruction can be read, in order, as
`D = A @ B + C`. For example, instruction 3 uses
[register](#register-file) 20 for its output `D`,
registers 14 and 16 for its inputs `A` and `B`, respectively, and re-uses
register 20 for its input `C`, effecting the computation `C += A @ B`.

This program partitions the full 16 by 16 square matrix multiplication into four
separate instructions, each itself a matrix multiplication of a 16 by 8 matrix
with an 8 by 8 matrix. Similarly, programs running large-scale matrix
multiplications must break their work down into smaller matrix multiplications,
like the 16 by 16 square matrix multiplication performed by the `mma_sync` call
we are dissecting. We walk through this program below.

![Register usage in a Tensor Core MMA for C = A @ B. The R11, R17, R16, and R18 registers are used in instructions 1, 2, 3, and 4, respectively. See surrounding text for details.](dist/diagrams/light-tensor-core-mma.png)

The first two instructions compute the matrix multiplication of the first eight
columns of the input `a`, from `R12`, with the first eight rows of the input
`b`, from `R11` and `R17`, producing a 16 by 16 matrix, which is stored in `R20`
and `R24`. This is a sort of "outer product": a tall and skinny matrix
multiplied by a short and wide matrix. (`RZ` is a special-purpose "register"
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
done in pure [CUDA C++](#cuda-c), requiring instead
[PTX](#parallel-thread-execution) intrinsics for
both computation and memory. It is generally recommended to instead use existing
kernels from kernel libraries like
[cuBLAS (CUDA Basic Linear Algebra Subroutines)](#cublas)
or higher-level kernel programming interfaces like
[CUTLASS (CUDA Templates for Linear Algebra Subroutines)](https://github.com/NVIDIA/cutlass).
For an introduction to CUTLASS, see
[this blog post series by Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/).

Tensor Cores are much larger and less numerous than
[CUDA Cores](#cuda-core). An H100 SXM5 has only
four Tensor Cores per
[SM](#streaming-multiprocessor), i.e. one per
[Warp Scheduler](#warp-scheduler), but has hundreds
of [CUDA Cores](#cuda-core). Tensor Cores are the
primary producers and consumers of
[Tensor Memory](#tensor-memory).

Tensor Cores were introduced in the V100 GPU, which represented a major
improvement in the suitability of NVIDIA GPUs for large neural network
workloads. For more, see
[the NVIDIA white paper introducing the V100](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf).

The internals of Tensor Cores are unknown, and likely differ from
[SM Architecture](#streaming-multiprocessor-architecture)
to
[SM Architecture](#streaming-multiprocessor-architecture).
They are commonly assumed to be systolic arrays, like TPUs, but there is no
consensus in the microbenchmarking literature.

### Tensor Memory Accelerator

Tensor Memory Accelerators are specialized hardware in Hopper and Blackwell
[architecture](#streaming-multiprocessor-architecture)
GPUs designed to accelerate access to multi-dimensional arrays in
[GPU RAM](#gpu-ram).

![The internal architecture of an H100 [Streaming Multiprocessor (SM)](#streaming-multiprocessor). Note the Tensor Memory Accelerator at the bottom of the [SM](#streaming-multiprocessor), shared between the four sub-units. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](dist/diagrams/light-gh100-sm.png)

The TMA loads data from
[global memory](#global-memory)/[GPU RAM](#gpu-ram)
to
[shared memory](#shared-memory)/[L1 data cache](#l1-data-cache),
bypassing the
[registers](#registers)/[register file](#register-file)
entirely.

The first advantage of the TMA comes from reducing the use of other compute and
memory resources. The TMA hardware calculates addresses for bulk affine memory
accesses, i.e. accesses of the form `addr = width * base + offset` for many
bases and offsets concurrently, which are the most common accesses for arrays.
Offloading this work to the TMA saves space in the
[register file](#register-file), reducing
"[register pressure](#register-pressure)", and reduces demand
on the [arithmetic bandwidth](#arithmetic-bandwidth) provided
by the [CUDA Cores](#cuda-core). The savings are
more pronounced for large (KB-scale) accesses to arrays with two or more
dimensions.

The second advantage comes from the asynchronous execution model of TMA copies.
A single [CUDA thread](#thread) can trigger a large
copy and then rejoin its [warp](#warp) to perform
other work. Those [threads](#thread) and others in
the same [thread block](#thread-block) can then
asynchronously detect the completion of the TMA copy after it finishes and
operate on the results (as in a producer-consumer model).

For details, see the TMA sections of
[Luo et al.'s Hopper micro-benchmarking paper](https://arxiv.org/abs/2501.12084v1).
and the
[NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator).

Note that, despite the name, the Tensor Memory Accelerator does not accelerate
operations using [Tensor Memory](#tensor-memory).

### Streaming Multiprocessor Architecture

[Streaming Multiprocessors (SMs)](#streaming-multiprocessor)
are versioned with a particular "architecture" that defines their compatibility
with
[Streaming Assembler (SASS)](#streaming-assembler)
code.

![A streaming multiprocessor with the "Hopper" SM90 architecture. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](dist/diagrams/light-gh100-sm.png)

![A streaming multiprocessor with the original "Tesla" SM architecture. Modified from [Fabien Sanglard's blog](https://fabiensanglard.net/cuda)](dist/diagrams/light-tesla-sm.png)

Most [SM](#streaming-multiprocessor) versions have
two components: a major version and a minor version.

The major version is _almost_ synonymous with GPU architecture family. For
example, all SM versions `6.x` are of the Pascal Architecture. Some NVIDIA
documentation even
[makes this claim directly](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html).
But, as an example, Ada GPUs have
[SM](#streaming-multiprocessor) architecture
version `8.9`, the same major version as Ampere GPUs.

Target [SM](#streaming-multiprocessor) versions for
[SASS](#streaming-assembler) compilation can be
specified when invoking `nvcc`, the
[NVIDIA CUDA Compiler Driver](#nvcc). Compatibility
across major versions is explicitly not guaranteed. For more on compatibility
across minor versions, see the
[documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)
for [nvcc](#nvcc).

### Texture Processing Cluster

A Texture Processing Cluster (TPC) is a pair of adjacent
[Streaming Multiprocessors (SMs)](#streaming-multiprocessor).

Before the Blackwell
[SM architecture](#streaming-multiprocessor-architecture),
TPCs were not mapped onto any level of the
[CUDA programming model](#cuda-programming-model)'s
[memory hierarchy](#memory-hierarchy) or
[thread hierarchy](#thread-hierarchy).

The fifth-generation [Tensor Cores](#tensor-core)
in the Blackwell
[SM architecture](#streaming-multiprocessor-architecture)
added the "CTA pair" level of the
[Parallel Thread eXecution (PTX)](#parallel-thread-execution)
[thread hierarchy](#thread-hierarchy), which maps
onto TPCs. Many `tcgen05`
[PTX](#parallel-thread-execution) instructions
include a `.cta_group` field that can use a single
[SM](#streaming-multiprocessor) (`.cta_group::1`)
or a pair of [SMs](#streaming-multiprocessor) in a
TPC (`::2`), which are mapped to `1SM` and `2SM` variants of
[Streaming Assembler (SASS)](#streaming-assembler)
instructions like `MMA`.

### Graphics/GPU Processing Cluster

A GPC is a collection of
[Texture Processing Clusters (TPCs)](#texture-processing-cluster)
(themselves groups of
[Streaming Multiprocessors](#streaming-multiprocessor)
or SMs) plus a raster engine. Apparently, some people use NVIDIA GPUs for
graphics, for which the raster engine is important. Relatedly, the name used to
stand for Graphics Processing Cluster, but is now, e.g. in the
[NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html),
expanded as "GPU Processing Cluster".

Since the introduction of
[compute capability](#compute-capability) 9.0 GPUs
like H100s, there is an additional layer of the
[CUDA programming model](#cuda-programming-model)'s
[thread hierarchy](#thread-hierarchy), a "cluster"
of [thread blocks](#thread-block) that are
scheduled onto the same GPC, just as the threads of a
[thread block](#thread-block) are scheduled onto
the same [SM](#streaming-multiprocessor), and have
their own level of the
[memory hierarchy](#memory-hierarchy), distributed
shared memory. Elsewhere, we elide discussion of this feature.

### Register File

The register file of the
[Streaming Multiprocessor](#streaming-multiprocessor)
is the primary store of bits in between their manipulation by the
[cores](#core).

![The internal architecture of an H100 SM. The register file is depicted in blue. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](dist/diagrams/light-gh100-sm.png)

Like registers in CPUs, these registers are made from very fast memory
technology that can keep pace with the compute
[cores](#core), about an order of magnitude faster
than the [L1 data cache](#l1-data-cache).

The register file is split into 32 bit registers that can be dynamically
reallocated between different data types, like 32 bit integers, 64 bit floating
point numbers, and (groups of) 16 bit or smaller floating point numbers. These
physical registers back the
[virtual registers](#registers) in the
[Parallel Thread eXecution (PTX)](#parallel-thread-execution)
intermediate representation.

Allocation of physical registers to
[threads](#thread) in
[Streaming Assembler (SASS)](#streaming-assembler)
is managed by a compiler like `ptxas`, which optimizes register file usage by
[thread blocks](#thread-block). If each
[thread block](#thread-block) consumes too much of
the register file (colloquially, high
"[register pressure](#register-pressure)"), then the number of
concurrently schedulable [threads](#thread) will be
reduced, leading to a low [occupancy](#occupancy) and possibly
impacting performance by reducing opportunities for
[latency hiding](#latency-hiding).

### L1 Data Cache

The L1 data cache is the private memory of the
[Streaming Multiprocessor](#streaming-multiprocessor)
(SM).

![The internal architecture of an H100 SM. The L1 data cache is depicted in light blue. Modified from NVIDIA's [H100 white paper](https://resources.nvidia.com/en-us-tensor-core).](dist/diagrams/light-gh100-sm.png)

Each SM partitions that memory among
[groups of threads](#thread-block) scheduled onto
it.

The L1 data cache is co-located with and only about an order of magnitude slower
than the components that effect computations (e.g. the
[CUDA Cores](#cuda-core)).

It is implemented with SRAM, the same basic semiconductor cell used in CPU
caches and registers and in the
[memory subsystem of Groq LPUs](https://groq.com/wp-content/uploads/2023/05/GroqISCAPaper2022_ASoftwareDefinedTensorStreamingMultiprocessorForLargeScaleMachineLearning-1.pdf).
The L1 data cache is accessed by the
[Load/Store Units](#load-store-unit) of the
[SM](#streaming-multiprocessor).

CPUs also maintain an L1 cache. In CPUs, that cache is fully hardware-managed.
In GPUs that cache is mostly programmer-managed, even in high-level languages
like [CUDA C](#cuda-c).

Each L1 data cache in each of an H100's SMs can store 256 KiB (2,097,152 bits).
Across the 132 SMs in an H100 SXM 5, that's 33 MiB (242,221,056 bits) of cache
space.

### Tensor Memory

Tensor Memory is a specialized memory in the
[Streaming Multiprocessor (SM)](#streaming-multiprocessor)
of certain GPUs, like the [B200](https://modal.com/blog/introducing-b200-h200),
for storing the inputs and outputs of
[Tensor Cores](#tensor-core).

Tensor Memory access is highly restricted. Data must be moved collectively by
four [warps](#warp) in a warpgroup, and they can
move memory only in specific patterns between Tensor Memory and
[registers](#registers), write
[shared memory](#shared-memory) to Tensor Memory,
or issue matrix-multiply-accumulate (MMA) instructions to
[Tensor Cores](#tensor-core) that use Tensor Memory
for specific operands. So much for a
["compute-unified" device architecture](#cuda-device-architecture)!

Specifically, for a `tcgen05.mma`
[Parallel Thread eXecution](#parallel-thread-execution)
instruction computing `D += A @ B` to use Tensor Memory, the "accumulator"
matrix `D` _must_ be in Tensor Memory, the left-hand matrix `A` _may_ be in
Tensor Memory or [shared memory](#shared-memory),
and the right-hand matrix B _must_ be in
[shared memory](#shared-memory), not Tensor Memory.
This is complex, but not arbitrary -- accumulators are accessed more frequently
during matmuls than are the tiles, so they benefit more from specialized
hardware, e.g. from shorter, simpler wiring between the
[Tensor Cores](#tensor-core) and the Tensor Memory.
Note that none of the matrices are in the
[registers](#registers).

Beware: Tensor Memory is not directly related to the
[Tensor Memory Accelerator](#tensor-memory-accelerator),
which instead loads into the
[L1 data cache](#l1-data-cache). Roughly speaking,
data is moved from that cache into Tensor Memory only as a result of a
[Tensor Core](#tensor-core) operation and then is
explicitly moved out for post-processing, e.g. the non-linearity after a matrix
multiplication in a neural network.

For details on tensor memory and patterns for its use in matrix multiplications,
see the
[_Programming Blackwell Tensor Cores with CUTLASS_ talk from GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72720/).

### GPU RAM

![In high-performance data center GPUs like the H100, RAM is located on a die directly next to the processor's. Adapted from the Wikipedia page for [high-bandwidth memory](https://en.wikipedia.org/wiki/High_Bandwidth_Memory).](dist/diagrams/light-hbm-schematic.png)

The bottom-level memory of the GPU is a large (many megabytes to gigabytes)
memory store that is addressable by all of the GPU's
[Streaming Multiprocessors (SMs)](#streaming-multiprocessor).

It is also known as GPU RAM (random access memory) or video RAM (VRAM). It uses
Dynamic RAM (DRAM) cells, which are slower but smaller than the Static RAM
(SRAM) used in [registers](#register-file) and
[cache memory](#l1-data-cache). For details on DRAM
and SRAM, we recommend Ulrich Drepper's 2007 article
["What Every Programmer Should Know About Memory"](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf).

It is generally not on the same die as the
[SMs](#streaming-multiprocessor), though in the
latest data center-grade GPUs like the H100, it is located on a shared
[interposer](https://en.wikipedia.org/wiki/Interposer) for decreased latency and
increased [bandwidth](#memory-bandwidth). These GPUs use
[High-Bandwidth Memory (HBM)](https://en.wikipedia.org/wiki/High_Bandwidth_Memory)
technology, rather than the more familiar Double Data Rate (DDR) memory in
consumer GPUs and CPUs.

RAM is used to implement the
[global memory](#global-memory) of the
[CUDA programming model](#cuda-programming-model)
and to store [register](#registers) data that
spills from the [register file](#register-file).

An H100 can store 80 GiB (687,194,767,360 bits) in its RAM.

## Device Software

These terms and technologies are used for software that runs on GPU — the
"device" in NVIDIA's lingo.

### CUDA (Programming Model)

CUDA stands for _Compute Unified Device Architecture_. Depending on the context,
"CUDA" can refer to multiple distinct things: a
[high-level device architecture](#cuda-device-architecture),
a parallel programming model for architectures with that design, or a
[software platform](#cuda-software-platform) that
extends high-level languages like C to add that programming model.

The vision for CUDA is laid out in the
[Lindholm et al., 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf)
white paper. We highly recommend this paper, which is the original source for
many claims, diagrams, and even specific turns of phrase in NVIDIA's
documentation.

Here, we focus on the CUDA _programming model_.

The Compute Unified Device Architecture (CUDA) programming model is a
programming model for programming massively parallel processors.

Per the
[NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#a-scalable-programming-model),
there are three key abstractions in the CUDA programming model:

- [**Hierarchy of thread groups**](#thread-hierarchy).
  Programs are executed in threads but can make reference to groups of threads
  in a nested hierarchy, from
  [blocks](#thread-block) to
  [grids](#thread-block-grid).
- [**Hierarchy of memories**](#memory-hierarchy).
  Thread groups at each level of the hierarchy have access to a memory resource
  for communication within the group. Accessing the
  [lowest layer](#shared-memory) of the memory
  hierarchy should be
  [nearly as fast as executing an instruction](#l1-data-cache).
- **Barrier synchronization.** Thread groups can coordinate execution by means
  of barriers.

The hierarchies of execution and memory and their mapping onto
[device hardware](#device-hardware) are summarized in the following
diagram.

![Left: the abstract thread group and memory hierarchies of the CUDA programming model. Right: the matching hardware implementing those abstractions. Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](dist/diagrams/light-cuda-programming-model.png)

Together, these three abstractions encourage the expression of programs in a way
that scales transparently as GPU devices scale in their parallel execution
resources.

Put provocatively: this programming model prevents programmers from writing
programs for NVIDIA's
[CUDA-architected](#cuda-device-architecture) GPUs
that fail to get faster when the program's user buys a new NVIDIA GPU.

For example, each [thread block](#thread-block) in
a CUDA program can coordinate tightly, but coordination between blocks is
limited. This ensures blocks capture parallelizable components of the program
and can be scheduled in any order — in the terminology of computer architecture,
the programmer "exposes" this parallelism to the compiler and hardware. When the
program is executed on a new GPU that has more scheduling units (specifically,
more
[Streaming Multiprocessors](#streaming-multiprocessor)),
more of these blocks can be executed in parallel.

![A CUDA program with eight [blocks](#thread-block) runs in four sequential steps (waves) on a GPU with two [SMs](#streaming-multiprocessor) but in half as many steps on one with twice as many [SMs](#streaming-multiprocessor). Modified from the [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).](dist/diagrams/light-wave-scheduling.png)

The CUDA programming model abstractions are made available to programmers as
extensions to high-level CPU programming languages, like the
[CUDA C++ extension of C++](#cuda-c). The programming
model is implemented in software by an instruction set architecture
[(Parallel Thread eXecution, or PTX)](#parallel-thread-execution)
and low-level assembly language
[(Streaming Assembler, or SASS)](#streaming-assembler).
For example, the [thread block](#thread-block)
level of the [thread hierarchy](#thread-hierarchy)
is implemented via
[cooperative thread arrays](#cooperative-thread-array)
in these languages.

### Streaming ASSembler

[Streaming ASSembler](https://stackoverflow.com/questions/9798258/what-is-sass-short-for)
(SASS) is the assembly format for programs running on NVIDIA GPUs. This is the
lowest-level format in which human-readable code can be written. It is one of
the formats output by `nvcc`, the
[NVIDIA CUDA Compiler Driver](#nvcc), alongside
[PTX](#parallel-thread-execution). It is converted
to device-specific binary microcodes during execution. Presumably, the
"Streaming" in "Streaming Assembler" refers to the
[Streaming Multiprocessors](#streaming-multiprocessor)
which the assembly language programs.

SASS is versioned and tied to a specific NVIDIA GPU
[SM architecture](#streaming-multiprocessor-architecture).
See also [Compute Capability](#compute-capability).

Some exemplary instructions in SASS for the SM90a architecture of Hopper GPUs:

- `FFMA R0, R7, R0, 1.5 ;` - perform a `F`used `F`loating point `M`ultiply `A`dd
  that multiplies the contents of `R`egister 7 and `R`egister 0, adds `1.5`, and
  stores the result in `R`egister 0.
- `S2UR UR4, SR_CTAID.X ;` - copy the `X` value of the
  [Cooperative Thread Array](#cooperative-thread-array)'s
  `I`n`D`ex from its `S`pecial `R`egister to `U`niform `R`egister 4.

Even more so than for CPUs, writing this "GPU assembler" by hand is very
uncommon. Viewing compiler-generated SASS while profiling and editing high-level
[CUDA C/C++](#cuda-c) code or in-line
[PTX](#parallel-thread-execution) is
[more common](https://docs.nvidia.com/gameworks/content/developertools/desktop/ptx_sass_assembly_debugging.htm),
especially in the production of the highest-performance kernels. Viewing
[CUDA C/C++](#cuda-c), SASS, and
[PTX](#parallel-thread-execution) together is
supported on [Godbolt](https://godbolt.org/z/5r9ej3zjW). For more detail on SASS
with a focus on performance debugging workflows, see
[this talk](https://www.youtube.com/watch?v=we3i5VuoPWk) from Arun Demeure.

SASS is _very_ lightly documented — the instructions are listed in the
[documentation for NVIDIA's CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#instruction-set-ref),
but their semantics are not defined. The mapping from ASCII assembler to binary
opcodes and operands is entirely undocumented, but it has been
reverse-engineered in certain cases
([Maxwell](https://github.com/NervanaSystems/maxas),
[Lovelace](https://kuterdinel.com/nv_isa_sm89/)).

### Parallel Thread eXecution

Parallel Thread eXecution (PTX) is an intermediate representation (IR) for code
that will run on a parallel processor (almost always an NVIDIA GPU). It is one
of the formats output by `nvcc`, the
[NVIDIA CUDA Compiler Driver](#nvcc). It is
pronounced "pee-tecks" by many NVIDIA engineers and "pee-tee-ecks" by everyone
else.

NVIDIA documentation refers to PTX as both a "virtual machine" and an
"instruction set architecture".

From the programmer's perspective, PTX is an instruction set for programming
against a virtual machine model. Programmers or compilers producing PTX can be
confident their program will run with the same semantics on many distinct
physical machines, including machines that do not yet exist. In this way, it is
also similar to CPU instruction set architectures like
[x86_64](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html),
[aarch64](https://developer.arm.com/documentation/ddi0487/latest/), or
[SPARC](https://www.gaisler.com/doc/sparcv8.pdf).

Unlike those ISAs, PTX is very much an
[intermediate representation](https://en.wikipedia.org/wiki/Intermediate_representation),
like LLVM-IR. The PTX components of a
[CUDA binary](#cuda-binary-utilities) will be
just-in-time (JIT) compiled by the host
[CUDA Drivers](#nvidia-gpu-drivers) into
device-specific [SASS](#streaming-assembler) for
execution.

In the case of NVIDIA GPUs, PTX is forward-compatible: GPUs with a matching or
higher [compute capability](#compute-capability)
version will be able to run the program, thanks to this mechanism of JIT
compilation. In this way, PTX is a
["narrow waist"](https://www.oilshell.org/blog/2022/02/diagrams.html) that
separates the worlds of hardware and software.

Some exemplary PTX:

```ptx
.reg .f32 %f<7>;
```

- a compiler directive for the
  PTX-to-[SASS](#streaming-assembler) compiler
  indicating that this kernel consumes seven 32-bit floating point
  [registers](#registers). Registers are
  dynamically allocated to groups of
  [threads](#thread)
  ([warps](#warp)) from the
  [SM](#streaming-multiprocessor)'s
  [register file](#register-file).

```ptx
fma.rn.f32 %f5, %f4, %f3, 0f3FC00000;
```

- apply a fused multiply-add (`fma`) operation to multiply the contents of
  registers `f3` and `f4` and add the constant `0f3FC00000`, storing the result
  in `f5`. All numbers are in 32 bit floating point representation. The `rn`
  suffix for the FMA operation sets the floating point rounding mode to
  [IEEE 754 "round even"](https://en.wikipedia.org/wiki/IEEE_754) (the default).

```ptx
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %tid.x;
```

- `mov`e the `x`-axis values of the `c`ooperative `t`hread `a`rray `i`n`d`ex,
  the cooperative thread array dimension index (`ntid`), and the `t`hread
  `i`n`d`ex into three `u32` registers `r1` - `r3`.

The PTX programming model exposes multiple levels of parallelism to the
programmer. These levels map directly onto the hardware through the PTX machine
model, diagrammed below.

![The PTX machine model. Modified from the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-machine-model).](dist/diagrams/light-ptx-machine-model.png)

Notably, in this machine model there is a single instruction unit for multiple
processors. While each processor runs one
[thread](#thread), those threads must execute the
same instructions — hence _parallel_ thread execution, or PTX. They coordinate
with each other through
[shared memory](#shared-memory) and effect
different results by means of private
[registers](#registers).

The documentation for the latest version of PTX is available from NVIDIA
[here](https://docs.nvidia.com/cuda/parallel-thread-execution/). The instruction
sets of PTX are versioned with a number called the
"[compute capability](#compute-capability)", which
is synonymous with "minimum supported
[Streaming Multiprocessor architecture](#streaming-multiprocessor-architecture)
version".

Writing in-line PTX by hand is uncommon outside of the cutting edge of
performance, similar to writing in-line `x86_64` assembly, as is done in
high-performance vectorized query operators in analytical databases and in
performance-sensitive sections of operating system kernels. At time of writing
in September of 2025, in-line PTX is the only way to take advantage of some
Hopper-specific hardware features like the `wgmma` and `tma` instructions, as in
[Flash Attention 3](https://arxiv.org/abs/2407.08608) or in the
[Machete w4a16 kernels](https://youtu.be/-4ZkpQ7agXM). Viewing
[CUDA C/C++](#cuda-c),
[SASS](#streaming-assembler), and
[PTX](#parallel-thread-execution) together is
supported on [Godbolt](https://godbolt.org/z/5r9ej3zjW). See the
[NVIDIA "Inline PTX Assembly in CUDA" guide](https://docs.nvidia.com/cuda/inline-ptx-assembly/)
for details.

### Compute Capability

Instructions in the
[Parallel Thread Execution](#parallel-thread-execution)
instruction set are compatible with only certain physical GPUs. The versioning
system used to abstract away details of physical GPUs from the instruction set
and [compiler](#nvcc) is called "Compute Capability".

Most compute capability version numbers have two components: a major version and
a minor version. NVIDIA promises forward compatibility (old
[PTX](#parallel-thread-execution) code runs on new
GPUs) across both major and minor versions following the
[onion layer](https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-module-directives-target)
model.

With Hopper, NVIDIA introduced an additional version suffix, the `a` in `9.0a`,
which includes features that deviate from the onion model: their future
compatibility is not guaranteed, even within major versions.

With Blackwell, NVIDIA introduced yet another version suffix, the `f` in
`10.0f`, which also deviates from the onion model, and is closer to
[SemVer](https://semver.org/): compatibility is guaranteed across minor versions
but not major versions.

Target compute capabilities for
[PTX](#parallel-thread-execution) compilation can
be specified when invoking `nvcc`, the
[NVIDIA CUDA Compiler Driver](#nvcc). By default, the
compiler will also generate optimized
[SASS](#streaming-assembler) for the matching
[Streaming Multiprocessor (SM) architecture](#streaming-multiprocessor-architecture).
The
[documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architectures)
for [`nvcc`](#nvcc) refers to compute capability as a
"virtual GPU architecture", in contrast to the "physical GPU architecture"
expressed by the [SM](#streaming-multiprocessor)
version.

The technical specifications for each compute capability version can be found in
the
[Compute Capability section of the NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

### Thread

![Threads are the lowest level of the thread group hierarchy (top, left) and are mapped onto the [cores](#core) of a [Streaming Multiprocessor](#streaming-multiprocessor). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](dist/diagrams/light-cuda-programming-model.png)

A _thread of execution_ (or "thread" for short) is the lowest unit of
programming for GPUs, the base and atom of the
[CUDA programming model](#cuda-programming-model)'s
[thread hierarchy](#thread-hierarchy). A thread has
its own [registers](#registers), but little else.

Both [SASS](#streaming-assembler) and
[PTX](#parallel-thread-execution) programs target
threads. Compare this to a typical C program in a POSIX environment, which
targets a process, itself a collection of one or more threads. Unlike POSIX
threads, [CUDA](#cuda-programming-model) threads
are not used to make syscalls.

Like a thread on a CPU, a GPU thread can have a private instruction
pointer/program counter. However, for performance reasons, GPU programs are
generally written so that all the threads in a
[warp](#warp) share the same instruction pointer,
executing instructions in lock-step (see also
[Warp Scheduler](#warp-scheduler)).

Also like threads on CPUs, GPU threads have stacks in
[global memory](#gpu-ram) for storing spilled
registers and a function call stack, but high-performance
[kernels](#kernel) generally limit use of both.

A single [CUDA Core](#cuda-core) executes
instructions from a single thread.

### Warp

A warp is a group of [threads](#thread) that are
scheduled together and execute in parallel. All
[threads](#thread) in a warp are scheduled onto a
single
[Streaming Multiprocessor (SM)](#streaming-multiprocessor).
A single [SM](#streaming-multiprocessor) typically
executes multiple warps, at the very least all warps from the same
[Cooperative Thread Array](#cooperative-thread-array),
aka [thread block](#thread-block).

Warps are the typical unit of execution on a GPU. In normal execution, all
[threads](#thread) of a warp execute the same
instruction in parallel — the so-called "Single-Instruction, Multiple Thread" or
SIMT model. When the [threads](#thread) in a warp
split from one another to execute different instructions, also known as
[warp divergence](#warp-divergence), performance generally
drops precipitously.

Warp size is technically a machine-dependent constant, but in practice (and
elsewhere in this glossary) it is 32.

When a warp is issued an instruction, the results are generally not available
within a single clock cycle, and so dependent instructions cannot be issued.
While this is most obviously true for fetches from
[global memory](#global-memory), which generally
[go off-chip](#gpu-ram), it is also true for some
arithmetic instructions (see
[the CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#arithmetic-instructions)
for a table of results per clock cycle for specific instructions).

A warp whose next instruction is delayed by missing operands is said to be
[stalled](#warp-execution-state).

Instead of waiting for an instructions results to return, when multiple warps
are scheduled onto a single
[SM](#streaming-multiprocessor), the
[Warp Scheduler](#warp-scheduler) will select
another warp to execute. This
[latency-hiding](#latency-hiding) is how GPUs achieve high
throughput and ensure work is always available for all of their cores during
execution. For this reason, it is often beneficial to maximize the number of
warps scheduled onto each
[SM](#streaming-multiprocessor), ensuring there is
always an [eligible](#warp-execution-state) warp for the
[SM](#streaming-multiprocessor) to run. The
fraction of cycles on which a warp was issued an instruction is known as the
[issue efficiency](#issue-efficiency). The degree of
concurrency in warp scheduling is known as
[occupancy](#occupancy).

Warps are not actually part of the
[CUDA programming model](#cuda-programming-model)'s
[thread hierarchy](#thread-hierarchy). Instead,
they are an implementation detail of the implementation of that model on NVIDIA
GPUs. In that way, they are somewhat akin to
[cache lines](https://www.nic.uoregon.edu/~khuck/ts/acumem-report/manual_html/ch03s02.html)
in CPUs: a feature of the hardware that you don't directly control and don't
need to consider for program correctness, but which is important for achieving
[maximum performance](#perf).

Warps are named in reference to weaving, "the first parallel thread technology",
according to
[Lindholm et al., 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf).
The equivalent of warps in other GPU programming models include
[subgroups](https://github.com/gpuweb/gpuweb/pull/4368) in WebGPU,
[waves](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_WaveSize.html)
in DirectX, and
[simdgroups](https://developer.apple.com/documentation/metal/compute_passes/creating_threads_and_threadgroups#2928931)
in Metal.

### Cooperative Thread Array

![Cooperative thread arrays correspond to the [thread block](#thread-block) level of the thread block hierarchy in the [CUDA programming model](#cuda-programming-model). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](dist/diagrams/light-cuda-programming-model.png)

A cooperative thread array (CTA) is a collection of threads scheduled onto the
same
[Streaming Multiprocessor (SM)](#streaming-multiprocessor).
CTAs are the
[PTX](#parallel-thread-execution)/[SASS](#streaming-assembler)
implementation of the
[CUDA programming model](#cuda-programming-model)'s
[thread blocks](#thread-block). CTAs are composed
of one or more [warps](#warp).

Programmers can direct [threads](#thread) within a
CTA to coordinate with each other. The programmer-managed
[shared memory](#shared-memory), in the
[L1 data cache](#l1-data-cache) of the
[SMs](#streaming-multiprocessor), makes this
coordination fast. Threads in different CTAs cannot coordinate with each other
via barriers, unlike threads within a CTA, and instead must coordinate via
[global memory](#global-memory), e.g. via atomic
update instructions. Due to driver control over the scheduling of CTAs at
runtime, CTA execution order is indeterminate and blocking a CTA on another CTA
can easily lead to deadlock.

The number of CTAs that can be scheduled onto a single
[SM](#streaming-multiprocessor) sets the
[achievable occupancy](#occupancy) and depends on a number of
factors. Fundamentally, the
[SM](#streaming-multiprocessor) has a limited set
of resources — lines in the
[register file](#register-file), "slots" for
[warps](#warp), bytes of
[shared memory](#shared-memory) in the
[L1 data cache](#l1-data-cache) — and each CTA uses
a certain amount of those resources (as calculated at
[compile](#nvcc) time) when scheduled onto an
[SM](#streaming-multiprocessor).

### Kernel

![A single kernel launch corresponds to a [thread block grid](#thread-block-grid) in the [CUDA programming model](#cuda-programming-model). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](dist/diagrams/light-cuda-programming-model.png)

A kernel is the unit of
[CUDA](#cuda-programming-model) code that
programmers typically write and compose, akin to a procedure or function in
languages targeting CPUs.

Unlike procedures, a kernel is called ("launched") once and returns once, but is
executed many times, once each by a number of
[threads](#thread). These executions are generally
concurrent (their execution order is non-deterministic) and parallel (they occur
simultaneously on different execution units).

The collection of all threads executing a kernel is organized as a kernel grid —
aka a [thread block grid](#thread-block-grid), the
highest level of the
[CUDA programming model](#cuda-programming-model)'s
[thread hierarchy](#thread-hierarchy). A kernel
grid executes across multiple
[Streaming Multiprocessors (SMs)](#streaming-multiprocessor)
and so operates at the scale of the entire GPU. The matching level of the
[memory hierarchy](#memory-hierarchy) is the
[global memory](#global-memory).

In [CUDA C++](#cuda-c), kernels are passed pointers
to [global memory](#global-memory) on the device
when they are invoked by the host and return nothing — they just mutate memory.

To give a flavor for CUDA kernel programming, let's walk through two
implementations of the "hello world" of CUDA kernels: matrix multiplication of
two square matrices, `A` and `B`. The two implementations will differ in how
they map the textbook matrix multiplication algorithm onto the
[thread hierarchy](#thread-hierarchy) and
[memory hierarchy](#memory-hierarchy).

In the simplest implementation, inspired by the first matmul kernel in
[Programming Massively Parallel Processors](https://www.amazon.com/dp/0323912311)
(4th edition, Figure 3.11), each [thread](#thread)
does all of the work to compute one element of the output matrix -- loading in
turn each element of a particular `row` of `A` and a particular `col`umn of `B`
into [registers](#registers), multiplying the
paired elements, summing the results, and placing the sum back in
[global memory](#global-memory).

```cpp
__global__ void mm(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

In this kernel, each [thread](#thread) does one
floating point operation (FLOP) per read from
[global memory](#global-memory): a multiply and an
add; a load from `A` and a load from `B`. You'll never
[use the whole GPU](https://modal.com/blog/gpu-utilization-guide) that way,
since the [arithmetic bandwidth](#arithmetic-bandwidth) of the
[CUDA Cores](#cuda-core) in FLOPs/s is much higher
than the [memory bandwidth](#memory-bandwidth) between the
[GPU RAM](#gpu-ram) and the
[SMs](#streaming-multiprocessor).

We can increase
[the ratio of FLOPs to memory operations](#arithmetic-intensity)
by more carefully mapping the work in this algorithm onto the
[thread hierarchy](#thread-hierarchy) and
[memory hierarchy](#memory-hierarchy). In the
"tiled" matmul kernel below, inspired by that in Figure 5.9 of the 4th edition
of
[Programming Massively Parallel Processors](https://www.amazon.com/dp/0323912311),
we map the loading of submatrices of `A` and `B` and the computation of
submatrices of `C` onto
[shared memory](#shared-memory) and
[thread blocks](#thread-block) respectively.

```cpp
#define TILE_WIDTH 16

__global__ void mm(float* A, float* B, float* C, int N) {

    // declare variables in shared memory ("smem")
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float c_output = 0;
    for (int m = 0; m < N/TILE_WIDTH; ++m) {

        // each thread loads one element of A and one of B from global memory into smem
        As[threadIdx.y][threadIdx.x] = A[row * N + (m * TILE_WIDTH + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * N + col];

        // we wait until all threads in the 16x16 block are done loading into smem
        // so that it contains two 16x16 tiles
        __syncthreads();

        // then we loop over the inner dimension,
        // performing 16 multiplies and 16 adds per pair of loads from global memory
        for (int k = 0; k < TILE_WIDTH; ++k) {
            c_output += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        // wait for all threads to finish computing
        // before any start loading the next tile into smem
        __syncthreads();
    }
    C[row * N + col] = c_output;
}
```

For each iteration of the outer loop, which loads two elements, a thread runs 16
iterations of the inner loop, which does a multiply and an add, for 16 FLOPs per
global memory read.

This is still far from a fully optimized kernel for matrix multiplication.
[This worklog by Si Boehm of Anthropic](https://siboehm.com/articles/22/CUDA-MMM)
walks through optimizations that further increase the FLOP to memory read ratio
and map the algorithm even more tightly onto the hardware. Our kernels resemble
his Kernel 1 and Kernel 3; the worklog covers ten kernels.

That worklog and this article only consider writing kernels for execution on the
[CUDA Cores](#cuda-core). The absolute fastest
matrix multiplication kernels run instead on
[Tensor Cores](#tensor-core), which have a much
higher [arithmetic bandwidth](#arithmetic-bandwidth).

### Thread Block

![Thread blocks are an intermediate level of the thread group hierarchy of the [CUDA programming model](#cuda-programming-model) (left). A thread block executes on a single [Streaming Multiprocessor](#streaming-multiprocessor) (right, middle). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](dist/diagrams/light-cuda-programming-model.png)

A thread block is a level of the
[CUDA programming model's](#cuda-programming-model)
[thread hierarchy](#thread-hierarchy) below a
[grid](#thread-block-grid) but above a
[thread](#thread). It is the
[CUDA programming model's](#cuda-programming-model)
abstract equivalent of the concrete
[cooperative thread arrays](#cooperative-thread-array)
in
[PTX](#parallel-thread-execution)/[SASS](#streaming-assembler).

Blocks are the smallest unit of thread coordination exposed to programmers in
the
[CUDA programming model](#cuda-programming-model).
Blocks must execute independently, so that any execution order for blocks is
valid, from fully serial in any order to all interleavings.

A single CUDA [kernel](#kernel) launch produces one
or more thread blocks (in the form of a
[thread block grid](#thread-block-grid)), each of
which contains one or more [warps](#warp). Blocks
can be arbitrarily sized, but they are typically multiples of the
[warp](#warp) size (32 on all current CUDA GPUs).

### Thread Block Grid

![Thread block grids are the highest level of the thread group hierarchy of the [CUDA programming model](#cuda-programming-model) (left). They map onto multiple [Streaming Multiprocessors](#streaming-multiprocessor) (right, bottom). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](dist/diagrams/light-cuda-programming-model.png)

When a CUDA [kernel](#kernel) is launched, it
creates a collection of [threads](#thread) known as
a thread block grid. Grids can be one, two, or three dimensional. They are made
up of [thread blocks](#thread-block).

The matching level of the
[memory hierarchy](#memory-hierarchy) is the
[global memory](#global-memory).

[Thread blocks](#thread-block) are effectively
independent units of computation. They execute concurrently, that is, with
indeterminate order, ranging from fully sequentially in the case of a GPU with a
single
[Streaming Multiprocessor](#streaming-multiprocessor)
to fully in parallel when run on a GPU with sufficient resources to run them all
simultaneously.

### Thread Hierarchy

![The thread hierarchy of the [CUDA programming model](#cuda-programming-model) spans from individual [threads](#thread) to [thread blocks](#thread-block) to [thread block grids](#thread-block-grid) (left), mapping onto the hardware from [CUDA Cores](#cuda-core) to [Streaming Multiprocessors](#streaming-multiprocessor) to the entire GPU (right). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](dist/diagrams/light-cuda-programming-model.png)

The thread hierarchy is a key abstraction of the
[CUDA programming model](#cuda-programming-model),
alongside the
[memory hierarchy](#memory-hierarchy). It organizes
the execution of parallel programs across multiple levels, from individual
threads up to entire GPU devices.

At the lowest level are individual
[threads](#thread). Like a thread of execution on a
CPU, each [CUDA thread](#thread) executes a stream
of instructions. The hardware resources that effect arithmetic and logic
instructions are called [cores](#core) or sometimes
"pipes". Threads are selected for execution by the
[Warp Scheduler](#warp-scheduler).

The intermediate level consists of
[thread blocks](#thread-block), which are also
known as
[cooperative thread arrays](#cooperative-thread-array)
in [PTX](#parallel-thread-execution) and
[SASS](#streaming-assembler). Each
[thread](#thread) has a unique identifier within
its [thread block](#thread-block). These thread
identifiers are index-based, to support easy assignment of work to threads based
on indices into input or output arrays. All threads within a block are scheduled
simultaneously onto the same
[Streaming Multiprocessor (SM)](#streaming-multiprocessor).
They can coordinate through
[shared memory](#shared-memory) and synchronize
with barriers.

At the highest level, multiple
[thread blocks](#thread-block) are organized into a
[thread block grid](#thread-block-grid) that spans
the entire GPU. [Thread blocks](#thread-block) are
strictly limited in their coordination and communication. Blocks within a grid
execute concurrently with respect to each other, with no guaranteed execution
order. [CUDA programs](#cuda-programming-model)
must be written so that any interleaving of blocks is valid, from fully serial
to fully parallel. That means
[thread blocks](#thread-block) cannot, for
instance, synchronize with barriers. Like
[threads](#thread), each
[thread block](#thread-block) has a unique,
index-based identifier to support assignment of work based on array index.

This hierarchy maps directly onto the
[GPU hardware](#gpu-glossary/device-software/thread) execute on individual
[cores](#core),
[thread blocks](#thread-block) are scheduled onto
[SMs](#streaming-multiprocessor), and
[grids](#thread-block-grid) utilize all available
[SMs](#streaming-multiprocessor) on the device.

### Memory Hierarchy

![[Shared memory](#shared-memory) and [global memory](#global-memory) are two levels of the memory hierarchy in the [CUDA programming model](#cuda-programming-model) (left), mapping onto the [L1 data cache](#l1-data-cache) and [GPU RAM](#gpu-ram), respectively. Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](dist/diagrams/light-cuda-programming-model.png)

As part of the
[CUDA programming model](#cuda-programming-model),
each level of the
[thread hierarchy](#thread-hierarchy) has access to
a distinct block of memory shared by all
[threads](#thread) in a group at that level: a
"memory hierarchy". This memory can be used for coordination and communication
and is managed by the programmer (not the hardware or a runtime).

For a [thread block grid](#thread-block-grid), that
shared memory is in the [GPU's RAM](#gpu-ram) and
is known as the [global memory](#global-memory).
Access to this memory can be coordinated with atomic operations and barriers,
but execution order across
[thread blocks](#thread-block) is indeterminate.

For a single [thread](#thread), the memory is a
chunk of the
[Streaming Multiprocessor's (SM's)](#streaming-multiprocessor)
[register file](#register-file). According to the
original semantics of the
[CUDA programming model](#cuda-programming-model),
this memory is private to a [thread](#thread), but
certain instructions added to
[PTX](#parallel-thread-execution) and
[SASS](#streaming-assembler) to target matrix
multiplication on [Tensor Cores](#tensor-core)
share inputs and outputs across [threads](#thread).

In between, the [shared memory](#shared-memory) for
the [thread block](#thread-block) level of the
thread hierarchy is stored in the
[L1 data cache](#l1-data-cache) of each
[SM](#streaming-multiprocessor). Careful management
of this cache — e.g. loading data into it to support the
[maximum number of arithmetic operations before new data is loaded](#arithmetic-intensity)
— is key to the art of designing [high-performance](#gpu-glossary/device-software/kernel).

### Registers

![Registers are the memory of the [memory hierarchy](#memory-hierarchy) associated with individual [threads](#thread) (left). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](dist/diagrams/light-cuda-programming-model.png)

At the lowest level of the
[memory hierarchy](#memory-hierarchy) are the
registers, which store information manipulated by a single
[thread](#thread).

The values in registers are generally stored in the
[register file](#register-file) of the
[Streaming Multiprocessor (SM)](#streaming-multiprocessor),
but they can also spill to the
[global memory](#global-memory) in the
[GPU RAM](#gpu-ram) at a substantial performance
penalty.

As when programming CPUs, these registers are not directly manipulated by
high-level languages like [CUDA C](#cuda-c). They are
only visible to a lower-level language, here
[Parallel Thread Execution (PTX)](#parallel-thread-execution).
They are typically managed by a compiler like `ptaxs`. Among the compiler's
goals is to limit the register space used by each
[thread](#thread) so that more
[thread blocks](#thread-block) can be
simultaneously scheduled into a single
[SM](#streaming-multiprocessor), increasing
[occupancy](#occupancy).

The registers used in the
[PTX](#parallel-thread-execution) instruction set
architecture are documented
[here](https://docs.nvidia.com/cuda/parallel-thread-execution/#register-state-space).
The registers used in [SASS](#streaming-assembler)
are not, to our knowledge, documented.

### Shared Memory

![Shared memory is the abstract memory associated with the [thread block](#thread-block) level (left, center) of the CUDA thread group hierarchy (left). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](dist/diagrams/light-cuda-programming-model.png)

Shared memory is the level of the
[memory hierarchy](#memory-hierarchy) corresponding
to the [thread block](#thread-block) level of the
[thread hierarchy](#thread-hierarchy) in the
[CUDA programming model](#cuda-programming-model).
It is generally expected to be much smaller but much faster (in throughput and
latency) than the [global memory](#global-memory).

A fairly typical [kernel](#kernel) therefore looks
something like this:

- load data from [global memory](#global-memory)
  into shared memory
- perform a number of arithmetic operations on that data via the
  [CUDA Cores](#cuda-core) and
  [Tensor Cores](#tensor-core)
- optionally, synchronize [threads](#thread) within
  a [thread block](#thread-block) by means of
  barriers while performing those operations
- write data back into
  [global memory](#global-memory), optionally
  preventing races across
  [thread blocks](#thread-block) by means of
  atomics

Shared memory is stored in the
[L1 data cache](#l1-data-cache) of the GPU's
[Streaming Multiprocessor (SM)](#streaming-multiprocessor).

### Global Memory

![Global memory is the highest level of the [memory hierarchy](#memory-hierarchy) in the [CUDA programming model](#cuda-programming-model). It is stored in the [GPU RAM](#gpu-ram). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](dist/diagrams/light-cuda-programming-model.png)

As part of the
[CUDA programming model](#cuda-programming-model),
each level of the
[thread hierarchy](#thread-hierarchy) has access to
matching memory from the
[memory hierarchy](#memory-hierarchy). This memory
can be used for coordination and communication and is managed by the programmer
(not the hardware or a runtime).

The highest level of that memory hierarchy is the global memory. Global memory
is global in its scope and its lifetime. That is, it is accessible by every
[thread](#thread) in a
[thread block grid](#thread-block-grid) and its
lifetime is as long as the execution of the program.

Access to data structures in the global memory can be synchronized across all
accessors using atomic instructions, as with CPU memory. Within a
[cooperative thread array](#cooperative-thread-array),
access can be more tightly synchronized, e.g. with barriers.

This level of the
[memory hierarchy](#memory-hierarchy) is typically
implemented in the [GPU's RAM](#gpu-ram) and
allocated from the host using a memory allocator provided by the
[CUDA Driver API](#cuda-driver-api) or the
[CUDA Runtime API](#cuda-runtime-api).

The terminology "global" unfortunately collides with the `__global__` keyword in
[CUDA C/C++](#cuda-c), which annotates functions that
are launched on the host but run on the device
([kernels](#kernel)), whereas global memory is only
on the device. Early CUDA architect Nicholas Wilt wrily notes that this choice
was made "for maximum developer confusion" in his
[_CUDA Handbook_](https://www.cudahandbook.com/).

## Host Software

These terms and technologies are used on the CPU (the "host" in NVIDIA's lingo)
when running GPU programs.

### CUDA (Software Platform)

CUDA stands for _Compute Unified Device Architecture_. Depending on the context,
"CUDA" can refer to multiple distinct things: a
[high-level device architecture](#cuda-device-architecture),
a
[parallel programming model for architectures with that design](#cuda-programming-model),
or a software platform that extends high-level languages like C to add that
programming model.

The vision for CUDA is laid out in the
[Lindholm et al., 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf)
white paper. We highly recommend this paper, which is the original source for
many claims, diagrams, and even specific turns of phrase in NVIDIA's
documentation.

Here, we focus on the CUDA _software platform_.

The CUDA software platform is a collection of software for developing CUDA
programs. Though CUDA software platforms exist for other languages, like
FORTRAN, we will focus on the dominant
[CUDA C++](#cuda-c) version.

This platform can be roughly divided into the components used to _build_
applications, like the
[NVIDIA CUDA Compiler Driver](#nvcc) toolchain, and
the components used _within_ or _from_ applications, like the
[CUDA Driver API](#cuda-driver-api) and the
[CUDA Runtime API](#cuda-runtime-api), diagrammed
below.

![The CUDA Toolkit. Adapted from the *Professional CUDA C Programming Guide*.](dist/diagrams/light-cuda-toolkit.png)

Built on top of these APIs are libraries of and for building optimized
[kernels](#kernel) for general and specific
domains, like [cuBLAS](#cublas) for linear algebra
and [cuDNN](#cudnn) for deep neural networks.

### CUDA C++ (programming language)

CUDA C++ is an implementation of the
[CUDA programming model](#cuda-programming-model)
as an extension of the C++ programming language.

CUDA C++ adds several features to C++ to implement the
[CUDA programming model](#cuda-programming-model),
including:

- **[Kernel](#kernel) definition** with
  **`__global__`**. CUDA [kernels](#kernel) are
  implemented as C++ functions that take in pointers and have return type
  `void`, annotated with this keyword.
- **[Kernel](#kernel) launches** with **`<<<>>>`**.
  [Kernels](#kernel) are executed from the CPU host
  using a triple bracket syntax that sets the
  [thread block grid](#thread-block-grid)
  dimensions.
- **[Shared memory](#shared-memory) allocation**
  with the `shared` keyword, **barrier synchronization** with the
  `__syncthreads()` intrinsic function, and
  **[thread block](#thread-block)** and
  **[thread](#thread) indexing** with the
  `blockDim` and `threadIdx` built-in variables.

CUDA C++ programs are compiled by a combination of host C/C++ compiler drivers
like `gcc` and the
[NVIDIA CUDA Compiler Driver](#nvcc), `nvcc`.

For information on how to use CUDA C++ on [Modal](https://modal.com), see
[this guide](https://modal.com/docs/guide/cuda).

### NVIDIA GPU Drivers

The NVIDIA GPU drivers mediate the interaction between host programs or the host
operating system and the GPU device. The primary interfaces to the GPU drivers
for applications are, in order, the
[CUDA Runtime API](#cuda-runtime-api) and the
[CUDA Driver API](#cuda-driver-api).

![The CUDA Toolkit. The NVIDIA GPU Driver is the only component that communicates directly with the GPU. Adapted from the *Professional CUDA C Programming Guide*.](dist/diagrams/light-cuda-toolkit.png)

NVIDIA has released the
[source](https://github.com/NVIDIA/open-gpu-kernel-modules) for their Linux Open
GPU [Kernel Module](#nvidia-ko).

### nvidia.ko

`nvidia.ko` is a binary
[kernel module](https://wiki.archlinux.org/title/Kernel_module) file at the core
of the [NVIDIA GPU drivers](#nvidia-gpu-drivers) for
Linux.

Like other kernel modules, it executes in privileged mode and communicates
directly with hardware on behalf of the user -- in this case, the GPU.

The Linux Open GPU Kernel Module is
[open source](https://github.com/NVIDIA/open-gpu-kernel-modules).

### CUDA Driver API

The [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
is the userspace component of the NVIDIA CUDA drivers. It provides utilities
familiar to users of the C standard library: a `cuMalloc` function for
allocating [memory](#global-memory) on GPU devices,
for example.

![The CUDA Toolkit. The CUDA Driver API sits between applications or other toolkit components and the GPU. Adapted from the *Professional CUDA C Programming Guide*.](dist/diagrams/light-cuda-toolkit.png)

Very few CUDA programs are written to directly use the CUDA Driver API. They
instead use the
[CUDA Runtime API](#cuda-runtime-api). See
[this section](https://docs.nvidia.com/cuda/cuda-driver-api/driver-vs-runtime-api.html#driver-vs-runtime-api)
of the CUDA Driver API docs.

The CUDA Driver API is generally not linked statically. Instead, it is linked
dynamically, typically under the name
[libcuda.so](#libcuda) on Linux systems.

The CUDA Driver API is binary-compatible: an application compiled against old
versions of the CUDA Driver API can run on systems with newer versions of the
CUDA Driver API. That is, the operating system's binary loader may load a newer
version of the CUDA Driver API and the program will function the same.

For details on distributing [CUDA C/C++](#cuda-c)
applications, see the
[CUDA C/C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide)
from NVIDIA.

The CUDA Driver API is closed source. You can find its documentation
[here](https://docs.nvidia.com/cuda/cuda-driver-api/index.html).

Though they are not commonly used, there are projects that attempt to provide or
use open source alternatives to the CUDA Driver API, like
[LibreCuda](https://github.com/mikex86/LibreCuda) and
[tinygrad](https://github.com/tinygrad). See
[their source code](https://github.com/tinygrad/tinygrad/blob/77f7ddf62a78218bee7b4f7b9ff925a0e581fcad/tinygrad/runtime/ops_nv.py)
for details.

### libcuda.so

The typical name for the binary shared object file that implements the
[CUDA Driver API](#cuda-driver-api) on Linux systems.
It is dynamically linked by CUDA programs. If it is missing, the drivers are
generally improperly installed.

### NVIDIA Management Library

The NVIDIA Management Library (NVML) is used for monitoring and managing the
state of NVIDIA GPUs. It exposes, for example, the power draw and temperature of
the GPU, the allocated memory, and the device's power limit and power limiting
state. For details on these metrics, including how to interpret power and
thermal readings, see
[this page on the Modal docs](https://modal.com/docs/guide/gpu-metrics).

The function of NVML are frequently accessed via the
[nvidia-smi](#nvidia-smi) command line utility, but
are also accessible to programs via wrappers, like
[pynvml in Python](https://pypi.org/project/pynvml/) and
[nvml_wrapper in Rust](https://docs.rs/nvml-wrapper/latest/nvml_wrapper/).

### libnvml.so

The typical name for the binary shared object file that implements the features
of [NVML](#nvml) on Linux systems.

### nvidia-smi

This command line utility is used to query and manage the state of the GPU
exposed by the [NVML](#nvml) management libraries.
Its outputs, a sample of which appears below, are familiar to users of NVIDIA
GPUs to the point of being a
[meme](https://x.com/boborado/status/1752724223934578760).

`nvidia-smi` reports the following:

- GPU identity information like the card's model name, a UUID, and the PCI ID
- live utilization metrics for kernel execution time and memory allocation
- live power and thermal information

For details on these metrics, including how to interpret power and thermal
readings, see
[this page on the Modal docs](https://modal.com/docs/guide/gpu-metrics).

`nvidia-smi` can also list processes currently using the GPU (`-q`, `--query`,
`pmon`). Common management tasks include setting persistence mode (`-pm`),
compute mode (`-c`), power limits (`-pl`), application/locked clocks (`-ac`,
`-lgc`, `-lmc`), and performing GPU resets (`-r`).

Output can be formatted as human-readable text or XML (`-x`). While
`nvidia-smi`'s text output format is not guaranteed to be stable, the underlying
[NVML C library](#nvml) offers a stable API for tool
development.

The documentation for `nvidia-smi` can be found
[here](https://docs.nvidia.com/deploy/nvidia-smi/), and the official Python
bindings can be found [here](http://pypi.python.org/pypi/nvidia-ml-py/).

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA B200                    On  |   00000000:51:00.0 Off |                    0 |
| N/A   27C    P0            136W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA B200                    On  |   00000000:52:00.0 Off |                    0 |
| N/A   25C    P0            140W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   2  NVIDIA B200                    On  |   00000000:62:00.0 Off |                    0 |
| N/A   27C    P0            138W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   3  NVIDIA B200                    On  |   00000000:63:00.0 Off |                    0 |
| N/A   26C    P0            138W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   4  NVIDIA B200                    On  |   00000000:75:00.0 Off |                    0 |
| N/A   27C    P0            139W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   5  NVIDIA B200                    On  |   00000000:76:00.0 Off |                    0 |
| N/A   25C    P0            140W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   6  NVIDIA B200                    On  |   00000000:86:00.0 Off |                    0 |
| N/A   27C    P0            142W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   7  NVIDIA B200                    On  |   00000000:87:00.0 Off |                    0 |
| N/A   26C    P0            138W / 1000W |       0MiB / 183359MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
```

### CUDA Runtime API

The CUDA Runtime API wraps the
[CUDA Driver API](#cuda-driver-api) and provides a
higher-level API for the same functions.

![The CUDA Toolkit. The CUDA Runtime API wraps the CUDA Driver API to make it more amenable to application programming. Adapted from the *Professional CUDA C Programming Guide*.](dist/diagrams/light-cuda-toolkit.png)

It is generally preferred over the
[Driver API](#cuda-driver-api) for better ergonomics,
but there are some small caveats around control of kernel launches and context
management. See
[this section](https://docs.nvidia.com/cuda/cuda-runtime-api/driver-vs-runtime-api.html#driver-vs-runtime-api)
of the CUDA Runtime API docs for more.

While the Runtime API may be statically linked, per
[Attachment A of the NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html#attachment-a),
it does not have to be. The shared object file for dynamic linking is usually
named [libcudart.so](#libcudart) on Linux systems.

The CUDA Runtime API is closed source. You can find its documentation
[here](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html).

### libcudart.so

The typical name for the binary shared object file that implements the
[CUDA Runtime API](#cuda-runtime-api) on Linux
systems. Deployed CUDA binaries often statically link this file, but libraries
and frameworks built on the CUDA Toolkit, like PyTorch, typically load it
dynamically.

### NVIDIA CUDA Compiler Driver

The NVIDIA CUDA Compiler Driver is a toolchain for compiling
[CUDA C/C++](#cuda-c) programs. It outputs binary
executables that conform to the host ABI and include
[PTX](#parallel-thread-execution) and/or
[SASS](#streaming-assembler) to be executed on the
GPU — a so-called "fat binary". These binaries are inspectable with the same
tools used for other binaries, like `readelf` on Linux, but can be additionally
manipulated with the specialized
[CUDA Binary Utilities](#cuda-binary-utilities).

The included [PTX](#parallel-thread-execution) code
is versioned by
[Compute Capability](#compute-capability),
configured by passing `compute_XYz` values to the `--gpu-architecture` or
`--gpu-code` options.

The included [SASS](#streaming-assembler) code is
versioned by
[SM architecture version](#streaming-multiprocessor-architecture),
configured by passing `sm_XYz` values to the `--gpu-architecture` or
`--gpu-code` options. Passing `compute_XYz` to `--gpu-code` will also trigger
the generation of [SASS](#streaming-assembler) code
with the same version as the
[PTX](#parallel-thread-execution).

Compilation of host/CPU code is done using the host system's compiler driver,
e.g. the `gcc` compiler driver. Note that compiler drivers are not to be
confused with hardware drivers, like the
[NVIDIA GPU Drivers](#nvidia-gpu-drivers).

The documentation for `nvcc` can be found
[here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/).

### NVIDIA Runtime Compiler

The NVIDIA Runtime Compiler (`nvrtc`) is a runtime compilation library for CUDA
C. It compiles [CUDA C++](#cuda-c) to
[PTX](#parallel-thread-execution) without requiring
a separate launch of the
[NVIDIA CUDA Compiler Driver](#nvcc) (`nvcc`) in
another process. It is used by some libraries or frameworks to, for example, map
generated C/C++ code to
[PTX](#parallel-thread-execution) code that can run
on a GPU.

Note that this [PTX](#parallel-thread-execution) is
then further JIT-compiled from the
[PTX](#parallel-thread-execution) IR to the
[SASS assembly](#streaming-assembler). This is done
by the [NVIDIA GPU drivers](#nvidia-gpu-drivers) and
is distinct from the compilation done by NVRTC. CUDA binaries that contain
[PTX](#parallel-thread-execution), as required for
forward compatibility, also pass through this compilation step.

NVRTC is closed source. You can find its documentation
[here](https://docs.nvidia.com/cuda/nvrtc/index.html).

### NVIDIA CUDA Profiling Tools Interface

The NVIDIA CUDA Profiling Tools Interface (CUPTI) provides a set of APIs for
profiling execution of [CUDA C++](#cuda-c),
[PTX](#parallel-thread-execution), and
[SASS](#streaming-assembler) code on GPUs.
Critically, it synchronizes timestamps across the CPU host and the GPU device.

CUPTI's interfaces are consumed by, for example, the
[NSight Systems Profiler](#nsight-systems) and the
[PyTorch Profiler](https://modal.com/docs/examples/torch_profiling).

You can find its documentation [here](https://docs.nvidia.com/cupti/).

For details on using profiling tools for GPU applications running on Modal, see
[this example from our documentation](/docs/examples/torch_profiling).

### NVIDIA Nsight Systems

NVIDIA Nsight Systems is a performance debugging tool for
[CUDA C++](#cuda-c) programs. It combines profiling,
tracing, and expert systems analysis in a GUI.

No one wakes up and says "today I want to write a program that runs on a hard to
use, expensive piece of hardware using a proprietary software stack". Instead,
GPUs are selected when normal computing hardware doesn't perform well enough to
solve a computing problem. So
[almost all GPU programs are performance-sensitive](#gpu-glossary/host-software/cupti) are
mission-critical.

You can find its documentation
[here](https://docs.nvidia.com/nsight-systems/index.html), but
[watching someone use the tool](https://www.youtube.com/watch?v=dUDGO66IadU) is
usually more helpful. For details on how to profile GPU applications on Modal,
see [our documentation](https://modal.com/docs/examples/nsys).

### CUDA Binary Utilities

The CUDA Binary Utilities are a collection of tools for examining the contents
of binaries like those output by `nvcc`, the
[NVIDIA CUDA Compiler driver](#nvcc).

One tool, `cuobjdump`, can be used to examine and manipulate the contents of
entire host binaries or of the CUDA-specific `cubin` files that are normally
embedded within those binaries.

Another, `nvidisasm`, is intended for manipulating `cubin` files. It can extract
[SASS assembler](#streaming-assembler) and
manipulate it, e.g. constructing control flow graphs and mapping assembly
instructions to lines in CUDA program files.

You can find their documentation
[here](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html).

### cuBLAS

cuBLAS (CUDA Basic Linear Algebra Subroutines) is NVIDIA's high-performance
implementation of the
[Basic Linear Algebra Subprograms (BLAS)](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)
standard. It is a proprietary software library that provides highly optimized
[kernels](#kernel) for common linear algebra
operations.

Instead of writing and optimizing common operations like matrix multiplication
from scratch, developers can call cuBLAS functions from their host code. The
library contains a wide array of kernels, each fine-tuned for specific data
types (e.g. FP32, FP16), matrix sizes, and
[Streaming Multiprocessor (SM) architectures](#streaming-multiprocessor-architecture).
At runtime, cuBLAS uses (unknown) internal heuristics to select the most
performant kernel and its optimal launch parameters. As a result, cuBLAS is the
foundation for most [high-performance](#gpu-glossary/device-software/kernel) libraries like
[cuDNN](#cudnn).

The single most common source of error when using cuBLAS is the matrix data
layout. For historical reasons, and to maintain compatibility with the original
BLAS standard (which was written in Fortran), cuBLAS expects matrices to be in
[column-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
This is the opposite of the commonly used row-major order in C, C++, and Python.
Furthermore, a BLAS function needs to know not just the size of the operation
(e.g., `M`, `N`, `K`), but also how to find the start of each column in memory.
This is specified by the leading dimension (e.g. `lda`). The leading dimension
is the stride between consecutive columns. When working with an entire allocated
matrix, the leading dimension is just the number of rows. However, if working
with a submatrix, the leading dimension would be the number of rows in the
larger, parent matrix from which the submatrix is taken.

Fortunately, for computationally intensive kernels like GEMM, it is not
necessary to reorder matrices from row-major to column-major. Instead, we can
use the mathematical identity that if `C = A @ B`, then `C^T = B^T @ A^T`. The
key insight is that a matrix stored in row-major order has the exact same memory
layout as its transpose stored in column-major order. Therefore, if we provide
our row-major matrices `A` and `B` to cuBLAS but swap their order in the
function call (along with their dimensions), cuBLAS will compute `C^T` and
output it in column-major order. This resulting block of memory, when
interpreted in row-major, is exactly the matrix `C` that we want. This technique
is demonstrated in the following function:

```cpp
#include <cublas_v2.h>

// performs single-precision C = alpha * A @ B + beta * C
// on row-major matrices using cublasSgemm
void sgemm_row_major(cublasHandle_t handle, int M, int N, int K,
                     const float *alpha,
                     const float *A, const float *B,
                     const float *beta,
                     float *C) {

  // A is M x K (row-major), cuBLAS sees it as A^T (K x M, column-major),
  //   the leading dimension of A^T is K
  // B is K x N (row-major), cuBLAS sees it as B^T (N x K, column-major),
  //   the leading dimension of B^T is N
  // C is M x N (row-major), cuBLAS sees it as C^T (N x M, column-major),
  //   the leading dimension of C^T is N

  // note the swapped A and B, and the swapped M and N
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              N, M, K,
              alpha,
              B, N,  // leading dimension of B^T
              A, K,  // leading dimension of A^T
              beta,
              C, N); // leading dimension of C^T
}
```

A complete, runnable version of this example is available on
[Godbolt](https://godbolt.org/z/axzYb75ro).

The `CUBLAS_OP_N` flag instructs the kernel to use the matrices as provided
(without an additional transpose operation from its perspective).

To use the cuBLAS library, it must be linked (e.g. using the flag `-lcublas`
when compiling with [nvcc](#nvcc)). Its functions are
exposed via the `cublas_v2.h` header.

For more information on cuBLAS, see the
[official cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/).

### cuDNN

NVIDIA's cuDNN (CUDA Deep Neural Network) is a library of primitives for
building GPU-accelerated deep neural networks.

cuDNN provides highly optimized [kernels](#kernel)
for operations arising frequently in neural networks. These include convolution,
self-attention (including scaled dot-product attention, aka "Flash Attention"),
matrix multiplication, various normalizations, poolings, etc.

cuDNN is a key library at the application layer of the
[CUDA software platform](#cuda-software-platform),
alongside its sibling library, [cuBLAS](#cublas).
Deep learning frameworks like PyTorch typically leverage
[cuBLAS](#cublas) for general-purpose linear algebra,
such as the matrix multiplications that form the core of dense (fully-connected)
layers. They rely on cuDNN for more specialized primitives like convolutional
layers, normalization routines, and attention mechanisms.

In modern cuDNN code, computations are expressed as operation graphs, which can
be constructed using open source
[Python and C++ frontend APIs](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/developer/overview.html).
via the declarative
[Graph API](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.14.0/developer/graph-api.html).

This API allows the developer to define a sequence of operations as a graph,
which cuDNN can then analyze to perform optimizations, most importantly
operation fusion. In operation fusion, a sequence of operations like
Convolution + Bias + ReLU are merged ("fused") into a single operation run as a
single [kernel](#kernel). Operation fusion helps
reduce demand on [memory bandwidth](#memory-bandwidth) by
keeping program intermediates in
[shared memory](#shared-memory) throughout a
sequence of operations.

The frontends interact with a lower-level, closed source
[C backend](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/overview.html),
which exposes an API for legacy use cases or direct C FFI.

For any given operation, cuDNN maintains multiple underlying implementations and
uses (unknown) internal heuristics to select the most performant one for the
target
[Streaming Multiprocessor (SM) architecture](#streaming-multiprocessor-architecture),
data types, and input sizes.

cuDNN's initial claim to fame was accelerating convolutional neural networks on
Ampere
[SM architecture](#streaming-multiprocessor-architecture)
GPUs. For Transformer neural networks on Hopper and especially Blackwell
[SM architectures](#streaming-multiprocessor-architecture),
NVIDIA has tended to place more emphasis on the
[CUTLASS](https://github.com/NVIDIA/cutlass) library.

For more information on cuDNN, see the
[official cuDNN documentation](https://docs.nvidia.com/deeplearning/cudnn/), and
the [open source frontend APIs](https://github.com/NVIDIA/cudnn-frontend).

## Performance

GPUs are used when the performance of an application is inadequate on
general-purpose hardware. That makes programming for them quite different from
most other forms of programming.

For a traditional computer application, like a database management system or a
web server, correctness is the primary concern. If the application loses data or
returns incorrect results, then the application has failed. Performance is often
ignored.

When programming GPUs, correctness is typically poorly-defined. "Correct"
outputs are defined only up to some number of significant bits or only for some
underdetermined subset of "well-behaved" inputs. And correctness is at best
necessary but not sufficient. If the programmers of the application cannot
achieve superior performance (per second, per dollar, or per Watt), then the
application has failed. Programming GPUs is too hard and too limited, and
running them too expensive, for anything else to be the case.

At NVIDIA, this fact is captured in a pithy slogan: "performance is the
product".

This section of the GPU Glossary collects together and defines the key terms
that you need to understand to optimize the performance of programs running on
GPUs.

Roughly speaking, it should cover every term that you run across when using
[NSight Compute](https://developer.nvidia.com/nsight-compute) to debug GPU
[kernel](#kernel) performance issues.

### Performance Bottleneck

The literal neck of a bottle limits the rate at which liquid can be poured; a
metaphorical performance bottleneck in a system limits the rate at which tasks
can be completed.

![[Roofline diagrams](#roofline-model) like this one are used to quickly identify performance bottlenecks in throughput-oriented systems. Adapted from [Williams, Waterman, and Patterson (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf).](dist/diagrams/light-roofline-model.png)

Bottlenecks are the target of performance optimization. The textbook approach to
optimization is to

- determine the bottleneck,
- elevate the bottleneck until it is no longer such, and
- repeat on the new bottleneck.

This approach is formalized in, for instance, the
["Theory of Constraints" by Eliyahu Goldratt](https://en.wikipedia.org/wiki/Theory_of_constraints)
that helped
[transmit the Toyota approach to manufacturing to manufacturers worldwide](https://www.leanproduction.com/theory-of-constraints/),
[thence to software engineering and operations](https://youtu.be/1jU7iUr-0xE).

In [this talk for Jane Street](https://youtu.be/139UPjoq7Kw?t=1229), Horace He
broke down the work done by the [kernels](#kernel)
of programs run on GPUs into three categories:

- Compute (running floating point operations on
  [CUDA Cores](#cuda-core) or
  [Tensor Cores](#tensor-core))
- Memory (moving data in the system's
  [memory hierarchy](#memory-hierarchy))
- Overhead (everything else)

And so for GPU [kernels](#kernel), performance
bottlenecks fall into three main\* categories:

- [compute-bound](#compute-bound)
  [kernels](#kernel), bottlenecked by the
  [arithmetic bandwidth](#arithmetic-bandwidth) of compute
  units, like large matrix-matrix multiplication,
- [memory-bound](#memory-bound)
  [kernels](#kernel), bottlenecked by the
  [bandwidth of memory subsystems](#memory-bandwidth), like
  large vector-vector multiplication, and
- [overhead-bound](#overhead)
  [kernels](#kernel) bottlenecked by latency, like
  small array operations.

[Roofline model](#roofline-model) analysis helps quickly
identify whether a program's performance is bottlenecked by
compute/[arithmetic bandwidth](#arithmetic-bandwidth) or
[memory bandwidth](#memory-bandwidth).

<small>Of course, _any_ resource can become a bottleneck. For instance, power
ingress and heat egress can and does bottleneck some GPUs below their
theoretical maximum performance. See
[this article from NVIDIA](https://developer.nvidia.com/blog/nvidia-sets-new-generative-ai-performance-and-scale-records-in-mlperf-training-v4-0/)
explaining a 4% end-to-end performance improvement by redirecting power from the
L2 cache to the
[Streaming Multiprocessors](#streaming-multiprocessor)
or
[this article from Horace He](https://www.thonking.ai/p/strangely-matrix-multiplications)
indicating that matrix multiplication performance varies depending on the input
data via the amount of power demanded by transistor switching. But compute and
memory are the most important resources and the most common bottlenecks.</small>

### Roofline Model

The roofline model is a simplified, visual model of performance used to quickly
determine whether a program is bound by
[memory bandwidth](#memory-bandwidth) or
[arithmetic bandwidth](#arithmetic-bandwidth).

![[Kernels](#kernel) to the left of the ridge point are [limited by the bandwidth of the memory subsystem](#memory-bound) and [kernels](#kernel) to the right of the ridge point are [limited by the bandwidth of the arithmetic subsystem](#compute-bound). Diagram adapted from [Williams, Waterman, and Patterson (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf), which introduced the roofline model.](dist/diagrams/light-roofline-model.png)

In the roofline model, two hardware‑derived "roofs" put a "ceiling" on the
possible performance:

- the "compute roof" – the [peak rate](#peak-rate) of the
  target hardware ([CUDA Cores](#cuda-core) or
  [Tensor Cores](#tensor-core)), aka the
  [arithmetic bandwidth](#arithmetic-bandwidth)
- the "memory roof" – the peak memory throughput of the target hardware, aka the
  [memory bandwidth](#memory-bandwidth).

These are visualized on a plane with the
[arithmetic intensity](#arithmetic-intensity) (in operations
per byte) on the x-axis and the performance (in operations per second) on the
y-axis. The "compute roof" is a horizontal line with height equal to the
[arithmetic bandwidth](#arithmetic-bandwidth). The "memory
roof" is a slanted line with slope equal to the
[memory bandwidth](#memory-bandwidth). Slope is "rise over
run", and so the line has units of bytes per second (operations per second
divided by operations per byte).

A specific [kernel's](#kernel) x-coordinate tells
you instantly whether it is fundamentally
[compute-bound](#compute-bound) (points beneath the flat roof)
or [memory-bound](#memory-bound) (points beneath the slanted
roof). [Kernels](#kernel) are rarely up against
either roof due to the effects of [overhead](#overhead).

The point on the boundary, i.e. where the diagonal and horizontal roof meet, is
called the "ridge point". Its x-coordinate is the minimum
[arithmetic intensity](#arithmetic-intensity) required to be
able to escape the memory
[bottleneck](#performance-bottleneck). Computer systems whose
ridge point is further to the left are easier to achieve maximum performance on,
but the relatively poor scaling of memory relative to compute generally has
pushed the ridge points of systems to the right over time.

The compute and memory roofs need only be derived once per subsystem (though
importantly they vary depending on the subsystem, not just the system;
[Tensor Cores](#tensor-core) have more FLOPS than
[CUDA Cores](#cuda-core)).

NVIDIA's NSight Compute tool for [kernel](#kernel)
performance engineering automatically performs roofline analysis for profiled
[kernels](#kernel).

The roofline model is deceptively simple. Note that, for instance, system
latencies do not appear anywhere in the diagram, only bandwidths and
throughputs. It is simple because it is highly opinionated, and understanding
those opinions and their reasoning is key to understanding the power and the
proper application of the roofline.

The roofline model was introduced by Samuel Williams, Andrew Waterman, and David
Patterson in
[this 2008 paper](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf).
They introduced it in the face of several hardware scaling trends that shaped
system architectures before and since.

First, as Patterson separately observed in a famous 2004 paper,
["latency lags bandwidth"](https://dl.acm.org/doi/pdf/10.1145/1022594.1022596).
More specifically, across subsystems like compute, memory, and storage, a linear
improvement in latency has historically been accompanied by a quadratic
improvement in bandwidth. This suggested that future systems would be, like
GPUs, throughput-oriented.

Second, as has long been observed, compute subsystems (like processor cores)
have scaled their performance much more rapidly than memory subsystems like
[caches](#l1-data-cache) and
[DRAM](#gpu-ram). This was popularized as the
["memory wall"](https://www.eecs.ucf.edu/~lboloni/Teaching/EEL5708_2006/slides/wulf94.pdf)
by Wulf and McKee in 1994.

Finally, the early 2000s saw the end of
[Dennard scaling](https://en.wikipedia.org/wiki/Dennard_scaling), aka increasing
clock speed at equal power, due primarily to the fixed leakage current of
transistors, which posed power draw and heat dissipation problems. Increasing
clock speed had previously buoyed general purpose, latency-oriented systems like
CPUs, over special purpose hardware. This slowdown was not accompanied by a
slowdown in [Moore's Law](https://en.wikipedia.org/wiki/Moore%27s_law), aka
increasing transistor count per chip. The architectural solution to an abundance
of transistors but scarcity of power was hardware specialization: disaggregating
computers into components specialized in completing distinct tasks. For a
well-documented example, see the
[Pixel Visual Core](https://blog.google/products/pixel/pixel-visual-core-image-processing-and-machine-learning-pixel-2/)
image co-processor, explained in detail in chapter 7 of the sixth edition of
Hennessy and Patterson's
[_Computer Architecture_](https://archive.org/details/computerarchitectureaquantitativeapproach6thedition/page/n13/mode/2up).

Taken together, these trends correctly suggested to the authors that future
systems would be throughput-oriented and that among the various bandwidths at
play, the [bandwidth of memory subsystems](#memory-bandwidth)
would be the primary
[performance bottleneck](#performance-bottleneck).
Applications of those systems that wanted to achieve peak performance would
therefore need to have high operational intensity for that hardware's
specialized operations — in the case of GPUs,
[arithmetic intensity](#arithmetic-intensity) for
[Tensor Cores](#tensor-core), which is to say very
large matrix multiplications.

### Compute-bound

[Kernels](#kernel) that are compute-bound are
limited by the [arithmetic bandwidth](#arithmetic-bandwidth)
of the [CUDA Cores](#cuda-core) or
[Tensor Cores](#tensor-core).

![In the [roofline diagram](#roofline-model) above, [kernels](#kernel) underneath the blue line are compute-bound. Diagram adapted from [Williams, Waterman, and Patterson (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf).](dist/diagrams/light-roofline-model.png)

Compute-bound kernels are characterized by high
[arithmetic intensity](#arithmetic-intensity) (many arithmetic
operations per byte of memory loaded or stored).
[Utilization of arithmetic pipes](#pipe-utilization) is the
limiting factor for a compute-bound kernel.

Technically, compute-boundedness is only defined for a single
[kernel](#kernel), as part of the
[roofline model](#roofline-model), but with a bit of squinting
it can be generalized to cover the multiple
[kernels](#kernel) that make up a typical workload.

Large diffusion model inference workloads are generally compute-bound.
Contemporary large language model inference workloads are often compute-bound
during batch prefill/prompt processing, when each weight can be loaded into
[shared memory](#shared-memory) once and then used
across many tokens.

Let's do a simple estimation, inspired by
[kipperrii](https://twitter.com/kipperrii)'s
[Transformer inference arithmetic](https://kipp.ly/transformer-inference-arithmetic)
framework, of the minimum latency between tokens (inter-token latency or time
per output token) for compute-bound Transformer language model inference. Assume
the model has 500B parameters, stored in 16-bit precision, for a total of 1 TB.
This model will perform roughly one trillion floating point operations (one
multiply and one accumulate per parameter) per batch element. Run on a GPU with
one petaFLOP/s of
[arithmetic bandwidth](#arithmetic-bandwidth) for 16-bit
matrix math, the minimum latency between tokens, assuming compute-boundedness,
is one millisecond per batch element.

Note that for this GPU to be compute-bound at batch size one, it would need a
[memory bandwidth](#memory-bandwidth) of 1 PB/s (so that it
can load all 1 TB of weights in one ms). Contemporary
[memory bandwidths](#memory-bandwidth) are in the TB/s range,
and so batches of hundreds of inputs are required to provide sufficient
[arithmetic intensity](#arithmetic-intensity) for execution to
be compute-bound.

For more on LLM inference, see our
[LLM Engineer's Almanac](https://modal.com/llm-almanac/summary).

### Memory-bound

[Kernels](#kernel) that are memory-bound are
limited by the [memory bandwidth](#memory-bandwidth) of the
GPU.

![Roofline diagrams, like the one above, help identify whether a program's performance is bottlenecked by compute power, memory bandwidth, or something else Diagram adapted from [Williams, Waterman, and Patterson (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf).](dist/diagrams/light-roofline-model.png)

Specifically, they are limited by
[the bandwidth](#memory-bandwidth) between the
[GPU RAM](#gpu-ram) and the
[local cache](#l1-data-cache) of the
[Streaming Multiprocessors](#streaming-multiprocessor),
because the problems of interest for GPU performance generally have
[working set sizes](https://en.wikipedia.org/wiki/Working_set_size) much larger
than any higher level of the
[memory hierarchy](#memory-hierarchy).

Memory-bound kernels have a lower
[arithmetic intensity](#arithmetic-intensity) (fewer
operations per byte moved), relative to the ridge point of their
[roofline model](#roofline-model).

Technically, memory-boundedness is only defined for a single
[kernel](#kernel), as part of the
[roofline model](#roofline-model), but with a bit of squinting
it can be generalized to cover the multiple
[kernels](#kernel) that make up a typical workload.

Contemporary large language model inference workloads are often memory-bound
during the decode/output generation stage, when the weights must be loaded once
in each forward pass. That happens once per output token, unless multi-token
prediction or speculative decoding are used, which makes it easy to calculate
the minimum latency between tokens (intertoken latency or time per output token)
for memory-bound Transformer large language model inference.

Assume the model has 500B parameters, stored in 16-bit precision, for a total of
1 TB. If we run inference on a single GPU with a
[memory bandwidth](#memory-bandwidth) of 10 TB/s, we can load
the weights once every 100 ms, and that puts a lower bound on our intertoken
latency. By batching multiple inputs together, we can linearly increase the
number of floating point operations done per parameter loaded (the
[arithmetic intensity](#arithmetic-intensity)), in principle
up the point of [compute-boundedness](#compute-bound), without
incurring any additional latency, which implies that the throughput improves
linearly in the batch size.

For more on LLM inference, see our
[LLM Engineer's Almanac](https://modal.com/llm-almanac/summary).

### Arithmetic Intensity

Arithmetic intensity is the ratio of arithmetic operations to memory operations
in a [kernel](#kernel).

![In the [roofline model](#roofline-model), operational/arithmetic intensity is plotted on the horizontal axis. Diagram adapted from [Williams, Waterman, and Patterson (2008)](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf).](dist/diagrams/light-roofline-model.png)

A high arithmetic intensity indicates that a
[kernel](#kernel) performs many arithmetic
operations per byte loaded. Due to the high ratio between
[arithmetic bandwidth](#arithmetic-bandwidth) and
[memory bandwidth](#memory-bandwidth) in modern GPUs, the most
efficient kernels have high arithmetic intensity. That means that when elevating
a memory [bottleneck](#performance-bottleneck), we can often
shift work from the memory subsystem to the compute subsystem, saving on
[memory bandwidth](#memory-bandwidth) but adding to the load
on the arithmetic units.

For example, compressing data in
[global memory](#global-memory) reduces memory
traffic since fewer bytes need to be transferred, but the compute units must
perform additional decompression operations. If we were previously
[bottlenecked](#performance-bottleneck) by memory, this can
improve performance. It also increases the ratio of FLOPs to bytes moved,
increasing the arithmetic intensity.

As another example, the
[backpropagation algorithm](https://www.nature.com/articles/323533a0) creates
long-lived intermediates (activation values) that generally must stored in
[global memory](#global-memory) during a forward
pass and then retrieved during a backwards pass. In some cases, it is faster to
store only a fraction of these intermediates and then recompute the remainder (a
technique known as [gradient checkpointing](https://arxiv.org/abs/1604.06174)),
which increases arithmetic intensity.

Because different algorithms inherently have different operational and memory
complexities, they inherently scale differently in arithmetic intensity. An
algorithm with O(1) operational complexity and O(N) memory complexity has O(1/N)
arithmetic intensity scaling, while one with O(N) operational complexity and
O(1) memory complexity has O(N) arithmetic intensity scaling.

| **Kernel**                |    **FLOPs** | **Bytes Moved** | **Arithmetic Intensity** | **Arithmetic Intensity Scaling** |
| :------------------------ | -----------: | --------------: | -----------------------: | -------------------------------: |
| SAXPY y = ax + y          |           2N |              8N |                      1/4 |                             O(1) |
| Single-Precision Real FFT | 5/2 N log(N) |             16N |              5/32 log(N) |                        O(log(N)) |
| SGEMM                     |         2N^3 |            6N^2 |                      N/8 |                             O(N) |

Notably, matrix multiplication scales linearly, i.e. is O(N), in arithmetic
intensity — it is O(N^3) in operational complexity and O(N^2) in memory
complexity. This favorable scaling makes it easy to map applications of matrix
multiplication onto arithmetic-intensity-oriented hardware (see discussion in
the [article on roofline modeling](#roofline-model)). It is a
key secret to the success of machine learning algorithms based on matrix
multiplication, like neural networks, in the past few decades.

For a discussion of arithmetic intensity as applied to Bahdanau attention, used
in Transformer neural networks, see
[this paper](https://arxiv.org/abs/2505.21487) by Zadouri, Strauss, and Dao.

The minimum arithmetic intensity required for work to be
[compute-bound](#compute-bound) (that is, to be past the ridge
point of the [roofline model](#roofline-model)) is a fixed
parameter of a system and so only needs to be derived once. Ridge point
arithmetic intensities for recent NVIDIA data center GPUs appear in the table
below. Notice that the highest ridge point has increased going from the Ampere
to Hopper to Blackwell
[Streaming Multiprocessor architectures](#streaming-multiprocessor-architecture).

| **System (Compute / Memory)**                                                                                                                               | **[Arithmetic Bandwidth](#arithmetic-bandwidth) (TFLOPs/s)** | **[Memory Bandwidth](#memory-bandwidth) (TB/s)** | **[Ridge Point](#roofline-model) (FLOPs/byte)** |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------- | -----------------------------------------------------------------------------: | -----------------------------------------------------------------: | ----------------------------------------------------------------: |
| [A100 80GB SXM BF16 TC / HBM2e](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) |                                                                            312 |                                                                  2 |                                                               156 |
| [H100 SXM BF16 TC / HBM3](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)                                                            |                                                                            989 |                                                               3.35 |                                                               295 |
| [B200 BF16 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                   |                                                                           2250 |                                                                  8 |                                                               281 |
| [H100 SXM FP8 TC / HBM3](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)                                                             |                                                                           1979 |                                                               3.35 |                                                               592 |
| [B200 FP8 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                    |                                                                           4500 |                                                                  8 |                                                               562 |
| [B200 FP4 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                    |                                                                           9000 |                                                                  8 |                                                              1125 |

### Overhead

Overhead latency is the time spent with no useful work being done.

Unlike time spent [bottlenecked](#performance-bottleneck) on
[compute](#compute-bound) or
[memory](#memory-bound), during which the GPU is working as
fast as possible, latency from overhead represents time where the GPU is instead
waiting to receive work.

Overhead often comes from CPU-side bottlenecks that prevent the GPU from
receiving work fast enough. For example, CUDA API call overhead adds on the
order of 10 μs per kernel launch. Moreover, frameworks like PyTorch or
TensorFlow spend time deciding which
[kernel](#kernel) to launch, which can take many
microseconds. We generally use the term "host overhead" here, though it's not
entirely standardized.
[CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/), which collect a
number of device-side [kernels](#kernel) together
into a single host-side launch, are a common solution to these overheads. For
more, see the
[_CUDA Techniques to Maximize Concurrency and System Utilization_ talk at GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72686/).

"Memory overhead" or "communications overhead" is overhead latency incurred
moving data back and forth from the CPU to the GPU or from one GPU to another.
But when communication bandwidth is the limiting factor, it's often better to
think of it as a form of [memory-boundedness](#memory-bound)
where the "memory" is distributed across machines.

### Little's Law

Little's Law establishes the amount of concurrency required to fully
[hide latency](#latency-hiding) with throughput.

```
concurrency (ops) = latency (s) * throughput (ops/s)
```

Little's Law is described as "the most important of the fundamental laws" of
analysis in
[the classic quantitative systems textbook by Lazowska and others](https://homes.cs.washington.edu/~lazowska/qsp/Images/Chap_03.pdf).

Little's Law determines how many instructions must be "in flight" for GPUs to
[hide latency](#latency-hiding) through
[warp](#warp) switching by
[warp schedulers](#warp-scheduler) (aka
fine-grained thread-level parallelism, like
[simultaneous multi-threading](https://en.wikipedia.org/wiki/Simultaneous_multithreading)
in CPUs).

If a GPU has a peak throughput of 1 instruction per cycle and a memory access
latency of 400 cycles, then 400 concurrent memory operations are needed across
all [active warps](#warp-execution-state) in a program. If the
throughput goes up to 10 instructions per cycle, then the program needs 4000
concurrent memory operations to properly take advantage of the increase. For
more detail, see the article on
[latency hiding](#latency-hiding).

For a non-trivial application of Little's Law, consider the following
observation, from Section 4.3 of
[Vasily Volkov's PhD thesis](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.pdf)
on [latency hiding](#latency-hiding): the number of warps
required to hide pure memory access latency is not much higher than that
required to hide pure arithmetic latency (30 vs 24, in his experiment).
Intuitively, the longer latency of memory accesses would seem to require more
concurrency. But the concurrency is determined not just by latency but also by
throughput. And because [memory bandwidth](#memory-bandwidth)
is so much lower than
[arithmetic bandwidth](#arithmetic-bandwidth), the required
concurrency turns out to be roughly the same — a useful form of balance for a
[latency hiding](#latency-hiding)-oriented system that will
mix arithmetic and memory operations.

### Memory Bandwidth

Memory bandwidth is the maximum rate at which data can be transferred between
different levels of the
[memory hierarchy](#memory-hierarchy).

It represents the theoretical maximum achievable throughput for moving data in
bytes per second. It determines the slope of the "memory roof" in a
[roofline model](#roofline-model) of the hardware.

There are many memory bandwidths in a complete system — one between each level
of the [memory hierarchy](#memory-hierarchy).

The most important bandwidth is that between the
[GPU RAM](#gpu-ram) and the
[register files](#register-file) of the
[Streaming Multiprocessors (SMs)](#streaming-multiprocessor),
because the [working sets](https://en.wikipedia.org/wiki/Working_set_size) of
most [kernels](#kernel) only fit in
[GPU RAM](#memory-hierarchy), not anywhere higher
up in the [memory hierarchy](#memory-hierarchy). It
is for this reason that that bandwidth is the primary one used in
[roofline modeling](#roofline-model) of GPU
[kernel](#kernel) performance.

Contemporary GPUs have memory bandwidths measured in terabytes per second. For
example, [B200 GPUs](https://modal.com/blog/introducing-b200-h200) have a
(bidirectional) memory bandwidth of 8 TB/sec to their HBM3e memory. This is much
lower than the [arithmetic bandwidth](#arithmetic-bandwidth)
of the [Tensor Cores](#tensor-core) in these GPUs,
leading to increased [ridge point](#roofline-model)
[arithmetic intensity](#arithmetic-intensity).

Representative bandwidth numbers for NVIDIA data center GPUs between the Ampere
and Blackwell
[Streaming Multiprocessor architecures](#streaming-multiprocessor-architecture)
are listed in the table below.

| **System (Compute / Memory)**                                                                                                                               | **[Arithmetic Bandwidth](#arithmetic-bandwidth) (TFLOPs/s)** | **Memory Bandwidth (TB/s)** | **[Ridge Point](#roofline-model) (FLOPs/byte)** |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------- | -----------------------------------------------------------------------------: | --------------------------: | ----------------------------------------------------------------: |
| [A100 80GB SXM BF16 TC / HBM2e](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) |                                                                            312 |                           2 |                                                               156 |
| [H100 SXM BF16 TC / HBM3](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)                                                            |                                                                            989 |                        3.35 |                                                               295 |
| [B200 BF16 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                   |                                                                           2250 |                           8 |                                                               281 |
| [H100 SXM FP8 TC / HBM3](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)                                                             |                                                                           1979 |                        3.35 |                                                               592 |
| [B200 FP8 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                    |                                                                           4500 |                           8 |                                                               562 |
| [B200 FP4 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                    |                                                                           9000 |                           8 |                                                              1125 |

### Arithmetic Bandwidth

Arithmetic bandwidth is the [peak rate](#peak-rate) at which
arithmetic work can be performed by a system.

It represents the theoretical maximum of the achievable throughput for
arithmetic operations per second. It determines the height of the "compute roof"
in a [roofline model](#roofline-model) of the hardware.

There are many arithmetic bandwidths in a complete system — one for each
grouping of hardware units that provide bandwidth for executing arithmetic
operations.

On many GPUs, the most important arithmetic bandwidth is the bandwidth of the
[CUDA Cores](#cuda-core) for floating point
arithmetic. GPUs generally provide more bandwidth for floating point operations
than for integer operations, and the key to the
[Compute Unified Device Architecture (CUDA)](#cuda-device-architecture)
is that the [CUDA Cores](#cuda-core) and supporting
systems provide a unified computing interface for GPU applications (unlike prior
GPU architectures).

But in recent GPUs, the unity of the architecture has been lessened by the
introduction of [Tensor Cores](#tensor-core), which
perform only matrix multiplication operations but do so at a much higher
arithmetic bandwidth than the
[CUDA Cores](#cuda-core) -- a ratio of 100:1
between [Tensor Core](#tensor-core) and
[CUDA Core](#cuda-core) bandwidth is a good rule of
thumb. That makes the [Tensor Core](#tensor-core)
arithmetic bandwidth the most important for
[kernels](#kernel) that wish to maximize
performance.

Contemporary GPUs have [Tensor Core](#tensor-core)
arithmetic bandwidths measured in petaFLOPS — quadrillions of floating point
operations per second. For example,
[B200 GPUs](https://modal.com/blog/introducing-b200-h200) have a bandwidth of
nine PFLOPS when running 4-bit floating point matrix multiplications.

Representative bandwidth numbers for NVIDIA data center GPUs between the Ampere
and Blackwell
[Streaming Multiprocessor architecures](#streaming-multiprocessor-architecture)
are listed in the table below.

| **System (Compute / Memory)**                                                                                                                               | **Arithmetic Bandwidth (TFLOPs/s)** | **[Memory Bandwidth](#memory-bandwidth) (TB/s)** | **[Ridge Point](#roofline-model) (FLOPs/byte)** |
| :---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------: | -----------------------------------------------------------------: | ----------------------------------------------------------------: |
| [A100 80GB SXM BF16 TC / HBM2e](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) |                                 312 |                                                                  2 |                                                               156 |
| [H100 SXM BF16 TC / HBM3](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)                                                            |                                 989 |                                                               3.35 |                                                               295 |
| [B200 BF16 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                   |                                2250 |                                                                  8 |                                                               281 |
| [H100 SXM FP8 TC / HBM3](https://resources.nvidia.com/en-us-gpu-resources/h100-datasheet-24306)                                                             |                                1979 |                                                               3.35 |                                                               592 |
| [B200 FP8 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                    |                                4500 |                                                                  8 |                                                               562 |
| [B200 FP4 TC / HBM3e](https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet)                                                                    |                                9000 |                                                                  8 |                                                              1125 |

### Latency Hiding

Latency hiding is a strategy to mask long-latency operations by
[running many of them concurrently](#littles-law).

Performant GPU programs hide latency by interleaving the execution of many
[threads](#thread). This allows programs to
maintain high throughput despite long instruction latencies. When one
[warp stalls](#warp-execution-state) on a slow memory
operation, the GPU immediately switches to execute instructions from another
[eligible warp](#warp-execution-state).

This keeps all execution units busy concurrently. While one
[warp](#warp) uses
[Tensor Cores](#tensor-core) for matrix
multiplication, another might execute arithmetic on
[CUDA Cores](#cuda-core) (say,
[quantizing or dequantizing matrix multiplicands](https://arxiv.org/abs/2408.11743)),
and a third could be fetching data through the
[load/store units](#load-store-unit).

Concretely, consider the following simple instruction sequence in
[Streaming Assembler](#streaming-assembler).

```nasm
LDG.E.SYS R1, [R0]        // memory load, 400 cycles
IMUL R2, R1, 0xBEEF       // integer multiply, 6 cycles
IADD R4, R2, 0xAFFE       // integer add, 4 cycles
IMUL R6, R4, 0x1337       // integer multiply, 6 cycles
```

Executed sequentially, this would take 416 cycles to complete. We can hide this
latency by operating concurrently. If we assume we can issue one instruction
every two cycles, then, by [Little's Law](#littles-law), if we
run 832 concurrent [threads](#thread), we can still
finish the sequence once per cycle (on average), hiding the latency of memory
from consumers of the data in `R6`.

Note that [threads](#thread) are not the unit of
instruction issuance, [warps](#warp) are. Each
[warp](#warp) contains 32
[threads](#thread), and so our fragment requires
832 ÷ 32 = 13 [warps](#warp). When successfully
hiding latency, the GPU's scheduling system maintains this many
[warps](#warp) in flight, switching between them
whenever one stalls, ensuring the execution units never idle while waiting for
slow operations to complete.

For a deep dive into latency hiding on
pre-[Tensor Core](#tensor-core) GPUs, see
[Vasily Volkov's PhD thesis](https://arxiv.org/abs/2206.02874).

### Warp Execution State

The state of the [warps](#warp) running a
[kernel](#kernel) is described with a number of
non-exclusive adjectives: active, stalled, eligible, and selected.

![Warp execution states are indicated by color. Diagram inspired by the [*CUDA Techniques to Maximize Compute and Instruction Throughput*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/) talk at GTC 2025.](dist/diagrams/light-cycles.png)

A [warp](#warp) is considered _active_ from the
time its [threads](#thread) begin executing to the
time when all [threads](#thread) in the
[warp](#warp) have exited from the
[kernel](#kernel). Active
[warps](#warp) form the pool from which
[warp schedulers](#warp-scheduler) select
candidates for instruction issue each cycle (i.e. to be put in one of the issue
slots).

The maximum number of active [warps](#warp) per
[Streaming Multiprocessor (SM)](#streaming-multiprocessor)
varies by
[architecture](#streaming-multiprocessor-architecture)
and is listed in
[NVIDIA's documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=compute%2520capability#compute-capabilities)
for [Compute Capability](#compute-capability). For
instance, on an H100 SXM GPU with
[Compute Capability](#compute-capability) 9.0,
there can be up to 64 active [warps](#warp) per
[SM](#streaming-multiprocessor) (2048 threads).
Note that active [warps](#warp) are not necessarily
executing instructions. There are active
[warps](#warp) in all but one slot+cycle in the
diagram above — a high [occupancy](#occupancy).

An _eligible_ [warp](#warp) is an active
[warp](#warp) that is ready to issue its next
instruction. For a [warp](#warp) to be eligible,
the following must be true:

- the next instruction has been fetched,
- the required execution unit is available,
- all instruction dependencies have been resolved, and
- no synchronization barriers block execution.

Eligible [warps](#warp) represent the immediate
candidates for instruction issue by the
[warp scheduler](#warp-scheduler). Eligible
[warps](#warp) appear on all cycles but cycle n + 2
in the diagram above. Having no eligible
[warps](#warp) on many cycles can be bad for
performance, especially if you are primarily using lower latency arithmetic
units like [CUDA Cores](#cuda-core).

A _stalled_ [warp](#warp) is an active
[warp](#warp) that cannot issue its next
instruction due to unresolved dependencies or resource conflicts.
[Warps](#warp) become stalled for various reasons
including:

- execution dependencies, i.e. they must wait for results from previous
  arithmetic instructions,
- memory dependencies, i.e. they must wait for results from previous memory
  operations,
- pipeline conflicts, i.e. the execution resources are currently occupied.

When warps are stalled on accesses to shared memory or on long-running
arithmetic instructions, they are said to be stalled on the "short scoreboard".
When warps are stalled on accesses to GPU RAM, they are said to be stalled on
the "long scoreboard". These are hardware units inside the
[warp scheduler](#warp-scheduler).
[Scoreboarding](https://www.cs.umd.edu/~meesh/411/website/projects/dynamic/scoreboard.html)
is a technique for dependency tracking in dynamic instruction scheduling that
dates back to the "first supercomputer", the
[Control Data Corporation 6600](https://en.wikipedia.org/wiki/CDC_6600), one of
which
[disproved Euler's sum of powers conjecture](https://www.ams.org/journals/bull/1966-72-06/S0002-9904-1966-11654-3/S0002-9904-1966-11654-3.pdf)
in 1966. Unlike in CPUs, scoreboarding isn't used for out-of-order execution
within [threads](#thread) (instruction-level
parallelism), only across them (thread-level parallelism); see
[this NVIDIA patent](https://patents.google.com/patent/US7676657).

Stalled [warps](#warp) appear in multiple slots in
each cycle in the diagram above. Stalled
[warps](#warp) are not inherently bad — a large
collection of concurrently stalled [warps](#warp)
might be necessary to [hide latency](#latency-hiding) from
long-running instructions, like memory loads or
[Tensor Core](#tensor-core) instructions like
`HMMA`, which [can run for dozens of cycles](https://arxiv.org/abs/2206.02874).

A _selected_ [warp](#warp) is an eligible
[warp](#warp) chosen by the
[warp scheduler](#warp-scheduler) to receive an
instruction during the current cycle. Each cycle,
[warp schedulers](#warp-scheduler) look at their
pool of eligible [warps](#warp), select one if
there are any, and issue it an instruction. There is a selected
[warp](#warp) on each cycle with an eligible
[warp](#warp). The fraction of
[active cycles](#active-cycle) on which a
[warp](#warp) is selected and an instruction is
issued is the [issue efficiency](#issue-efficiency).

### Active Cycle

An active cycle is a clock cycle in which a
[Streaming Multiprocessor](#streaming-multiprocessor)
has at least one [active warp](#warp-execution-state)
resident. The [warp](#warp) may be
[eligible](#warp-execution-state) or
[stalled](#warp-execution-state).

![All cycles depicted in this diagram are active cycles. Diagram inspired by the [*CUDA Techniques to Maximize Compute and Instruction Throughput*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/) talk at GTC 2025.](dist/diagrams/light-cycles.png)

### Occupancy

Occupancy is the ratio of the
[active warps](#warp-execution-state) to the maximum number of
[active warps](#warp-execution-state) on a device.

![There are four warp slots per cycle on each of four clock cycles and so there are 16=4*4 total warp slots, and there are active warps in 15 of them, for an occupancy of ~94%. Diagram inspired by the [*CUDA Techniques to Maximize Compute and Instruction Throughput*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/) talk at GTC 2025.](dist/diagrams/light-cycles.png)

There are two types of occupancy measurements:

- _Theoretical Occupancy_ represents the upper limit for occupancy due to the
  kernel launch configuration and device capabilities.
- _Achieved Occupancy_ measures the actual occupancy during
  [kernel](#kernel) execution, aka on
  [active cycles](#active-cycle).

As part of the
[CUDA programming model](#cuda-programming-model),
all the [threads](#thread) in a
[thread block](#thread-block) are scheduled onto
the same
[Streaming Multiprocessor (SM)](#streaming-multiprocessor).
Each [SM](#streaming-multiprocessor) has resources
(like space in [shared memory](#shared-memory))
that must be partitioned across
[thread blocks](#thread-block) and so limit the
number of [thread blocks](#thread-block) that can
be scheduled on the
[SM](#streaming-multiprocessor).

Let's work through an example. Consider an NVIDIA H100 GPU, which has these
specifications:

```
Maximum warps/SM: 64
Maximum blocks/SM: 32
(32 bit) Registers: 65536
Shared memory (smem): 228 KB
```

For a [kernel](#kernel) using 32
[threads](#thread) per
[thread block](#thread-block), 8
[registers](#registers) per
[thread](#thread), and 12 KB
[shared memory](#shared-memory) per
[thread block](#thread-block), we end up limited by
[shared memory](#shared-memory):

```
64 > 1   = warps/block = 32 threads/block ÷ 32 threads/warp
32 < 256 = blocks/register-file = 65,536 registers/register-file ÷ (32 threads/block × 8 registers/thread)
32       = blocks/SM
19       = blocks/smem = 228 KB/smem ÷ 12 KB/block
```

Even though our [register file](#register-file) is
big enough to support 256
[thread blocks](#thread-block) concurrently, our
[shared memory](#shared-memory) is not, and so we
can only run 19 [thread blocks](#thread-block) per
[SM](#streaming-multiprocessor), corresponding to
19 [warps](#warp). This is the common case where
the size of program intermediates stored in
[registers](#registers) is much smaller than
elements of the program's
[working set](https://en.wikipedia.org/wiki/Working_set) that need to stay in
[shared memory](#shared-memory).

Low occupancy can hurt performance when there aren't enough
[eligible warps](#warp-execution-state) to
[hide the latency](#latency-hiding) of instructions, which
shows up as low instruction
[issue efficiency](#issue-efficiency) and
[under-utilized pipes](#pipe-utilization). However, once
occupancy is sufficient for [latency hiding](#latency-hiding),
increasing it further may actually degrade performance. Higher occupancy reduces
resources per [thread](#thread), potentially
[bottlenecking the kernel on registers](#register-pressure) or
reducing the [arithmetic intensity](#arithmetic-intensity)
that modern GPU architectures are designed to exploit.

More generally, occupancy measures what fraction of its maximum parallel tasks
the GPU is handling simultaneously, which is not inherently a target of
optimization in most kernels. Instead, we want to maximize the
[utilization](#pipe-utilization) of compute resources if we
are [compute-bound](#compute-bound) or memory resources if we
are [memory-bound](#memory-bound).

In particular, high-performance GEMM kernels on Hopper and Blackwell
[architecture](#streaming-multiprocessor-architecture)
GPUs often run at single-digit occupancy percentages because they don't need
many [warps](#warp) to fully saturate the
[Tensor Cores](#tensor-core).

### Pipe Utilization

Pipe utilization measures how effectively a
[kernel](#kernel) uses the execution resources
within each
[Streaming Multiprocessor (SM)](#streaming-multiprocessor).

Each [SM](#streaming-multiprocessor) contains
multiple independent execution pipes optimized for different instruction types -
[CUDA Cores](#cuda-core) for general floating-point
arithmetic, [Tensor Cores](#tensor-core) for tensor
contractions, [load/store units](#load-store-unit)
for memory access, and control flow units for branching. Pipe utilization shows
what percentage of each pipeline's [peak rate](#peak-rate) is
being achieved when that pipe is actively executing at least one
[warp](#warp), averaged across all active
[SMs](#streaming-multiprocessor).

Before debugging application performance at the level of pipe utilization, GPU
programmers should first consider
[GPU kernel utilization](https://modal.com/blog/gpu-utilization-guide) and
[SM utilization](#streaming-multiprocessor-utilization).

Pipe utilization is available in the the
`sm__inst_executed_pipe_*.avg.pct_of_peak_sustained_active` metrics from
[NSight Compute](https://developer.nvidia.com/nsight-compute) (`ncu`), where the
asterisk represents specific pipelines like
[`fma`](#cuda-core),
[`tensor`](#tensor-core),
[`lsu`](#load-store-unit), or `adu` (address).

### Peak Rate

Peak rate is the theoretical maximum rate at which a hardware system can
complete work.

Peak rate represents the absolute upper bound of GPU performance when every
execution unit operates at maximum capacity with perfect efficiency. It assumes
ideal operation, where no resource constraints
([registers](#registers),
[memory bandwidth](#memory-bandwidth), synchronization
barriers, etc.) create [bottlenecks](#performance-bottleneck).

Peak rate is the yardstick against which all achieved performance is measured.
It sets the [compute-bound](#compute-bound) "roof" in a
[roofline analysis](#roofline-model). It is the denominator in
the utilization fraction reported in
[pipe utilization](#pipe-utilization) metrics and the
[ultimate arbiter of GPU utilization](https://modal.com/blog/gpu-utilization-guide).

Poetically, NVIDIA engineers often call it the "speed of light" — the limit on
program speed imposed by physics.

Peak rate is computed directly from the fixed hardware specifications of each
GPU architecture.

For example,
[an NVIDIA H100 GPU](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)
with 132 SMs, each containing 128 FP32 cores, can issue 1 single precision Fused
Multiply Add (`FMA`) operation, which comprises 2 floating point operations per
core. That's 33,792
[instructions per clock](https://en.wikipedia.org/wiki/Instructions_per_cycle).
The H100 can operate its compute subsystem clock at a maximum rate of 1980 MHz
(million clocks per second) when using the FP32 cores, and so the peak rate is
66,908 billion FLOPS, or 66.9 TFLOPS.

This precisely matches the Peak FP32 TFLOPS (non-Tensor) rate advertised in
[NVIDIA's H100 whitepaper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c).

### Issue Efficiency

Issue efficiency measures how effectively the
[warp scheduler](#warp-scheduler) keeps execution
pipes busy by issuing instructions from
[eligible warps](#warp-execution-state).

![Of the four clock cycles in this diagram, instructions were issued on three, for an issue efficiency of 75%. Diagram inspired by the [*CUDA Techniques to Maximize Compute and Instruction Throughput*](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/) talk at GTC 2025.](dist/diagrams/light-cycles.png)

An issue efficiency of 100% means every
[scheduler](#warp-scheduler) issued an instruction
on every cycle, indicating at least one
[eligible warp](#warp-execution-state) on each cycle. Values
below 100 % signal that, during some cycles, all
[active warps](#warp-execution-state) were
[stalled](#warp-execution-state) - waiting on data, resources,
or dependencies - so the
[scheduler](#warp-scheduler) sat idle and overall
instruction throughput fell.

### Streaming Multiprocessor Utilization

SM utilization measures the percentage of time that
[Streaming Multiprocessors (SMs)](#streaming-multiprocessor)
are executing instructions.

SM utilization is akin to the more familiar
[kernel utilization](https://modal.com/blog/gpu-utilization-guide) reported by
[`nvidia-smi`](#nvidia-smi), but more fine-grained.
Instead of reporting the fraction of time that a
[kernel](#kernel) is executing anywhere on the GPU,
it reports the fraction of time all
[SMs](#streaming-multiprocessor) spend executing
[kernels](#kernel). If a
[kernel](#kernel) uses only one
[SM](#streaming-multiprocessor), e.g. because it
only has one [thread block](#thread-block), then it
will achieve 100% GPU utilization while it is active, but the SM utilization
will be at most one over the number of
[SMs](#streaming-multiprocessor) — under 1% in an
H100 GPU.

[As with GPU utilization but unlike CPU utilization](https://modal.com/blog/gpu-utilization-guide),
SM utilization should be high, even up to 100%.

But even though SM utilization is finer-grained than GPU utilization, it still
isn't fine-grained enough to capture how well the GPU's compute resources are
being used. If SM utilization is high, but performance is still inadequate,
programmers should check
[pipe utilization](#pipe-utilization), which measures how
effectively each SM uses its internal functional units. High SM utilization with
low [pipe utilization](#pipe-utilization) indicates that your
[kernel](#kernel) is running on many SMs but not
fully utilizing the computational resources within each one.

### Warp Divergence

Warp divergence occurs when threads within a
[warp](#warp) take different execution paths due to
control flow statements.

For example, consider this [kernel](#kernel):

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

When the [threads](#thread) within a
[warp](#warp) encounter the data-dependent
conditional, some [threads](#thread) must execute
block A while others must execute block B, depending on the value at
`data[idx]`. Because of this data-dependency and the structural constraints of
the
[CUDA programming model](#cuda-programming-model)
and its implementation in the
[PTX machine model](#parallel-thread-execution),
there is no way for a programmer or a compiler to avoid this split in control
flow inside of the [warp](#warp).

Instead, the [warp scheduler](#warp-scheduler) must
handle concurrent execution of these divergent code paths, which it achieves by
"masking" some [threads](#thread) so that they
don't execute the instruction. This is achieved using predicate
[registers](#registers).

Let's examine the generated
[SASS](#streaming-assembler)
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
[threads](#thread) in the
[warp](#warp) execute `FSETP.GT.AND` concurrently
(`L2`), and each [thread](#thread) gets its own
`P0` value based on the `data` value in `R4`. Then, we have a bit of
[compiler](#nvcc) cleverness: in `L3` _all_
[threads](#thread) execute the code in A, writing
to `R0`. Only those for whom `P0` is true then execute the code in B (`L4`),
over-writing the value written to `R0` in `L3`. On this instruction, the
[warp](#warp) is said to be "divergent". On `L5`,
all [threads](#thread) are back to executing the
same code. Once the
[warp scheduler](#warp-scheduler) brings them back
into alignment by issuing the same instruction on the same clock cycle, the warp
has "converged".

This is presumably more efficient than the naïve encoding of the branch into
[SASS](#streaming-assembler), which would instead
predicate both lines `L3` and `L4` — "presumably" in that we can trust the
[compiler](#nvcc) and in that, heuristically, we are
trading use of cheap, plentiful
[CUDA Core](#cuda-core) computation for more
expensive flow control. As often in GPU programming, it's better to waste
compute (an unnecessary `FADD` for every execution of `L4`) than to add
complexity, even if it's just a simple predication!

One reason compilers might aggressively avoid divergence is that in early
(pre-Volta) GPUs, divergent [warps](#warp) were
always fully serialized. While warp divergence still reduces efficiency, modern
GPUs with independent thread scheduling don't necessarily experience the full
serialization penalties.

### Branch Efficiency

Branch efficiency measures how often all
[threads](#thread) in a
[warp](#warp) take the same execution path when
encountering conditional statements.

Branch efficiency is calculated as the ratio of uniform control flow decisions
to total branch instructions executed. Control flow uniformity is measured at
the level of [warps](#warp), and so branch
efficiency indicates the absence of
[warp divergence](#warp-divergence).

Not all conditionals reduce branch efficiency. The common "bounds-check"
fragment that appears in most [kernels](https://godbolt.org/z/d1PsYYPnW)

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
```

will generally have very high branch efficiency, since most
[warps](#warp) will be composed of
[threads](#thread) that all have the same value for
the conditional, save for a single [warp](#warp)
whose [threads](#thread)' indices are above and
below `n`.

While CPUs also care about the uniformity of branching behavior, they tend to
care primarily about uniformity of branch behavior over time, as part of
hardware-controlled branch prediction and speculative execution. That is, as
circuits within the CPU accumulate data about a branch as it is encountered
multiple times during program execution, the performance should improve.

GPUs instead care about uniformity in space. That is, uniformity is measured
within [warps](#warp), whose
[threads](#thread) execute concurrently in time but
are mapped onto distinct data, and performance improves if those
[threads](#thread) branch uniformly.

### Bank Conflict

When multiple [threads](#thread) in a
[warp](#warp) simultaneously request memory within
the same bank in [shared memory](#shared-memory)
but across distinct addresses, we say there is a bank conflict.

![When [threads](#thread) access distinct [shared memory](#shared-memory) banks, accesses are serviced in parallel (left). When they all access the same bank, but at different addresses, accesses are serialized (right).](dist/diagrams/light-bank-conflict.png)

When bank conflicts occur, the accesses by the distinct
[threads](#thread) are serialized. This reduces
memory throughput substantially, that is by an integral factor, preventing the
saturation of [memory bandwidth](#memory-bandwidth).

Like other SRAM cache memories, the
[shared memory](#shared-memory) in a
[Streaming Multiprocessor](#streaming-multiprocessor)
is organized into groups called "banks". These banks can be accessed
simultaneously, which increases the bandwidth.

In GPUs, there are 32 banks, each bank is 4 bytes wide, and consecutive words of
32 bits (not 64 bits; GPUs were designed with 32-bit floats and integers in
mind) map to consecutive banks.

```
Address:  0x00  0x04  0x08  0x0C  0x10  0x14  0x18  0x1C  ...  0x7C
Bank:       0     1     2     3     4     5     6     7   ...    31

Address:  0x80  0x84  0x88  0x8C  0x90  0x94  0x98  0x9C  ...  0xFC

Bank:       0     1     2     3     4     5     6     7   ...    31
```

Addresses that differ by 32 × 4 = 128 bytes map to the same bank.
[Shared memories](#shared-memory) are roughly
kilobyte scale, and so multiple addresses map onto the same bank.

If we access sequential elements of an array in shared memory, each
[thread](#thread) in our
[warp](#warp) will hit a different bank:

```cpp
__shared__ float data[1024];  // array in shared memory

// all 32 threads access consecutive elements of data
int tid = threadIdx.x;
float value = data[tid];  // address LSBs: 0x000, 0x040, 0x080, ...
```

All 32 accesses complete in one memory transaction because each
[thread](#thread) hits a different bank. This is
depicted on the left in the figure above.

But say we wanted our [threads](#thread) to access
a column in a row-major
[shared memory](#shared-memory) array with 32
elements per row, and so we wrote:

```cpp
float value = data[tid * 32];  // address LSBs: 0x000, 0x080, 0x100 ...
// recall: floats are 4 bits wide
```

As depicted in the right side of the diagram above, all accesses hit the same
bank, Bank 0, and so must be serialized, resulting in a 32x increase in latency,
rising from on the order of ten cycles to on the order of hundreds. We could
solve this bank conflict by transposing our
[shared memory](#shared-memory) array. For more
techniques to resolve bank conflicts, see the
[_Introduction to CUDA Programming and Performance Optimization_ talk from GTC 2024](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62191/).

Note that if [threads](#thread) access the same
address in the same bank, i.e. the exact same data, conflict need not occur, as
the data can be multi-/broad-cast.

### Register Pressure

Register pressure is a colorful term used when the
[register file](#register-file) is a
[bottleneck](#performance-bottleneck).

[Registers](#registers) in the
[Parallel Thread eXecution (PTX)](#parallel-thread-execution)
language are virtual and unlimited, but the
[register files](#register-file) of the
[Streaming Multiprocessor (SM)](#streaming-multiprocessor)
are physical and so limited.

The amount of space in the
[register file](#register-file) consumed by a
[thread](#thread) is determined by the
[Streaming ASSembler (SASS)](#streaming-assembler)
code for the [kernel](#kernel), and since all
[threads](#thread) in a
[thread block](#thread-block) are scheduled onto
the same [SM](#streaming-multiprocessor), the total
space required by a [thread block](#thread-block)
is determined also by the [kernel](#kernel) launch
configuration. As the space allocated per
[thread block](#thread-block) increases, fewer
[thread blocks](#thread-block) can be scheduled
onto the same [SM](#streaming-multiprocessor),
reducing [occupancy](#occupancy) and making it more difficult
to [hide latency](#latency-hiding).

See
[this excellent article by SemiAnalysis](https://semianalysis.com/2025/06/23/nvidia-tensor-core-evolution-from-volta-to-blackwell/)
for an account of the relationship between register pressure and key features
added in recent
[Streaming Multiprocessor architectures](#streaming-multiprocessor-architecture),
like asynchronous copies (added in Ampere), the
[Tensor Memory Accelerator](#tensor-memory-accelerator)
(TMA, added in Hopper), and
[tensor memory](#tensor-memory) (added in
Blackwell).

Register pressure also occurs in CPUs, where similar register
[bottlenecks](#performance-bottleneck) limit the degree to
which loops can be
[strip-mined during auto-vectorization](https://hogback.atmos.colostate.edu/rr/old/tidbits/intel/macintel/doc_files/source/extfile/optaps_for/common/optaps_vec_mine.htm).

## Contributors

This list is incomplete; you can help by
[expanding it](https://github.com/modal-labs/gpu-glossary).

### Authors

- [Charles Frye](https://twitter.com/charles_irl) wrote the majority of the
  material and takes full responsibility for any errors.
- [Matthew Nappo](https://www.linkedin.com/in/mattnappo/) wrote the initial
  internal "GPU Glossary" document from which this sprung.
- [Harmya Bhatt](https://twitter.com/racerfunction) of
  [Tensara](https://tensara.org/) co-wrote the material on
  [performance](#perf).
- [Philip Fabianek](https://www.linkedin.com/in/philip-fabianek/) contributed
  the articles on [cuBLAS](#cublas) and
  [cuDNN](#cudnn).
- [You](https://github.com/modal-labs/gpu-glossary) can contribute to keep the
  glossary up-to-date and erratum-free!

### Design

- [Sona Dolasia](https://twitter.com/teenychairs) designed the glossary.
- [Anna Carey](https://twitter.com/anna_carey) implemented the design and UX.

### Review

- [Abhinav Upadhyay](https://twitter.com/abhi9u) of
  [Coding Confessions](https://blog.codingconfessions.com/) and
  [`@Pauleonix`](https://github.com/pauleonix) of the
  [GPU MODE Discord](https://discord.gg/gpumode), from outside Modal, provided
  valuable external technical review of the first version of the glossary. We
  particularly thank Abhinav for his perspective on comparisons with CPUs and
  Pauleonix for his detailed insights on GPU hardware internals.
- [Alex Zhang](https://alexzhang13.github.io/),
  [David Wang](https://www.linkedin.com/in/dcw02/),
  [Mark Saroufim](https://twitter.com/marksaroufim), and
  [Mit Kotak](https://mitkotak.github.io/) reviewed the material on
  [performance](#perf).
- [Akshat Bubna](https://twitter.com/akshat_b),
  [Nathan Wang](https://www.linkedin.com/in/nathan-r-wang/), and
  [Colin Weld](https://www.linkedin.com/in/colin-weld/) gave technical feedback
  on early drafts of the glossary.
- [Eric Zhang](https://twitter.com/ekzhang1) and
  [Ro Arepally](https://twitter.com/rarepally) reviewed the design and
  implementation.

### Acknowledgements

- [Mark Saroufim](https://twitter.com/marksaroufim) and Andreas Kopf for
  bringing together the [GPU MODE Discord community](https://discord.gg/gpumode)
- [Fabien Sanglard](https://twitter.com/fabynou) for authoring an
  [excellent history of CUDA GPUs](https://fabiensanglard.net/cuda)
- Jen-Hsun Huang for leading an organization that makes some pretty decent chips

### Error Correction

We thank the following GPU enthusiasts who came in through the world wide web to
correct errors:

<!-- This list is ordered alphabetically by the anchor text, ignoring case -->

- [Alex Zhang](https://alexzhang13.github.io/)
- [Erik Schultheis](https://www.linkedin.com/in/erik-schultheis-606a52119/)
- Ismail Zaidi
- [Michal Nawrot](https://github.com/michalnawrot)
- [Nicolas Blin](https://www.nicolas-blin.fr/)
