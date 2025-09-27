---
title: What is Parallel Thread Execution?
abbreviation: PTX
---

Parallel Thread eXecution (PTX) is an intermediate representation (IR) for code
that will run on a parallel processor (almost always an NVIDIA GPU). It is one
of the formats output by `nvcc`, the
[NVIDIA CUDA Compiler Driver](/gpu-glossary/host-software/nvcc). It is
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
[CUDA binary](/gpu-glossary/host-software/cuda-binary-utilities) will be
just-in-time (JIT) compiled by the host
[CUDA Drivers](/gpu-glossary/host-software/nvidia-gpu-drivers) into
device-specific [SASS](/gpu-glossary/device-software/streaming-assembler) for
execution.

In the case of NVIDIA GPUs, PTX is forward-compatible: GPUs with a matching or
higher [compute capability](/gpu-glossary/device-software/compute-capability)
version will be able to run the program, thanks to this mechanism of JIT
compilation. In this way, PTX is a
["narrow waist"](https://www.oilshell.org/blog/2022/02/diagrams.html) that
separates the worlds of hardware and software.

Some exemplary PTX:

```ptx
.reg .f32 %f<7>;
```

- a compiler directive for the
  PTX-to-[SASS](/gpu-glossary/device-software/streaming-assembler) compiler
  indicating that this kernel consumes seven 32-bit floating point
  [registers](/gpu-glossary/device-software/registers). Registers are
  dynamically allocated to groups of
  [threads](/gpu-glossary/device-software/thread)
  ([warps](/gpu-glossary/device-software/warp)) from the
  [SM](/gpu-glossary/device-hardware/streaming-multiprocessor)'s
  [register file](/gpu-glossary/device-hardware/register-file).

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

![The PTX machine model. Modified from the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-machine-model).](themed-image://ptx-machine-model.svg)

Notably, in this machine model there is a single instruction unit for multiple
processors. While each processor runs one
[thread](/gpu-glossary/device-software/thread), those threads must execute the
same instructions — hence _parallel_ thread execution, or PTX. They coordinate
with each other through
[shared memory](/gpu-glossary/device-software/shared-memory) and effect
different results by means of private
[registers](/gpu-glossary/device-software/registers).

The documentation for the latest version of PTX is available from NVIDIA
[here](https://docs.nvidia.com/cuda/parallel-thread-execution/). The instruction
sets of PTX are versioned with a number called the
"[compute capability](/gpu-glossary/device-software/compute-capability)", which
is synonymous with "minimum supported
[Streaming Multiprocessor architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)
version".

Writing in-line PTX by hand is uncommon outside of the cutting edge of
performance, similar to writing in-line `x86_64` assembly, as is done in
high-performance vectorized query operators in analytical databases and in
performance-sensitive sections of operating system kernels. At the time of writing
in September of 2025, in-line PTX is the only way to take advantage of some
Hopper-specific hardware features like the `wgmma` and `tma` instructions, as in
[Flash Attention 3](https://arxiv.org/abs/2407.08608) or in the
[Machete w4a16 kernels](https://youtu.be/-4ZkpQ7agXM). Viewing
[CUDA C/C++](/gpu-glossary/host-software/cuda-c),
[SASS](/gpu-glossary/device-software/streaming-assembler), and
[PTX](/gpu-glossary/device-software/parallel-thread-execution) together is
supported on [Godbolt](https://godbolt.org/z/5r9ej3zjW). See the
[NVIDIA "Inline PTX Assembly in CUDA" guide](https://docs.nvidia.com/cuda/inline-ptx-assembly/)
for details.
