---
title: What is Streaming Assembler?
abbreviation: SASS
---

[Streaming ASSembler](https://stackoverflow.com/questions/9798258/what-is-sass-short-for)
(SASS) is the assembly format for programs running on NVIDIA GPUs. This is the
lowest-level format in which human-readable code can be written. It is one of
the formats output by `nvcc`, the
[NVIDIA CUDA Compiler Driver](/gpu-glossary/host-software/nvcc), alongside
[PTX](/gpu-glossary/device-software/parallel-thread-execution). It is converted
to device-specific binary microcodes during execution. Presumably, the
"Streaming" in "Streaming Assembler" refers to the
[Streaming Multiprocessors](/gpu-glossary/device-hardware/streaming-multiprocessor)
which the assembly language programs.

SASS is versioned and tied to a specific NVIDIA GPU
[SM architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).
See also [Compute Capability](/gpu-glossary/device-software/compute-capability).

Some exemplary instructions in SASS for the SM90a architecture of Hopper GPUs:

- `FFMA R0, R7, R0, 1.5 ;` - perform a `F`used `F`loating point `M`ultiply `A`dd
  that multiplies the contents of `R`egister 7 and `R`egister 0, adds `1.5`, and
  stores the result in `R`egister 0.
- `S2UR UR4, SR_CTAID.X ;` - copy the `X` value of the
  [Cooperative Thread Array](/gpu-glossary/device-software/cooperative-thread-array)'s
  `I`n`D`ex from its `S`pecial `R`egister to `U`niform `R`egister 4.

Even more so than for CPUs, writing this "GPU assembler" by hand is very
uncommon. Viewing compiler-generated SASS while profiling and editing high-level
[CUDA C/C++](/gpu-glossary/host-software/cuda-c) code or in-line
[PTX](/gpu-glossary/device-software/parallel-thread-execution) is
[more common](https://docs.nvidia.com/gameworks/content/developertools/desktop/ptx_sass_assembly_debugging.htm),
especially in the production of the highest-performance kernels. Viewing
[CUDA C/C++](/gpu-glossary/host-software/cuda-c), SASS, and
[PTX](/gpu-glossary/device-software/parallel-thread-execution) together is
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
