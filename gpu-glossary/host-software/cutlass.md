---
title: What is CUTLASS?
---

CUDA Templates for Linear Algebra Subroutines and Solvers (CUTLASS) is a library
of abstractions for implementing high-performance linear algebra in
[CUDA](/gpu-glossary/device-software/cuda-programming-model)
[kernels](/gpu-glossary/device-software/kernel).

Like [cuBLAS](/gpu-glossary/host-software/cublas), CUTLASS is named in reference
to the
[Basic Linear Algebra Subprograms (BLAS)](https://netlib.org/blas/blast-forum/)
standard for low-level routines for linear algebraic computations. Unlike
cuBLAS, CUTLASS is a toolkit for constructing kernels, rather than a library of
ready-to-call routines. CUTLASS is primarily associated with the third level of
the BLAS hierarchy, general matrix multiplications ("GEMMs").

As the name suggests, CUTLASS includes a collection of
[CUDA C++](/gpu-glossary/host-software/cuda-c) template abstractions.
[Templates](https://en.cppreference.com/cpp/language/templates) are the C++
implementation of
[parametric polymorphism](https://bartoszmilewski.com/2014/09/22/parametricity-money-for-nothing-and-theorems-for-free/),
which you may have encountered in the form of
[generics](https://doc.rust-lang.org/rust-by-example/generics.html) in other
languages. Polymorphic functions are written once but can operate on inputs with
different types.

The core of modern CUTLASS is the [CuTe](/gpu-glossary/host-software/cute)
library, which defines `Layout` and `Tensor` types for composably describing and
manipulating tensors of [data](/gpu-glossary/device-software/memory-hierarchy)
and [threads](/gpu-glossary/device-software/thread-hierarchy). It is not to be
confused with [CuTe DSL](/gpu-glossary/host-software/cute-dsl), which exposes
CuTe/CUTLASS templates via a Domain-Specific Language (DSL) in Python.

Atop CuTe, CUTLASS exposes a header-only CUDA C++ library that operates at three
levels: the whole `device`, a single
[`kernel`](/gpu-glossary/device-software/kernel), or a `collective` of
[threads](/gpu-glossary/device-software/thread) (typically a
[thread block](/gpu-glossary/device-software/thread-block)). At the `collective`
layer, matrix-matrix multiplications are typically split into "mainloops" and
"epilogues". Mainloops express the core algorithm, like tiling strategies.
Epilogues describe post-processing steps, like the application of scaling
factors or scalar non-linearities (popular in neural networks).

CUTLASS is very commonly used to write some of the highest-performing kernels,
especially matrix-matrix multiplications on hardware from more recent
[Streaming Multiprocessor architectures](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).
These kernels require careful programming of the
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core) to achieve anything
like peak [performance](/gpu-glossary/perf).

CUTLASS is
[open source and available on GitHub](https://github.com/nvidia/cutlass). The
library also includes many implementations of high-performance open-source
kernels using CUTLASS, which are regularly used as references elsewhere in
open-source kernel development. We can highly recommend the
[popular tutorials by Jay Shah of Colfax International](https://research.colfax-intl.com/),
which explain in detail how the key components of CUTLASS are used to achieve
maximum performance. Note, however, that like most C++ template metaprogramming,
CUTLASS is not for the faint of heart!
