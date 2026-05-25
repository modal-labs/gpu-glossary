---
title: What is CuTe DSL?
---

CuTe DSL is a Python-based Domain-Specific Language (DSL) for writing and
dynamically compiling [kernels](/gpu-glossary/device-software/kernel) at high
performance and with high developer productivity.

CuTe DSL is part of [CUTLASS](/gpu-glossary/host-software/cutlass), a collection
of [CUDA C++](/gpu-glossary/host-software/cuda-c) templates and DSLs. Unlike
[cuBLAS](/gpu-glossary/host-software/cublas) or
[cuDNN](/gpu-glossary/host-software/cudnn), which provide ready-to-call kernels
for common operations, the CUTLASS stack provides tools for composably defining
high-performance kernels.

The core abstractions of CuTe DSL include layouts, tensors, hardware atoms, and
tiled operations. Layouts describe how data is organized in memory and across
threads. Tensors combine data pointers or iterators with layout metadata. Atoms
represent fundamental hardware operations such as matrix multiply-accumulate
(MMA) or memory copy. Tiled operations describe how atoms are applied across
[thread blocks](/gpu-glossary/device-software/thread-block) and
[warps](/gpu-glossary/device-software/warp). For the underlying details, see
[CuTe](/gpu-glossary/host-software/cute).

When launching a CuTe DSL kernel from Python, the Python program calls a
`@cute.jit` function, and that function launches a `@cute.kernel` function.

The `@cute.jit` decorator declares a JIT-compiled function that can be called
from Python or from other CuTe DSL functions. The `@cute.kernel` decorator
defines a GPU kernel function that can be launched from a `@cute.jit` function.
Python code cannot call a `@cute.kernel` function directly.

For example, let's look at a naive (unoptimized) CuTe DSL kernel for elementwise
addition of two one-dimensional tensors -- the "hello world" for GPU programming
that goes back to
[Ian Buck's Brook framework](https://graphics.stanford.edu/papers/brookgpu/brookgpu.pdf)
that preceded and inspired
[CUDA](/gpu-glossary/device-software/cuda-programming-model). You can edit this
kernel and execute it on a B200 GPU using
[this Modal Notebook](https://modal.com/notebooks/modal-labs/examples/nb-Vnwf5bQck2WSSETJUPk2UD).

```python
import cutlass.cute as cute
import torch

Tensor = cute.Tensor | torch.Tensor


@cute.kernel
def elem_add_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    block_x, _, _ = cute.arch.block_idx()
    block_dim_x, _, _ = cute.arch.block_dim()
    thread_x, _, _ = cute.arch.thread_idx()

    i = block_x * block_dim_x + thread_x

    if i < out.shape[0]:
        out[i] = a[i] + b[i]


@cute.jit
def elem_add(a: Tensor, b: Tensor, out: Tensor):
    n = out.shape[0]
    threads_per_block = 128
    blocks = (n + threads_per_block - 1) // threads_per_block

    elem_add_kernel(a, b, out).launch(
        grid=(blocks, 1, 1),
        block=(threads_per_block, 1, 1),
    )
```

The `elem_add_kernel` function is the
[kernel](/gpu-glossary/device-software/kernel). Each
[thread](/gpu-glossary/device-software/thread) computes one output element. The
global element index `i` is computed from the
[thread block](/gpu-glossary/device-software/thread-block) index, the number of
threads in the block, and the thread index inside the block:

```python
i = block_x * block_dim_x + thread_x
```

The `elem_add` function computes the number of thread blocks needed to cover the
output tensor and launches the kernel with a one-dimensional
[thread block grid](/gpu-glossary/device-software/thread-block-grid).

This example is pedagogical, not optimized. Even so, it shows a good basic
access pattern: adjacent threads read adjacent elements of `a` and `b`, then
write adjacent elements of `out`. That is the pattern needed for coalesced
accesses to [global memory](/gpu-glossary/device-software/global-memory); see
[memory coalescing](/gpu-glossary/perf/memory-coalescing).

Layout concerns are one reason why CuTe DSL is useful for high-performance
kernels. Engineering for [performance](/gpu-glossary/perf) is difficult because
kernels must be closely mapped to hardware: which threads handle which data, how
memory is accessed, how work is tiled, and which hardware operations the
generated code should use. CuTe DSL allows programmers to express these mappings
explicitly while reusing much of the same kernel code across a variety of shapes
and
[Streaming Multiprocessor architectures](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).

This may be surprising to performance-focused engineers from other domains --
how can a program written in an interpreted language like Python hope to compete
with programs written in compiled languages?

The answer is that CuTe DSL kernels are compiled, Just-In-Time (JIT). Python
source code is converted to an abstract syntax tree (AST), traced with proxy
arguments, and then compiled. Note that only a subset of Python semantics are
supported in JIT-compiled code.

At time of writing, in CUTLASS 4.x, the compilation stack passes through
[Multi-Level Intermediate Representation (MLIR)](https://mlir.llvm.org/) to the
[PTX](/gpu-glossary/device-software/parallel-thread-execution) IR to
device-specific [SASS](/gpu-glossary/device-software/streaming-assembler) before
being executed.

Consider the [FlashAttention-4](https://arxiv.org/abs/2603.05451) kernels. Our
[writeup](https://modal.com/blog/reverse-engineer-flash-attention-4) of the open
source code walks through how it uses pipelined warp specialization,
[Tensor Core](/gpu-glossary/device-hardware/tensor-core) operations, and
[Tensor Memory](/gpu-glossary/device-hardware/tensor-memory) &
[Tensor Memory Accelerator](/gpu-glossary/device-hardware/tensor-memory-accelerator)
operations to achieve state-of-the-art performance directly from CuTe DSL.

For more details on CuTe DSL, see NVIDIA's
[CuTe DSL documentation](https://docs.nvidia.com/cutlass/4.4.2/media/docs/pythonDSL/cute_dsl.html)
and
[CuTe DSL overview blog](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/).
