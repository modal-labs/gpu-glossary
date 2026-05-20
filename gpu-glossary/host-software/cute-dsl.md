---
title: What is CuTe DSL?
---

CuTe DSL is a Python-based domain-specific language for writing and dynamically
compiling [GPU kernels](/gpu-glossary/device-software/kernel) for NVIDIA GPUs.

CUTLASS 3.0 introduced CuTe as a C++ CUDA template library for describing and
manipulating tensors of threads and data. CUTLASS 4.0.0 introduced CuTe DSL, a
Python DSL centered around CuTe's abstractions.

CuTe DSL exposes layouts, tensors, hardware atoms, and tiled operations. Layouts
describe how data is organized in memory and across threads. Tensors combine
data pointers or iterators with layout metadata. Atoms represent fundamental
hardware operations such as matrix multiply-accumulate (MMA) or memory copy.
Tiled operations describe how atoms are applied across thread blocks and warps.

This makes CuTe DSL different from libraries like
[cuBLAS](/gpu-glossary/host-software/cublas) or
[cuDNN](/gpu-glossary/host-software/cudnn). Those libraries provide optimized
[kernels](/gpu-glossary/device-software/kernel) for common operations. CuTe DSL
is a language for writing or generating custom kernels.

When launching a CuTe DSL kernel from Python, the Python program calls a
`@cute.jit` function, and that function launches a `@cute.kernel` function.

The `@cute.jit` decorator declares a JIT-compiled function that can be called
from Python or from other CuTe DSL functions. The `@cute.kernel` decorator
defines a GPU kernel function that can be launched from a `@cute.jit` function.
Python code cannot call a `@cute.kernel` function directly.

For example, this CuTe DSL kernel performs a naive elementwise addition of two
one-dimensional tensors:

```python
import cutlass.cute as cute

@cute.kernel
def elem_add_kernel(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
    block_x, _, _ = cute.arch.block_idx()
    block_dim_x, _, _ = cute.arch.block_dim()
    thread_x, _, _ = cute.arch.thread_idx()

    i = block_x * block_dim_x + thread_x

    if i < out.shape[0]:
        out[i] = a[i] + b[i]

@cute.jit
def elem_add(a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
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

A `cute.Tensor` combines an engine with a layout: the engine is a data pointer
or iterator over storage, while the layout describes how logical coordinates map
to offsets or execution coordinates. In simple vector addition, the layout is
mostly hidden. In tiled matrix multiplication, copy-heavy kernels, and
[Tensor Core](/gpu-glossary/device-hardware/tensor-core) kernels, layout is
often the central problem.

For a deeper treatment of CuTe's layout model, see Colfax Research's
[note on the algebra of CuTe layouts](https://research.colfax-intl.com/a-note-on-the-algebra-of-cute-layouts/)
and
[categorical foundations for CuTe layouts](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/).
For the underlying formal treatment, see
[Cris Cecka's paper on CuTe layout representation and algebra](https://arxiv.org/abs/2603.02298).

That is why CuTe DSL is useful for high-performance kernels. Many GPU kernels
are not difficult because of the scalar arithmetic they perform. They are
difficult because of the mapping to hardware: which threads handle which data,
how memory is accessed, how work is tiled, and which hardware operations the
generated code should use.

A recent example is FlashAttention-4. The
[FlashAttention-4 paper](https://arxiv.org/abs/2603.05451) reports that its
Blackwell attention kernel is implemented entirely in CuTe-DSL embedded in
Python, with 20-30x faster compile times than traditional C++ template-based
implementations while preserving low-level expressivity. Modal's
[reverse-engineering writeup](https://modal.com/blog/reverse-engineer-flash-attention-4)
walks through how that kernel uses pipelining,
[Tensor Core](/gpu-glossary/device-hardware/tensor-core),
[Tensor Memory](/gpu-glossary/device-hardware/tensor-memory), and explicit
memory movement.

CuTe DSL makes this staging explicit. Python source code does not execute
directly on the GPU. Python source flows through AST preprocessing and
interpreter-driven tracing to produce an intermediate representation, which is
then lowered and compiled to device code.

In short:

```text
Python source with @cute.jit and @cute.kernel
        |
AST preprocessing of Python control flow
        |
Interpreter driven tracing with proxy tensor arguments
        |
Intermediate representation compiled using MLIR infrastructure
        |
        PTX
        |
CUBIN containing target-specific device code
        |
Loaded and launched as a GPU kernel
```

For framework integration, CuTe DSL can also compile JIT functions through
[TVM FFI](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html),
an optional calling path that lets compiled functions accept DLPack-compatible
objects such as `torch.Tensor` directly and reduce CPU-side invocation overhead
for eager workloads.

[PTX](/gpu-glossary/device-software/parallel-thread-execution) is an
intermediate representation for NVIDIA GPU code. A CUBIN is a CUDA device-code
binary for a target GPU architecture. When compiled device code is disassembled,
the architecture-specific instruction stream is seen as
[SASS](/gpu-glossary/device-software/streaming-assembler).

This places CuTe DSL in the host-software part of the GPU stack, near
[CUDA C++](/gpu-glossary/host-software/cuda-c),
[nvcc](/gpu-glossary/host-software/nvcc), and the
[NVIDIA Runtime Compiler](/gpu-glossary/host-software/nvrtc): it is a
host-facing programming and compilation interface that produces code for the
device.

For more details, see NVIDIA's
[CuTe DSL documentation](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html)
and
[CuTe DSL overview blog](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/).
