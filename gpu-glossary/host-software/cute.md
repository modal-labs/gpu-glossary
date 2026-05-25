---
title: What is CuTe?
---

CUDA Templates (CuTe) is a header-only
[CUDA C++](/gpu-glossary/host-software/cuda-c) library within
[CUTLASS](/gpu-glossary/host-software/cutlass) for describing and manipulating
tensors of [data](/gpu-glossary/device-software/memory-hierarchy) and
[threads](/gpu-glossary/device-software/thread-hierarchy).

As the name implies, CuTe uses CUDA C++
[templates](https://en.cppreference.com/cpp/language/templates). Templates are
the C++ implementation of
[parametric polymorphism](https://bartoszmilewski.com/2014/09/22/parametricity-money-for-nothing-and-theorems-for-free/),
which you may have encountered in the form of
[generics](https://doc.rust-lang.org/rust-by-example/generics.html) in other
languages. Polymorphic functions are written once but can operate on inputs with
different types. CuTe is not to be confused with
[CuTe DSL](/gpu-glossary/host-software/cute-dsl), which exposes CuTe/CUTLASS via
a Domain-Specific Language (DSL) in Python.

At the core of CuTe's type system are `Layouts`. `Layouts` describe regular
patterns of access to CuTe `Tensors`. `Tensors` combine a `Layout` with a
pointer to [memory](/gpu-glossary/device-software/memory-hierarchy). These
`Layouts` are, critically, composable -- they form
[a category](https://arxiv.org/abs/2601.05972) with a
[rich algebra](https://arxiv.org/abs/2603.02298) and so combine both
expressiveness and structure. Note that `Layout`s are themselves composed of
`Shape` and `Stride` tuples used to describe memory extents and how to traverse
them.

CuTe uses the type system to encode key program metadata like memory
organization, strided accesses, and tiling such that the
[compiler](/gpu-glossary/host-software/nvcc) can check many aspects of
correctness and preserve invariants while applying optimizations. This allows
for very high-level metaprogramming of
[kernels](/gpu-glossary/device-software/kernel) without sacrificing performance.
For instance, the same template can be compiled into highly-optimized kernels
across several
[Streaming Multiprocessor architectures](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).
Because layouts are resolved at compile time, memory accesses carry zero
additional runtime overhead, which might otherwise kill
[performance](/gpu-glossary/perf) for
[memory-bound](/gpu-glossary/perf/memory-bound) workloads.

For additional details, see NVIDIA's
[CuTe documentation](https://docs.nvidia.com/cutlass/4.4.2/media/docs/cpp/cute/index.html).

The CuTe-based matrix transpose kernel below, based on the initial "naive"
implementation from
[this article by Colfax International](https://research.colfax-intl.com/tutorial-matrix-transpose-in-cutlass/),
demonstrates the core features and types of CuTe -- templating, shapes, layouts,
and tensors. You can run it on an H100 via
[this Modal Notebook](https://modal.com/notebooks/modal-labs/examples/nb-owEUD0kdSVeL4KeEX5sjh1).

```cpp
// one CuTe trick: transpose a row-major matrix just using Layouts
template <typename T>
__global__ void transpose_kernel(const T* __restrict__ d_S,
                                 T* __restrict__ d_D,
                                 int M, int N)
{
    // define the Shape of tiles worked on by thread blocks
    using b = Int<32>;
    auto block_shape = make_shape(b{}, b{});

    // define the Shape of input/output Tensors
    auto tensor_shape = make_shape(M, N);

    // define the Layout of the input and output Tensors in global memory
    auto gmemLayoutS  = make_layout(tensor_shape, GenRowMajor{}); // input:  row-major
    auto gmemLayoutDT = make_layout(tensor_shape, GenColMajor{}); // output: col-major

    // construct the Tensors
    auto tensor_S  = make_tensor(make_gmem_ptr(d_S), gmemLayoutS);
    auto tensor_DT = make_tensor(make_gmem_ptr(d_D), gmemLayoutDT);

    // define a tile-ing of the Tensors (as a "Tensor of Tensors")
    auto tiled_tensor_S  = tiled_divide(tensor_S,  block_shape);
    auto tiled_tensor_DT = tiled_divide(tensor_DT, block_shape);

    // pull out the tiles this thread block will be working on
    auto tile_S  = tiled_tensor_S (make_coord(_, _), blockIdx.x, blockIdx.y);
    auto tile_DT = tiled_tensor_DT(make_coord(_, _), blockIdx.x, blockIdx.y);

    // create a Layout for threads in the thread block
    auto thr_layout = make_layout(
        make_shape(Int<8>{}, Int<32>{}),
        GenRowMajor{}
    );

    // pull out the tile this thread will work on
    auto thr_tile_S  = local_partition(tile_S,  thr_layout, threadIdx.x);
    auto thr_tile_DT = local_partition(tile_DT, thr_layout, threadIdx.x);

    // define a "Tensor" in register memory
    auto rmem = make_tensor_like<T>(thr_tile_S);

    // copy tile into registers
    copy(thr_tile_S, rmem);
    // copy tile out of registers as though it were column-major
    copy(rmem, thr_tile_DT);
}
```
