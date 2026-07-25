---
title: What is a Kernel Launch Configuration?
---

Every [kernel](/gpu-glossary/device-software/kernel) launch carries a
configuration that fixes the shape of the
[thread block grid](/gpu-glossary/device-software/thread-block-grid) that launch
creates: how many [threads](/gpu-glossary/device-software/thread) make up each
[thread block](/gpu-glossary/device-software/thread-block), how many blocks make
up the grid, and how much
[shared memory](/gpu-glossary/device-software/shared-memory) each block is
allocated. NVIDIA's documentation calls this the
[execution configuration](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#kernel-configuration)
of the launch.

The configuration belongs to the launch, not to the
[kernel](/gpu-glossary/device-software/kernel) — the same compiled
[kernel](/gpu-glossary/device-software/kernel) can be launched with a different
configuration every time it is called. It is where the abstract
[thread hierarchy](/gpu-glossary/device-software/thread-hierarchy) of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model)
becomes concrete numbers.

In [CUDA C++](/gpu-glossary/host-software/cuda-c), the configuration goes
between the [kernel](/gpu-glossary/device-software/kernel)'s name and its
argument list, inside triple chevrons:

```cpp
// 4 blocks of 256 threads, 2 KB of dynamic smem per block, on `stream`
vecAdd<<<4, 256, 2048, stream>>>(A, B, C, N);
```

The four arguments are

- the **grid dimensions**, of type `dim3`, whose product is the number of
  [thread blocks](/gpu-glossary/device-software/thread-block) launched,
- the **block dimensions**, also a `dim3`, whose product is the number of
  [threads](/gpu-glossary/device-software/thread) per
  [block](/gpu-glossary/device-software/thread-block),
- the **dynamic [shared memory](/gpu-glossary/device-software/shared-memory)**
  size in bytes, optional and zero by default, allocated per
  [block](/gpu-glossary/device-software/thread-block) on top of whatever the
  [kernel](/gpu-glossary/device-software/kernel) declares statically and read
  through its `extern __shared__` arrays, and
- the **stream** the launch is submitted to, optional and defaulting to the null
  stream.

A `dim3` is a triple of unsigned integers, `x`, `y`, and `z`, each defaulting to
one, so the plain integers above describe a one-dimensional grid of
one-dimensional blocks. Both grids and blocks can be up to three-dimensional,
which is a convenience for indexing multi-dimensional data and has no effect on
performance.

[Threads](/gpu-glossary/device-software/thread) read the configuration back out
of four built-in variables: `gridDim` and `blockDim`, which hold the dimensions
the [kernel](/gpu-glossary/device-software/kernel) was launched with, plus
`blockIdx` and `threadIdx`, which hold the zero-based coordinates of this
[thread](/gpu-glossary/device-software/thread) within its
[block](/gpu-glossary/device-software/thread-block) and of its
[block](/gpu-glossary/device-software/thread-block) within the
[grid](/gpu-glossary/device-software/thread-block-grid). Those coordinates are
the only thing that distinguishes one
[thread](/gpu-glossary/device-software/thread)'s execution of a
[kernel](/gpu-glossary/device-software/kernel) from another's, so they are how
work gets assigned:

```cpp
__global__ void vecAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's element

    if (i < N) {  // the grid is generally a bit bigger than the problem
        C[i] = A[i] + B[i];
    }
}
```

Because the block size rarely divides the problem size, host code rounds the
grid dimensions up — `(N + threads - 1) / threads`, or `cuda::ceil_div` from the
CUDA Core Compute Libraries — and the
[kernel](/gpu-glossary/device-software/kernel) guards against out-of-bounds
indices, as above. Idle [threads](/gpu-glossary/device-software/thread) in an
otherwise busy [block](/gpu-glossary/device-software/thread-block) are cheap;
entire [blocks](/gpu-glossary/device-software/thread-block) of idle
[threads](/gpu-glossary/device-software/thread) are not.

Note that the first argument is a count of
[blocks](/gpu-glossary/device-software/thread-block), not of
[threads](/gpu-glossary/device-software/thread). Sizing it from the problem
rather than from the device is what lets the same launch scale onto GPUs with
more
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor),
as in the
[wave scheduling diagram here](/gpu-glossary/device-software/cuda-programming-model).

The configuration is bounded on both ends of the
[hierarchy](/gpu-glossary/device-software/thread-hierarchy). Because all
[threads](/gpu-glossary/device-software/thread) of a
[block](/gpu-glossary/device-software/thread-block) are resident on one
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor) and share its
resources, [blocks](/gpu-glossary/device-software/thread-block) are small: at
most 1024 [threads](/gpu-glossary/device-software/thread) on current devices,
with per-dimension maxima of 1024, 1024, and 64.
[Grids](/gpu-glossary/device-software/thread-block-grid) are large: up to
`2^31 - 1` [blocks](/gpu-glossary/device-software/thread-block) in `x` and
65,535 in each of `y` and `z`. The exact values are tabulated by
[compute capability](/gpu-glossary/device-software/compute-capability) in the
[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).

These limits are enforced at launch, not at
[compile time](/gpu-glossary/host-software/nvcc). A configuration that exceeds
them comes back as an "invalid configuration argument" error, and one that fits
but demands more [registers](/gpu-glossary/device-software/registers) or
[shared memory](/gpu-glossary/device-software/shared-memory) per
[block](/gpu-glossary/device-software/thread-block) than an
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor) can supply comes
back as "too many resources requested for launch".

A [kernel](/gpu-glossary/device-software/kernel) can also constrain its own
configuration with `__launch_bounds__`, promising the
[compiler](/gpu-glossary/host-software/nvcc) an upper bound on the block size it
will ever be launched with. The [compiler](/gpu-glossary/host-software/nvcc)
uses that promise to budget [registers](/gpu-glossary/device-software/registers)
per [thread](/gpu-glossary/device-software/thread), and launches that violate it
fail.

Within those limits, choosing a configuration is largely empirical, but some
rules of thumb apply, drawn from the
[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#thread-and-block-heuristics):

- Make the block size a multiple of the
  [warp](/gpu-glossary/device-software/warp) size, 32 on current devices. A
  partly-filled [warp](/gpu-glossary/device-software/warp) still occupies a full
  [warp](/gpu-glossary/device-software/warp) slot on an
  [SM](/gpu-glossary/device-hardware/streaming-multiprocessor). Between 128 and
  256 [threads](/gpu-glossary/device-software/thread) per
  [block](/gpu-glossary/device-software/thread-block) is a reasonable place to
  start.
- Launch many more [blocks](/gpu-glossary/device-software/thread-block) than
  there are [SMs](/gpu-glossary/device-hardware/streaming-multiprocessor).
  [Blocks](/gpu-glossary/device-software/thread-block) become resident on
  [SMs](/gpu-glossary/device-hardware/streaming-multiprocessor) in "waves" of
  concurrently-executing [blocks](/gpu-glossary/device-software/thread-block),
  and a grid that isn't a multiple of the wave size leaves the final wave partly
  empty — a "tail effect" that only matters when there are few waves to amortize
  it over.
- Remember that block size and dynamic
  [shared memory](/gpu-glossary/device-software/shared-memory) size together
  determine how many [blocks](/gpu-glossary/device-software/thread-block) fit on
  an [SM](/gpu-glossary/device-hardware/streaming-multiprocessor), and so set
  the [theoretical occupancy](/gpu-glossary/perf/occupancy) of the launch. But
  [occupancy](/gpu-glossary/perf/occupancy) is not itself the target: past the
  point where there are enough [warps](/gpu-glossary/device-software/warp) to
  [hide latency](/gpu-glossary/perf/latency-hiding), raising it just squeezes
  the [registers](/gpu-glossary/device-software/registers) available to each
  [thread](/gpu-glossary/device-software/thread).

For the block size that maximizes [occupancy](/gpu-glossary/perf/occupancy) for
a given [kernel](/gpu-glossary/device-software/kernel), the
[CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api) will compute
one: `cudaOccupancyMaxPotentialBlockSize`.

The configuration doesn't have to be derived from the problem size at all. A
[kernel](/gpu-glossary/device-software/kernel) written as a
["grid-stride loop"](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/),
in which each [thread](/gpu-glossary/device-software/thread) walks its input in
strides of `gridDim.x * blockDim.x` — as in the
[memory coalescing](/gpu-glossary/perf/memory-coalescing) article — works for
any grid size, so the grid can instead be sized to fill the device once.

The triple chevrons themselves are syntactic sugar.
[`nvcc`](/gpu-glossary/host-software/nvcc) lowers them to a call to
`cudaLaunchKernel` in the
[CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api), which in turn
calls `cuLaunchKernel` in the
[CUDA Driver API](/gpu-glossary/host-software/cuda-driver-api). The
configuration travels with the launch at runtime rather than living inside the
compiled [kernel](/gpu-glossary/device-software/kernel): in
[PTX](/gpu-glossary/device-software/parallel-thread-execution) and
[SASS](/gpu-glossary/device-software/streaming-assembler), the built-in
variables above are reads of the special registers `%tid`, `%ntid`, `%ctaid`,
and `%nctaid`, which the hardware populates per
[thread](/gpu-glossary/device-software/thread) as
[CTAs](/gpu-glossary/device-software/cooperative-thread-array) are scheduled.
`__launch_bounds__`, by contrast, is compiled in, as the `.maxntid` directive.

Configurations too rich for four arguments go through `cudaLaunchKernelEx`,
which takes a configuration struct plus a list of launch attributes. That is how
the dimensions of a
[thread block cluster](/gpu-glossary/device-hardware/graphics-processing-cluster)
are set on
[compute capability](/gpu-glossary/device-software/compute-capability) 9.0 and
later devices, if they weren't fixed at
[compile time](/gpu-glossary/host-software/nvcc) with the `__cluster_dims__`
attribute. Even for a clustered launch, the grid dimensions still count
[thread blocks](/gpu-glossary/device-software/thread-block) — they must just be
a multiple of the cluster dimensions.

Launches are asynchronous with respect to the host and cost on the order of
microseconds of [overhead](/gpu-glossary/perf/overhead) each, so
[kernels](/gpu-glossary/device-software/kernel) with configurations too small to
occupy the GPU for long are frequently better fused, or submitted together as a
[CUDA Graph](/gpu-glossary/host-software/cuda-graph).
