---
title: What is a Kernel?
---

![A single kernel launch corresponds to a [thread block grid](/gpu-glossary/device-software/thread-block-grid) in the [CUDA programming model](/gpu-glossary/device-software/cuda-programming-model). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](themed-image://cuda-programming-model.svg)

A kernel is the unit of CUDA code that programmers typically write and compose,
akin to a procedure or function in languages targeting CPUs.

Unlike procedures, a kernel is called ("launched") once and returns once, but is
executed many times, once each by a number of
[threads](/gpu-glossary/device-software/thread). These executions are generally
concurrent (their execution order is non-deterministic) and parallel (they occur
simultaneously on different execution units).

The collection of all threads executing a kernel is organized as a kernel grid —
aka a [thread block grid](/gpu-glossary/device-software/thread-block-grid), the
highest level of the
[CUDA programming model](/gpu-glossary/device-software/cuda-programming-model)'s
thread hierarchy. A kernel grid executes across multiple
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor)
and so operates at the scale of the entire GPU. The matching level of the
[memory hierarchy](/gpu-glossary/device-software/memory-hierarchy) is the
[global memory](/gpu-glossary/device-software/global-memory).

In [CUDA C++](/gpu-glossary/host-software/cuda-c), kernels are passed pointers
to [global memory](/gpu-glossary/device-software/global-memory) on the device
when they are invoked by the host and return nothing — they just mutate memory.

To give a flavor for CUDA kernel programming, let's walk through two
implementations of the "hello world" of CUDA kernels: matrix multiplication of
two square matrices, `A` and `B`. The two implementations will differ in how
they map the textbook matrix multiplication algorithm onto the thread hierarchy
and [memory hierarchy](/gpu-glossary/device-software/memory-hierarchy).

In the simplest implementation, inspired by the first matmul kernel in
[Programming Massively Parallel Processors](https://www.amazon.com/dp/0323912311)
(4th edition, Figure 3.11), each [thread](/gpu-glossary/device-software/thread)
does all of the work to compute one element of the output matrix -- loading in
turn each element of a particular `row` of `A` and a particular `col`umn of `B`
into [registers](/gpu-glossary/device-software/registers), multiplying the
paired elements, summing the results, and placing the sum back in
[global memory](/gpu-glossary/device-software/global-memory).

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

In this kernel, each [thread](/gpu-glossary/device-software/thread) does one
floating point operation (FLOP) per read from
[global memory](/gpu-glossary/device-software/global-memory): a multiply and an
add; a load from `A` and a load from `B`. You'll never
[use the whole GPU](https://modal.com/blog/gpu-utilization-guide) that way,
since the bandwidth of the [CUDA Cores](/gpu-glossary/device-hardware/cuda-core)
in FLOPs/s is much higher than the bandwidth between the
[GPU RAM](/gpu-glossary/device-hardware/gpu-ram) and the
[SMs](/gpu-glossary/device-hardware/streaming-multiprocessor).

We can increase the ratio of FLOPs to reads by more carefully mapping the work
in this algorithm onto the thread hierarchy and
[memory hierarchy](/gpu-glossary/device-software/memory-hierarchy). In the
"tiled" matmul kernel below, inspired by that in Figure 5.9 of the 4th edition
of
[Programming Massively Parallel Processors](https://www.amazon.com/dp/0323912311),
we map the loading of submatrices of `A` and `B` and the computation of
submatrices of `C` onto
[shared memory](/gpu-glossary/device-software/shared-memory) and
[thread blocks](/gpu-glossary/device-software/thread-block) respectively.

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
[CUDA Cores](/gpu-glossary/device-hardware/cuda-core). The absolute fastest
matrix multiplication kernels run instead on
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core).
