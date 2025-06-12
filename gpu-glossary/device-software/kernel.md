---
title: What is a Kernel?
---

![A single kernel launch corresponds to a [thread block grid](/gpu-glossary/device-software/thread-block-grid) in the [CUDA programming model](/gpu-glossary/device-software/cuda-programming-model). Modified from diagrams in NVIDIA's [CUDA Refresher: The CUDA Programming Model](https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/) and the NVIDIA [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model).](themed-image://cuda-programming-model.svg)

A kernel is the unit of CUDA code that programmers typically write and compose,
akin to a procedure or function in typical languages targeting CPUs.

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

For example, here is a naive matrix multiplication kernel for two square matrices of size N which assigns each [thread](/gpu-glossary/device-software/thread) to compute one element in the output matrix.

```cpp
__global__ void matmul_naive(float* A, float* B, float* C, int N) {
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

The kernel accesses [global memory](/gpu-glossary/device-software/global-memory) directly for all matrix elements. A more sophisticated approach makes use of the [memory hierarchy](/gpu-glossary/device-software/memory-hierarchy) by loading data into fast [shared memory](/gpu-glossary/device-software/shared-memory) that is shared among threads within a [thread block](/gpu-glossary/device-software/thread-block):

```cpp
#DEFINE TILE_WIDTH 16

__global__ void matmul_tiled(float* A, float* B, float* C, int N) {

    //declare variables in shared memory
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    float c_output = 0;
    // Loop over the A and B tiles
    for (int m = 0; m < N/TILE_WIDTH; ++m) {
        
        // Load A and B tiles into shared memory
        As[ty][tx] = A[row * N + (m*TILE_WIDTH + tx)];
        Bs[ty][tx] = B[(m*TILE_WIDTH + ty)*N + col];

        // all threads in the block must wait before anything is allowed to proceed
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; ++k) {
            c_output += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    C[row * N + col] = c_output;
}
```
