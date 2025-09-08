---
title: What is cuBLAS?
---

cuBLAS (CUDA Basic Linear Algebra Subroutines) is NVIDIA's high-performance
implementation of the
[Basic Linear Algebra Subprograms (BLAS)](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)
standard. It is a proprietary software library that provides highly optimized
[kernels](/gpu-glossary/device-software/kernel) for common linear algebra
operations.

Instead of writing and optimizing common operations like matrix multiplication
from scratch, developers can call cuBLAS functions from their host code. The
library contains a wide array of kernels, each fine-tuned for specific data
types (e.g. FP32, FP16), matrix sizes, and
[Streaming Multiprocessor (SM) architectures](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture).
At runtime, cuBLAS uses (unknown) internal heuristics to select the most
performant kernel and its optimal launch parameters. As a result, cuBLAS is the
foundation for most [high-performance](/gpu-glossary/perf) numerical computing
on NVIDIA GPUs and is used extensively by deep learning frameworks like PyTorch
to accelerate their core operations, along with more specialized
[kernel](/gpu-glossary/device-software/kernel) libraries like
[cuDNN](/gpu-glossary/host-software/cudnn).

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
when compiling with [nvcc](/gpu-glossary/host-software/nvcc)). Its functions are
exposed via the `cublas_v2.h` header.

For more information on cuBLAS, see the
[official cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/).
