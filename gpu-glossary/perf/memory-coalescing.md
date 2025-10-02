---
title: What is Memory Coalescing?
---

Memory coalescing is a hardware technique to improve the utilization of
[memory bandwidth](/gpu-glossary/perf/memory-bandwidth) by servicing multiple
_logical_ memory reads in a single _physical_ memory access.

Memory coalescing occurs during accesses of
[global memory](/gpu-glossary/device-software/global-memory). For efficient
access of [shared memory](/gpu-glossary/device-software/shared-memory), see the
article on [bank conflict](/gpu-glossary/perf/bank-conflict).

In [CUDA](/gpu-glossary/device-hardware/cuda-device-architecture) GPUs,
[global memory](/gpu-glossary/device-software/global-memory) is backed by the
[GPU RAM](/gpu-glossary/device-hardware/gpu-ram), built with Dynamic Random
Access Memory (DRAM) technologies like GDDR or HBM. These technologies have high
[memory bandwidth](/gpu-glossary/perf/memory-bandwidth) but long access latency
(even compared to the peer technology used in CPU RAM, DDR5). DRAM access
latency is limited by the speed at which the small capacitors can charge up
their access lines, which is fundamentally limited by thermal, power, and size
constraints. Due to this high latency, if all logical memory accesses are
serviced as separate physical accesses, the GPU's
[memory bandwidth](/gpu-glossary/perf/memory-bandwidth) will not be fully
utilized.

Memory coalescing takes advantage of the internals of DRAM technology to enable
full bandwidth utilization for certain access patterns. Each time a DRAM address
is accessed, multiple consecutive addresses are fetched together in parallel in
a single clock. For a bit more detail, see Section 6.1 of
[the 4th edition of Programming Massively Parallel Processors](https://www.amazon.com/dp/0323912311);
for comprehensive detail, see Ulrich Drepper's excellent article
[_What Every Programmer Should Know About Memory_](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf).
The access and transfer of these consecutive memory locations is referred to as
a _DRAM burst_. If multiple concurrent logical accesses are serviced by a single
physical burst, the access is said to be _coalesced_. Note that a physical
access is part of a memory transaction, terminology you may see elsewhere in
descriptions of memory coalescing.

On CPUs, a similar mapping of bursts onto cache lines improves access
efficiency. As is common in GPU programming, what is automatic cache behavior in
CPUs is here programmer-managed.

That's not as hard as it could be, because DRAM bursts align elegantly with the
single-instruction, multiple thread (SIMT) execution model of
[CUDA PTX](/gpu-glossary/device-software/parallel-thread-execution). That is, in
normal execution all [threads](/gpu-glossary/device-software/thread) in a
[warp](/gpu-glossary/device-software/warp) execute the same instruction at the
same time. That makes it easy for a
[CUDA](/gpu-glossary/device-software/cuda-programming-model) programmer to write
programs with coalesced access and simple for the memory management hardware to
detect accesses that can be coalesced. Typically, a single burst can service 128
bytes – not coincidentally, enough for each of the 32
[threads](/gpu-glossary/device-software/thread) in a
[warp](/gpu-glossary/device-software/warp) to load one 32 bit float.

To demonstrate the performance impact of memory coalescing, let's consider the
following [kernel](/gpu-glossary/device-software/kernel), which reads values
from an array with a variable `stride`, or spacing between accessed elements.
With increasing stride, the number of DRAM bursts required to service the read
issued by each [warp](/gpu-glossary/device-software/warp) will increase, leading
to more physical accesses per logical access and so to reduced memory
throughput.

```cpp
__global__ void strided_read_kernel(const float* __restrict__ in,
                                    float* __restrict__ out,
                                    size_t N, int stride)
{
    const size_t t  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t T  = gridDim.x * (size_t)blockDim.x;

    float acc = 0.f;

    for (size_t j = (size_t)t * (size_t)stride; j < N; j += (size_t)T * (size_t)stride) {
        // across a warp, addresses differ by (stride * sizeof(float))
        float v = in[j]; // perfectly coalesced for stride == 1
        acc = acc * 1.000000119f + v;  // force compiler to keep the load
    }

    // do one write per thread (negligible vs reads)
    if (t < N) out[t] = acc;
}
```

When we run this kernel through a micro-benchmark on Godbolt (which you can
reproduce [here](https://godbolt.org/z/KbWhEWjcb)), we observe the expected
relationship between stride and throughput:

```
# Device: Tesla T4 (SM 75)
# N = 67108864 floats (256.0 MB), iters = 10
stride        GB/s
    1       206.0
    2       130.5
    4        68.8
    8        33.8
   16        16.8
   32        15.2
   64        13.6
  128        11.2
```

That is, adding a stride of two cuts the throughput in half, as the number of
DRAM bursts required to service each
[warp's](/gpu-glossary/device-software/warp) request doubles. Doubling the
stride to four again cuts throughput in half once more. The pattern changes once
we hit a 16x reduction in throughput at a stride of 16. Performance degrades
differently from there, presumably due to increasing visibility of other memory
subsystem components and their degraded performance from reduced locality (e.g.
on-device TLB misses).

For more best practices for global memory access, see the post
[_How to Access Global Memory Efficiently in CUDA C/C++ Kernels_](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)
on the NVIDIA Developers blog.
