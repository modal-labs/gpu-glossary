---
title: What is a Warpgroup?
---

A warpgroup is a set of four contiguous
[warps](/gpu-glossary/device-software/warp) such that the warp-rank of the first
warp is a multiple of 4.

Upon dispatching a warpgroup-level instruction, we coordinate 128
[threads](/gpu-glossary/device-software/thread) -- 4 warps per warpgroup × 32
threads per warp. Operating at a larger granularity removes the need for
explicit inter-warp synchronization and allows work to be performed on larger
problem sizes per instruction, especially larger matrix multiplications. Larger
matrix multiplications more readily saturate the massive
[arithmetic bandwidth](/gpu-glossary/perf/arithmetic-bandwidth) of the
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core) of recent data center
GPUs.

Warpgroups were introduced in NVIDIA's Hopper
[Streaming Multiprocessor architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture),
where they are used to support warpgroup-level matrix multiplication, like
`wgmma.mma_async`. See
[this blog post from Colfax](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)
for a deep dive. Warpgroups feature prominently in the organization of pipeline
components of high-performance Hopper and Blackwell
[kernels](/gpu-glossary/device-software/kernel), like
[Flash Attention 4](https://modal.com/blog/reverse-engineer-flash-attention-4).

In
[Parallel Thread Execution (PTX)](/gpu-glossary/device-software/parallel-thread-execution)
IR, the warp-rank of a warp is:

```cpp
int linearIdx = (%tid.x + %tid.y * %ntid.x  + %tid.z * %ntid.x * %ntid.y);
int warpRank = linearIdx / 32;
```

where `tid` is the thread index, accessed via special PTX
[registers](/gpu-glossary/device-software/registers).

So the valid warpgroups for an 8-warp dispatch are:

- **Warpgroup 0**: warp-ranks 0, 1, 2, and 3
- **Warpgroup 1**: warp-ranks 4, 5, 6, and 7.

To our knowledge, the purpose of the warp-rank alignment restriction is not
documented. But
[Streaming Multiprocessors](/gpu-glossary/device-hardware/streaming-multiprocessor)
for recent data center GPUs appear to contain four (unnamed) subunits, each with
their own [Warp Scheduler](/gpu-glossary/device-hardware/warp-scheduler) and
Tensor Core.
