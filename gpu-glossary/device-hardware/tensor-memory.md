---
title: What is Tensor Memory?
---

Tensor Memory is a specialized memory in the
[Streaming Multiprocessor (SM)](/gpu-glossary/device-hardware/streaming-multiprocessor)
of certain GPUs, like the [B200](https://modal.com/blog/introducing-b200-h200),
for storing the inputs and outputs of
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core).

Tensor Memory access is highly restricted. Data must be moved collectively by
four [warps](/gpu-glossary/device-software/warp) in a warpgroup, and they can
move memory only in specific patterns between Tensor Memory and
[registers](/gpu-glossary/device-software/registers), write
[shared memory](/gpu-glossary/device-software/shared-memory) to Tensor Memory,
or issue matrix-multiply-accumulate (MMA) instructions to
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core) that use Tensor Memory
for specific operands. So much for a
["compute-unified" device architecture](/gpu-glossary/device-hardware/cuda-device-architecture)!

Specifically, for a `tcgen05.mma`
[Parallel Thread eXecution](/gpu-glossary/device-software/parallel-thread-execution)
instruction computing `D += A @ B` to use Tensor Memory, the "accumulator"
matrix `D` _must_ be in Tensor Memory, the left-hand matrix `A` _may_ be in
Tensor Memory or [shared memory](/gpu-glossary/device-software/shared-memory),
and the right-hand matrix B _must_ be in
[shared memory](/gpu-glossary/device-software/shared-memory), not Tensor Memory.
This is complex, but not arbitrary -- accumulators are accessed more frequently
during matmuls than are the tiles, so they benefit more from specialized
hardware, e.g. from shorter, simpler wiring between the
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core) and the Tensor Memory.
Note that none of the matrices are in the
[registers](/gpu-glossary/device-software/registers).

Beware: Tensor Memory is not directly related to the
[Tensor Memory Accelerator](/gpu-glossary/device-hardware/tensor-memory-accelerator),
which instead loads into the
[L1 data cache](/gpu-glossary/device-hardware/l1-data-cache). Roughly speaking,
data is moved from that cache into Tensor Memory only as a result of a
[Tensor Core](/gpu-glossary/device-hardware/tensor-core) operation and then is
explicitly moved out for post-processing, e.g. the non-linearity after a matrix
multiplication in a neural network.

For details on tensor memory and patterns for its use in matrix multiplications,
see the
[_Programming Blackwell Tensor Cores with CUTLASS_ talk from GTC 2025](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72720/).
