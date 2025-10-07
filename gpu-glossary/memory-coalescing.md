# Memory Coalescing

Definition: Align and group warp memory accesses so threads hit a minimal number of wide transactions.

Why it matters:
- Maximizes effective bandwidth, lowers latency
- Crucial for memory-bound kernels

Key takeaways:
- Consecutive threads -> consecutive addresses
- Prefer contiguous layouts; avoid large strides
- Align starting addresses

References:
- CUDA Best Practices Guide
- Nsight Compute memory analysis
