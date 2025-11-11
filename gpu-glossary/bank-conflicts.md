# Bank Conflicts (Shared Memory)

Definition: Multiple threads in a warp accessing different addresses within the same shared-memory bank, causing serialized transactions.

Why it matters:
- Increases shared-memory access latency
- Can bottleneck otherwise compute-efficient kernels

Key takeaways:
- Pad shared-memory tiles to avoid bank aliasing
- Prefer access patterns mapping one thread per bank
- Validate with profiler bank-conflict metrics

References:
- CUDA Programming Guide: Shared Memory Banks
