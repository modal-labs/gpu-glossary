# Roofline Model

Definition: A visual performance model that bounds attainable FLOP/s by either the compute peak or memory bandwidth, given arithmetic intensity (FLOPs/byte).

Why it matters:
- Identifies compute-bound vs memory-bound regimes
- Guides optimization focus: math vs memory
- Sets realistic performance targets

Key takeaways:
- Raise arithmetic intensity via reuse/fusion
- Compute-bound: use Tensor Cores, mixed precision
- Memory-bound: improve coalescing, tiling, shared memory

References:
- NVIDIA Nsight Compute roofline
- LBL Roofline paper
