# Register Pressure

Definition: Excessive per-thread register usage that limits active warps/blocks per SM (lower occupancy).

Why it matters:
- Lower occupancy reduces ability to hide latency
- Spills to local memory increase memory traffic

Key takeaways:
- Simplify kernels; reduce live ranges and temporaries
- Tune block size to balance registers vs occupancy
- Inspect with compiler reports and Nsight Compute

References:
- CUDA Best Practices: Registers and Occupancy
