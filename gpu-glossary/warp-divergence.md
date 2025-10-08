# Warp Divergence

Definition: When threads within a warp take different control-flow paths, forcing serial execution of each path.

Why it matters:
- Reduces parallel efficiency; increases execution time
- Can negate benefits of high occupancy

Key takeaways:
- Minimize data-dependent branches inside hot kernels
- Use predication or warp-level ops where possible
- Restructure algorithms to keep warps converged

References:
- CUDA Programming Guide: Control Flow
