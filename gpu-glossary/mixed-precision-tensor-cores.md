# Mixed Precision and Tensor Cores

Definition: Use reduced precision (FP16/BF16/INT8) with proper scaling/accumulation; Tensor Cores accelerate MMA on such formats.

Why it matters:
- Higher throughput on supported GPUs
- Lower memory traffic; larger batches possible

Key takeaways:
- Prefer cuBLAS/cuDNN paths using Tensor Cores
- Validate numerical stability (loss scaling, INT8 calibration)
- Match tiling/layout to MMA requirements

References:
- NVIDIA Mixed Precision Training docs
- cuBLASLt Tensor Core GEMM
