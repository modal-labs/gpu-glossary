---
title: What is peak rate?
---

Peak rate is the theoretical maximum rate at which a hardware system can complete work.

Peak rate represents the absolute upper bound of GPU performance when every execution unit operates at maximum capacity with perfect efficiency. It assumes ideal operation, where no resource constraints ([registers](https://modal.com/gpu-glossary/device-software/registers), [memory bandwidth](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21), synchronization barriers, etc.) create [bottlenecks](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21).

Peak rate is the yardstick against which all achieved performance is measured. It sets the [compute-bound](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) "roof" in a [roofline analysis](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21). It is the denominator in the utilization fraction reported in [pipe utilization](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21) metrics and the [ultimate arbiter of GPU utilization](https://modal.com/blog/gpu-utilization-guide).

Poetically, NVIDIA engineers often call it the "speed of light" — the limit on program speed imposed by physics.

Peak rate is computed directly from the fixed hardware specifications of each GPU architecture.

For example, [an NVIDIA H100 GPU](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c) with 132 SMs, each containing 128 FP32 cores, can issue 1 single precision Fused Multiply Add (`FMA`) operation, which comprises 2 floating point operations per core. That's 33,792 [instructions per clock](https://en.wikipedia.org/wiki/Instructions_per_cycle). The H100 can operate its compute subsystem clock at a maximum rate of 1980 MHz (million clocks per second) when using the FP32 cores, and so the peak rate is 66,908 billion FLOPS, or 66.9 TFLOPS.

This precisely matches the Peak FP32 TFLOPS (non-Tensor) rate advertised in [NVIDIA's H100 whitepaper](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c).
