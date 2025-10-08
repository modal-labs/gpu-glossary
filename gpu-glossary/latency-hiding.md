# Latency Hiding

Definition: Overlapping long-latency operations (e.g., memory, H2D/D2H) with other ready work using many warps and streams.

Why it matters:
- Keeps SMs busy despite memory and IO latency
- Improves throughput without changing core math

Key takeaways:
- Ensure enough parallelism (active warps) per SM
- Use asynchronous copies and multiple streams
- Tile to reuse data in shared memory and caches

References:
- CUDA Streams and Events Guide
- Nsight Systems timeline analysis
