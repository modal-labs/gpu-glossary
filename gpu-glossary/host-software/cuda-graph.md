---
title: What is a CUDA Graph?
---

A CUDA Graph is a graph of [kernel](/gpu-glossary/device-software/kernel)
launches and other work that can be submitted by the host to the device all at
once.

The primary use case for CUDA Graphs is reducing
[overhead](/gpu-glossary/perf/overhead) from host identification, configuration,
and submission of large numbers of
[kernels](/gpu-glossary/device-software/kernel) in short periods. Each launch
takes on the order of microseconds, so if hundreds of
[kernels](/gpu-glossary/device-software/kernel) need to be launched in
milliseconds, this overhead can be very noticeable. This is commonly the case
for
[low-latency LLM inference](https://modal.com/docs/guide/high-performance-llm-inference).

CUDA Graphs are most commonly created via the stream capture API in the
[CUDA Runtime](/gpu-glossary/host-software/cuda-runtime-api), which allows all
of the operations that occur on a single CUDA stream to be captured and then
later replayed, like

```cpp
// capture
cudaStreamBeginCapture(stream);
kernelGemm<<<{32, 20},64,19200,stream>>>(a, b, c);
kernelEpilogue<<<{256,2},{8,32},0,stream>>>(c, c);
cudaStreamEndCapture(stream, &graph);

// launch
cudaGraphInstantiate(&graphExec, graph, flags);
cudaGraphLaunch(graphExec, stream);
```

The [CUDA Runtime](/gpu-glossary/host-software/cuda-runtime-api) interface to
CUDA Graphs is documented by NVIDIA
[here](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

This API is wrapped by PyTorch, e.g. via the `torch.cuda.graph` context manager,
which is how CUDA Graphs are generally captured for neural network training and
inference.

Below is a sample CUDA Graph, captured from a B200 GPU executing a
`torch.Linear` layer:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                          NODE 0: KERNEL                           │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │  ID:         0 (topoId: 1)                                        │  │
│  │  Kernel:     cutlass3x_sm100_simt_sgemm_f32_f32_f32_f32_f32_      │  │
│  │              64x32x16_1x1x1_3_tnn_align1_bias_f32_relu            │  │
│  │              <<<{32,20},64,19200>>>                               │  │
│  │  Node handle: 0x0000564604539520                                  │  │
│  │  Func handle: 0x0000564603AFCC00                                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              │                                          │
│                              ▼                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                          NODE 1: KERNEL                           │  │
│  ├───────────────────────────────────────────────────────────────────┤  │
│  │  ID:         1 (topoId: 0)                                        │  │
│  │  Kernel:     _ZN8cublasLt8epilogue4impl12globalKernelILi8E...     │  │
│  │              <<<{256,2},{8,32},0>>>                               │  │
│  │  Node handle: 0x0000564604539C88                                  │  │
│  │  Func handle: 0x00005646044770F0                                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

Notice that [kernels](/gpu-glossary/device-software/kernel) are identified by
pointers, e.g. `0x564603AFCC00`. Inputs and outputs are also defined by
pointers. These and other references to device resources prevent serialization
of CUDA Graphs and make them non-portable, outside of fully
[checkpointing and then restoring the host and device memory](https://modal.com/docs/guide/memory-snapshots).
