---
title: What is a Streaming Multiprocessor Architecture?
---

[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor)
are versioned with a particular "architecture" that defines their compatibility
with
[Streaming Assembler (SASS)](/gpu-glossary/device-software/streaming-assembler)
code.

![A streaming multiprocessor with the "Hopper" SM90 architecture. Modified from NVIDIA's [H100 white paper](https://modal-cdn.com/gpu-glossary/gtc22-whitepaper-hopper.pdf).](themed-image://gh100-sm.svg)

![A streaming multiprocessor with the original "Tesla" SM architecture. Modified from [Fabien Sanglard's blog](https://fabiensanglard.net/cuda)](themed-image://tesla-sm.svg)

Most [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) versions have
two components: a major version and a minor version.

The major version is _almost_ synonymous with GPU architecture family. For
example, all SM versions `6.x` are of the Pascal Architecture. Some NVIDIA
documentation even
[makes this claim directly](https://docs.nvidia.com/cuda/ptx-writers-guide-to-interoperability/index.html).
But, as an example, Ada GPUs have
[SM](/gpu-glossary/device-hardware/streaming-multiprocessor) architecture
version `8.9`, the same major version as Ampere GPUs.

Target [SM](/gpu-glossary/device-hardware/streaming-multiprocessor) versions for
[SASS](/gpu-glossary/device-software/streaming-assembler) compilation can be
specified when invoking `nvcc`, the
[NVIDIA CUDA Compiler Driver](/gpu-glossary/host-software/nvcc). Compatibility
across major versions is explicitly not guaranteed. For more on compatibility
across minor versions, see the
[documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)
for [nvcc](/gpu-glossary/host-software/nvcc).
