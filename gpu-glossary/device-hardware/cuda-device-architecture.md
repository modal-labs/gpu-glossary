---
title: What is a CUDA Device Architecture?
---

CUDA stands for _Compute Unified Device Architecture_. Depending on the context,
"CUDA" can refer to multiple distinct things: a high-level device architecture,
a
[parallel programming model](/gpu-glossary/device-software/cuda-programming-model)
for architectures with that design, or a
[software platform](/gpu-glossary/host-software/cuda-software-platform) that
extends high-level languages like C to add that programming model.

The vision for CUDA is laid out in the
[Lindholm et al., 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf)
white paper. We highly recommend this paper, which is the original source for
many claims, diagrams, and even specific turns of phrase in NVIDIA's
documentation.

Here, we focus on the _device architecture_ part of CUDA. The core feature of a
"compute unified device architecture" is simplicity, relative to preceding GPU
architectures.

Prior to the GeForce 8800 and the Tesla data center GPUs it spawned, NVIDIA GPUs
were designed with a complex pipeline shader architecture that mapped software
shader stages onto heterogeneous, specialized hardware units. This architecture
was challenging for the software and hardware sides alike: it required software
engineers to map programs onto a fixed pipeline and forced hardware engineers to
guess the load ratios between pipeline steps.

![A diagram of a fixed-pipeline device architecture (G71). Note the presence of a separate group of processors for handling fragment and vertex shading. Adapted from [Fabien Sanglard's blog](https://fabiensanglard.net/cuda/).](themed-image://fixed-pipeline-g71.svg)

GPU devices with a unified architecture are much simpler: the hardware units are
entirely uniform, each capable of a wide array of computations. These units are
known as
[Streaming Multiprocessors (SMs)](/gpu-glossary/device-hardware/streaming-multiprocessor)
and their main subcomponents are the
[CUDA Cores](/gpu-glossary/device-hardware/cuda-core) and (for recent GPUs)
[Tensor Cores](/gpu-glossary/device-hardware/tensor-core).

![A diagram of a compute unified device architecture (G80). Note the absence of distinct processor types — all meaningful computation occurs in the identical [Streaming Multiprocessors](/gpu-glossary/device-hardware/streaming-multiprocessor) in the center of the diagram, fed with instructions for vertex, geometry, and pixel threads. Modified from [Peter Glazkowsky's 2009 white paper on the Fermi Architecture](https://www.nvidia.com/content/pdf/fermi_white_papers/p.glaskowsky_nvidia%27s_fermi-the_first_complete_gpu_architecture.pdf).](themed-image://cuda-g80.svg)

For an accessible introduction to the history and design of CUDA hardware
architectures, see [this blog post](https://fabiensanglard.net/cuda/) by Fabien
Sanglard. That blog post cites its (high-quality) sources, like NVIDIA's
[Fermi Compute Architecture white paper](https://fabiensanglard.net/cuda/Fermi_Compute_Architecture_Whitepaper.pdf).
The white paper by
[Lindholm et al. in 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf)
introducing the Tesla architecture is both well-written and thorough. The
[NVIDIA whitepaper for the Tesla P100](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf)
is less scholarly but documents the introduction of a number of features that
are critical for today's large-scale neural network workloads, like NVLink and
[on-package high-bandwidth memory](/gpu-glossary/device-hardware/gpu-ram).
