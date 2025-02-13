---
title: What is the CUDA Software Platform?
---

CUDA stands for _Compute Unified Device Architecture_. Depending on the context,
"CUDA" can refer to multiple distinct things: a high-level device architecture,
a parallel programming model for architectures with that design, or a software
platform that extends high-level languages like C to add that programming model.

The vision for CUDA is laid out in the
[Lindholm et al., 2008](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf)
white paper. We highly recommend this paper, which is the original source for
many claims, diagrams, and even specific turns of phrase in NVIDIA's
documentation.

Here, we focus on the CUDA _software platform_.

The CUDA software platform is a collection of software for developing CUDA
programs. Though CUDA software platforms exist for other languages, like
FORTRAN, we will focus on the dominant
[CUDA C++](/gpu-glossary/host-software/cuda-c) version.

This platform can be roughly divided into the components used to _build_
applications, like the
[NVIDIA CUDA Compiler Driver](/gpu-glossary/host-software/nvcc) toolchain, and
the components used _within_ or _from_ applications, like the
[CUDA Driver API](/gpu-glossary/host-software/cuda-driver-api) and the
[CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api), diagrammed
below.

![The CUDA Toolkit. Adapted from the *Professional CUDA C Programming Guide*.](themed-image://cuda-toolkit.svg)
