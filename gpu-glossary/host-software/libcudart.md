---
title: What is libcudart.so?
---

The typical name for the binary shared object file that implements the
[CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api) on Linux
systems. Deployed CUDA binaries often statically link this file, but libraries
and frameworks built on the CUDA Toolkit, like PyTorch, typically load it
dynamically.
