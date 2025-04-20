---
title: What is the CUDA Driver API?
---

The [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/index.html)
is the userspace component of the NVIDIA CUDA drivers. It provides utilities
familiar to users of the C standard library: a `cuMalloc` function for
allocating [memory](/gpu-glossary/device-software/global-memory) on GPU devices,
for example.

![The CUDA Toolkit. The CUDA Driver API sits between applications or other toolkit components and the GPU. Adapted from the *Professional CUDA C Programming Guide*.](themed-image://cuda-toolkit.svg)

Very few CUDA programs are written to directly use the CUDA Driver API. They
instead use the
[CUDA Runtime API](/gpu-glossary/host-software/cuda-runtime-api). See
[this section](https://docs.nvidia.com/cuda/cuda-driver-api/driver-vs-runtime-api.html#driver-vs-runtime-api)
of the CUDA Driver API docs.

The CUDA Driver API is generally not linked statically. Instead, it is linked
dynamically, typically under the name
[libcuda.so](/gpu-glossary/host-software/libcuda) on Linux systems.

The CUDA Driver API is binary-compatible: an application compiled against old
versions of the CUDA Driver API can run on systems with newer versions of the
CUDA Driver API. That is, the operating system's binary loader may load a newer
version of the CUDA Driver API and the program will function the same.

For details on distributing CUDA C applications, see the
[CUDA C/C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide)
from NVIDIA.

The CUDA Driver API is closed source. You can find its documentation
[here](https://docs.nvidia.com/cuda/cuda-driver-api/index.html).

Though they are not commonly used, there are projects that attempt to provide or
use open source alternatives to the CUDA Driver API, like
[LibreCuda](https://github.com/mikex86/LibreCuda) and
[tinygrad](https://github.com/tinygrad). See
[their source code](https://github.com/tinygrad/tinygrad/blob/77f7ddf62a78218bee7b4f7b9ff925a0e581fcad/tinygrad/runtime/ops_nv.py)
for details.
