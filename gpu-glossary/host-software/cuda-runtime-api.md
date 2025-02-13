---
title: What is the CUDA Runtime API?
---

The CUDA Runtime API wraps the
[CUDA Driver API](/gpu-glossary/host-software/cuda-driver-api) and provides a
higher-level API for the same functions.

![The CUDA Toolkit. The CUDA Runtime API wraps the CUDA Driver API to make it more amenable to application programming. Adapted from the *Professional CUDA C Programming Guide*.](themed-image://cuda-toolkit.svg)

It is generally preferred over the
[Driver API](/gpu-glossary/host-software/cuda-driver-api) for better ergonomics,
but there are some small caveats around control of kernel launches and context
management. See
[this section](https://docs.nvidia.com/cuda/cuda-runtime-api/driver-vs-runtime-api.html#driver-vs-runtime-api)
of the CUDA Runtime API docs for more.

While the Runtime API may be statically linked, per
[Attachment A of the NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html#attachment-a),
it does not have to be. The shared object file for dynamic linking is usually
named [libcudart.so](/gpu-glossary/host-software/libcudart) on Linux systems.

The CUDA Runtime API is closed source. You can find its documentation
[here](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html).
