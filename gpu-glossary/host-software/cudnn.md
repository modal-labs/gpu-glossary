---
title: What is cuDNN?
---

NVIDIA's cuDNN (CUDA Deep Neural Network) is a GPU-accelerated library of
primitives for deep neural networks. It provides highly optimized
implementations for operations arising frequently in neural networks. These
include convolution, self-attention (scaled dot-product attention), matrix
multiplication, various normalizations, poolings, etc. Modern cuDNN computations
are expressed as operation graphs, which can be constructed using open source
[Python and C++ frontend APIs](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/developer/overview.html).
These frontends interact with a lower-level
[C backend](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/overview.html),
which also exposes an API for legacy use cases or special cases where Python/C++
isn’t appropriate.

cuDNN is a key library within the
[CUDA software platform](/gpu-glossary/host-software/cuda-software-platform),
sitting alongside its sibling library,
[cuBLAS](/gpu-glossary/host-software/cublas). Deep learning frameworks like
PyTorch leverage [cuBLAS](/gpu-glossary/host-software/cublas) for
general-purpose linear algebra, such as the matrix multiplications that form the
core of dense (fully-connected) layers. In contrast, they rely on cuDNN for
complex, specialized primitives that require their own optimized
[kernels](/gpu-glossary/device-software/kernel), such as convolutional layers,
normalization routines, and attention mechanisms.

Modern code uses cuDNN through its declarative
[Graph API](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.14.0/developer/graph-api.html).
This API allows a developer to define a sequence of operations as a graph, which
cuDNN can then analyze to perform an operation fusion, combining a sequence like
Convolution + Bias + ReLU into a single, highly optimized GPU
[kernel](/gpu-glossary/device-software/kernel). For any given operation, cuDNN
also contains multiple underlying algorithms and uses (unknown) internal
heuristics to select the most performant one for the target
[hardware](/gpu-glossary/device-hardware), data types, and input sizes. This
fusion capability is a key technique also leveraged by tools like
[TensorRT](https://developer.nvidia.com/tensorrt).

For more information on cuDNN, see the
[official cuDNN documentation](https://docs.nvidia.com/deeplearning/cudnn/), and
the [open source frontend APIs](https://github.com/NVIDIA/cudnn-frontend).
