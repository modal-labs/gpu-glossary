---
title: What is a bank conflict?
---

When multiple [threads](/gpu-glossary/device-software/thread) in a [warp](/gpu-glossary/device-software/warp) simultaneously request memory within the same bank in [shared memory](/gpu-glossary/device-software/shared-memory) but across distinct addresses, we say there is a bank conflict.

![When [threads](/gpu-glossary/device-software/thread) access distinct [shared memory](/gpu-glossary/device-software/shared-memory) banks, accesses are serviced in parallel (left). When they all access the same bank, but at different addresses, accesses are serialized (right).](GPU%20Performance%20Glossary%202251e7f1694980bd93e4f67a75c6e489/terminal-bank-conflict.png)

When bank conflicts occur, the accesses by the distinct [threads](/gpu-glossary/device-software/thread) are serialized. This reduces memory throughput substantially, that is by an integral factor, preventing the saturation of [memory bandwidth](https://www.notion.so/GPU-Performance-Glossary-2251e7f1694980bd93e4f67a75c6e489?pvs=21).

Like other SRAM cache memories, the [shared memory](/gpu-glossary/device-software/shared-memory) in a [Streaming Multiprocessor](/gpu-glossary/device-hardware/streaming-multiprocessor) is organized into groups called "banks". These banks can be accessed simultaneously, which increases the bandwidth.

In GPUs, there are 32 banks, each bank is 4 bytes wide, and consecutive words of 32 bits (not 64 bits; GPUs were designed with 32-bit floats and integers in mind) map to consecutive banks.

```
Address:  0x00  0x04  0x08  0x0C  0x10  0x14  0x18  0x1C  ...  0x7F
Bank:       0     1     2     3     4     5     6     7   ...    31

Address:  0x80  0x84  0x88  0x8C  0x90  0x94  0x98  0x9C  ...  0x7F

Bank:       0     1     2     3     4     5     6     7   ...    31
```

Addresses that differ by 32 × 4 = 128 bytes map to the same bank. [Shared memories](/gpu-glossary/device-software/shared-memory) are roughly kilobyte scale, and so multiple addresses map onto the same bank.

If we access sequential elements of an array in shared memory, each [thread](/gpu-glossary/device-software/thread) in our [warp](/gpu-glossary/device-software/warp) will hit a different bank:

```cpp
__shared__ float data[1024];  // array in shared memory

// all 32 threads access consecutive elements of data
int tid = threadIdx.x;
float value = data[tid];  // addresses: 0x000, 0x001, 0x002, ...
```

All 32 accesses complete in one memory transaction because each [thread](/gpu-glossary/device-software/thread) hits a different bank. This is depicted on the left in the figure above.

But say we wanted our [threads](/gpu-glossary/device-software/thread) to access a column in a row-major [shared memory](/gpu-glossary/device-software/shared-memory) array with 32 elements per row, and so we wrote:

```cpp
float value = data[tid * 32];  // addresses: 0x000, 0x080, 0x100 ...
// recall: floats are 4 bits wide
```

As depicted in the right side of the diagram above, all accesses hit the same bank, Bank 0, and so must be serialized, resulting in a 32x increase in latency, rising from on the order of ten cycles to on the order of hundreds. We could solve this bank conflict by transposing our [shared memory](/gpu-glossary/device-software/shared-memory) array. For more techniques to resolve bank conflicts, see the [*Introduction to CUDA Programming and Performance Optimization* talk from GTC 2024](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62191/).

Note that if [threads](/gpu-glossary/device-software/thread) access the same address in the same bank, i.e. the exact same data, conflict need not occur, as the data can be multi-/broad-cast.
