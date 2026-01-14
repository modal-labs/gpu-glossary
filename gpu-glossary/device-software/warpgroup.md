---
title: What is a Warpgroup?
---

A warpgroup is a set of four contiguous [warps](/gpu-glossary/device-software/warp) such that the warp-rank of the first warp is a multiple of 4.

The warp-rank of a warp is defined as:

```cpp
int linearIdx = (%tid.x + %tid.y * %ntid.x  + %tid.z * %ntid.x * %ntid.y); 
int warpRank = linearIdx / 32; 
```

So the valid warpgroups for an 8 warp dispatch are:

- **Warpgroup 0**: warp-ranks {0, 1, 2, 3}
- **Warpgroup 1**: warp-ranks {4, 5, 6, 7}

Introduced in NVIDIA's Hopper architecture, warpgroups are used to enable inter-warp collaboration for instructions such
as `wgmma.mma_async`. Upon dispatching a warpgroup level instruction, we coordinate 128 threads (4 warps * 32
threads/warp). Operating at a larger granularity removes the need for explicit inter-warp synchronization and allows work to be
performed on larger problem sizes (e.g `.m64n128k16` as opposed to `m16n8k16` for warp level instructions). 


