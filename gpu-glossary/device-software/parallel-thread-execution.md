---
title: What is Parallel Thread Execution?
abbreviation: PTX
---

Parallel Thread eXecution (PTX) is an intermediate representation (IR) for code
that will run on a parallel processor (almost always an NVIDIA GPU). It is one
of the formats output by `nvcc`, the
[NVIDIA CUDA Compiler Driver](/gpu-glossary/host-software/nvcc).

NVIDIA documentation refers to PTX as both a "virtual machine" and an
"instruction set architecture".

From the programmer's perspective, PTX is an instruction set for programming
against a virtual machine model. Programmers or compilers producing PTX can be
confident their program will run with the same semantics on many distinct
physical machines, including machines that do not yet exist. In this way, it is
also similar to CPU instruction set architectures like
[x86_64](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html),
[aarch64](https://developer.arm.com/documentation/ddi0487/latest/), or
[SPARC](https://www.gaisler.com/doc/sparcv8.pdf).

Unlike those ISAs, PTX is very much an
[intermediate representation](https://en.wikipedia.org/wiki/Intermediate_representation),
like LLVM-IR. The PTX components of a
[CUDA binary](/gpu-glossary/host-software/cuda-binary-utilities) will be
just-in-time (JIT) compiled by the host
[CUDA Drivers](/gpu-glossary/host-software/nvidia-gpu-drivers) into
device-specific [SASS](/gpu-glossary/device-software/streaming-assembler) for
execution.

In the case of NVIDIA GPUs, PTX is forward-compatible: GPUs with a matching or
higher [compute capability](/gpu-glossary/device-software/compute-capability)
version will be able to run the program, thanks to this mechanisn of JIT
compilation.

Some exemplary PTX:

```ptx
.reg .f32 %f<7>;
```

- a compiler directive for the
  PTX-to-[SASS](/gpu-glossary/device-software/streaming-assembler) compiler
  indicating that this kernel consumes seven 32-bit floating point
  [registers](/gpu-glossary/device-software/registers). Registers are
  dynamically allocated to groups of
  [threads](/gpu-glossary/device-software/thread)
  ([warps](/gpu-glossary/device-software/warp)) from the
  [SM](/gpu-glossary/device-hardware/streaming-multiprocessor)'s
  [register file](/gpu-glossary/device-hardware/register-file).

```ptx
fma.rn.f32 %f5, %f4, %f3, 0f3FC00000;
```

- apply a fused multiply-add (`fma`) operation to multiply the contents of
  registers `f3` and `f4` and add the constant `0f3FC00000`, storing the result
  in `f5`. All numbers are in 32 bit floating point representation. The `rn`
  suffix for the FMA operation sets the floating point rounding mode to
  [IEEE 754 "round even"](https://en.wikipedia.org/wiki/IEEE_754) (the default).

```ptx
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ntid.x;
mov.u32 %r3, %tid.x;
```

- `mov`e the `x`-axis values of the `c`ooperative `t`hread `a`rray `i`n`d`ex,
  the cooperative thread array dimension index (`ntid`), and the `t`hread
  `i`n`d`ex into three `u32` registers `r1` - `r3`.

The PTX programming model exposes multiple levels of parallelism to the
programmer. These levels map directly onto the hardware through the PTX machine
model, diagrammed below.

![The PTX machine model. Modified from the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#ptx-machine-model).](themed-image://ptx-machine-model.svg)

Notably, in this machine model there is a single instruction unit for multiple
processors. While each processor runs one
[thread](/gpu-glossary/device-software/thread), those threads must execute the
same instructions — hence _parallel_ thread execution, or PTX. They coordinate
with each other through
[shared memory](/gpu-glossary/device-software/shared-memory) and effect
different results by means of private
[registers](/gpu-glossary/device-software/registers).

The documentation for the latest version of PTX is available from NVIDIA
[here](https://docs.nvidia.com/cuda/parallel-thread-execution/). The instruction
sets of PTX are versioned with a number called the
"[compute capability](/gpu-glossary/device-software/compute-capability)", which
is synonymous with "minimum supported
[Streaming Multiprocessor architecture](/gpu-glossary/device-hardware/streaming-multiprocessor-architecture)
version".

Writing in-line PTX by hand is uncommon but not unheard of, similar to writing
in-line `x86_64` assembly, as is done in high-performance vectorized query
operators in analytical databases and in performance-sensitive sections of
operating system kernels. At time of writing in October of 2024, in-line PTX is
the only way to take advantage of some Hopper-specific hardware features like
the `wgmma` and `tma` instructions, as in
[Flash Attention 3](https://arxiv.org/abs/2407.08608) or in the
[Machete w4a16 kernels](https://youtu.be/-4ZkpQ7agXM). Viewing
[CUDA C/C++](/gpu-glossary/host-software/cuda-c),
[SASS](/gpu-glossary/device-software/streaming-assembler), and
[PTX](/gpu-glossary/device-software/parallel-thread-execution) together is
supported on
[Godbolt](https://godbolt.org/#z:OYLghAFBqd5TKALEBjA9gEwKYFFMCWALugE4A0BIEAZgQDbYB2AhgLbYgDkAjF%2BTXRMiAZVQtGIHgBYBQogFUAztgAKAD24AGfgCsp5eiyahUAV0wtyKxqiIEh1ZpgDC6embZMQAJnLOAGQImbAA5TwAjbFJfWQAHdCViByY3Dy9fcgSk%2ByEgkPC2KJifWRtsOxSRIhZSIjTPbz9yyqFq2qJ8sMjo2OsauoaM5oHO4O6i3tKASmt0M1JUTi4AUh8AZjjSFmA2FgBqISWVrQBBE/ON4NQPHH2V9ZdxJRU6h9wLtfXr2%2Bx7x9QSiIhHQ70%2BVyYNwsfweTyBmHoBAiYLOXx%2B0P%2BTzMESMSgA%2BgA3HwAOiQKMu30hv0x5kseNIZmEBA4pPJFzxeOA9HQEQkHP2BPQBEw%2ByUwGwbDYnO5vPoeI4UowEmwSiWEGCRH2AFlyPsNftQrr9QBpXU0bksTUSOJIKwXfYOx1O50u11ul0YJhA/bm9CW/YAKlOus93t9/oDACFyPb3XH4/Hw5qojUzRbNQGXNN7gB2SOx/WRgIAeRcxpEAEkAFofdYAEXWPge%2BbODtDmv1qAASugAO7/Ov7HHoVAAawrmHUxPUgf2RdL5eruHuPkj%2BwgRCQpGwLEwE6nM4A9HOS2XKzXps3Y%2B29cJ9qg0gOh9yx/viQBPWfzs9Lldrjdbjue6TtOK4AKwngu564Je6wtucrb7Iex4EDQoo1EQErMB2Sj7CESwvLUn4kPseyjn8m7BMAuG9mQo77IyOCkPs9iMPsACOZjGPYABelopAWaEQN2faYtqK4AGxrBJ96PrCBrZiseaxg6SYsWwcRPloxJaFeiGqWQ676gQWnNnqYnGmZaz5quBCKcp%2BmOkQGl/g8g7nGBkYif2Ab7Maf56isYGDr5%2BaeSZvmhAFD7uEFdZ6acTpKfFjkuEFXk9j5BrRWkcVPtatqzs5mnWUO2A1LOaWed5s5RaVMX0HFCUOslnw5nWXCzPQ3Bgfw3hcDo5DoNwLgKHWiVpaupVKPMiwwhsfDkEQ2idbMSBAb0ECzKOsTEgAnAdh1HUdEmGNw0h9StQ3cPwSggFoS0rbMcCwCgGAaQw0SUNQ71xJ9MRMASqCoDwPA5uQOAEgQSwAGoENgvbFnEzBXXQ9CYaQd0QBEV0RMERHcIt70cMIxZMPQ75XTgewmJIA38IQ26VASKpXdg6gVGYmGE/wGrYN19OGEi2ykO%2Bbg4FdRCkMyPOzOaOxKHDCNIyjvD8IIwhiMqUiyBrigqBoV36DwhjGKYFhWIiER3ZAszoHEuRetwAC0xbrPszt1qEda4JGCgAOIe874ohNsmHOxgOBubUqBkvWv1c9geIABynZH2BuelCeYSnp1KFKqd5c7zsO%2BoLBKM79uO0obnOwSHvwm5qDWdZYO3fzFSO04TCuO4jQGIE4yFMUBjZMkQhDN4Jtj47XTD70JstI77SDH3GSLx3rRMCvYwFD0MSL6Mk8GECHRz/vUizDNCxLL4XU9ZdgvDVwBowy4Lj7KDxI5jp674MQhkvg%2BGmPwZa9NphrQ2jELa5AdqlH2sdRBB1ToCwuuQfqg1n63Xuo9cB5AXrIDQFgPAhASAUCoLQT6rAOA8zkJrcQkgZB0P1moTQgt9B%2BCMCYNAFtrCby7hAZwx8/CDz3pMEo8REjj1SGvJoWQpGzyHhfUofDbDLyPrIzIS8qijHPuIvop9V7pDkYY3eEwR4zDmDfZYXwtg7D2IcSEGdUSoghFCO4sJnivCIGyNx1JPHwgcL4yk7iYQAnhFbYJ6IPEAmxLiQkJI44fFcSE/xsS6QMiZCyJJnwzgci5DyPkeIBRChFGKCUUoCmynlBKPEjIlSMFVNgTA6o7w6lvJqQ0HS/Jpj9FaegNo7SOQTCMkZN41JBhDEIMM6ZAzRhUqMxZcY1IpisD6WZmZ7LwQdIWU8i4axuUbM1e80yOx3hqm5Z8I5xwgRnKFPZ0EAoAW3LuN8M41gQW/PsmCxybydjkvWK5r4QKfnuVBX8pVnlATeUhSCP4LwJVjMhPUaFT6YRJjhPCFQVRKCIixdApEWDkRYkgKiNE6IMR7tEFin0OJcSZHxR2gl1wXMeOJaS0lZLuDEqELZCy1LFS0jpY5BlmKtI7KZOC5l5KWSlS3WyfLhnqRKlNQFHkMqiV8v5UqBA8qhXShFbK9VcrBRFbmFKiVHRVQ1VlOqq4uWNWCvlAZhVfKCtKqsyqWdMq1RyrFU1cFYytVRO1TqZ0uC9XQVdZ%2Bo1xr7EmjZNc185orkbKAp6kDdybW2rtJBSCUHnX4GwEA6wf5gSjU/G61gcFgJ0M9RAEA3roA%2BowchP1m1/VbSAQGwNQbg0htDbAStEbIwwerBgGMsY40FnjVgotaHE2wmTCmVMJRmzpoNRmncCAszuoLdmnNuZq0oMIfmV0rYizFsQyW0ti1qzlkYaiw6Vb9UWnrLWjDdbyGUKwo2mQuHm0sELa28A7YOxSHu127tPbe19gHIOIdoiWmwBHYh0dFhxzrNnJOqdyDp0ztVDtidc7WALlJJ1xdS7l0ruB6Ztd67O0bvHeVkY26qO3Y4QRPdj4m1EeYhe8icgpB44J6ReiR4bzUTojoIntFtF0Uo/Rh8ZOaOU3UcTC8r6zVvjwe%2BEbH6YO4K/d%2Bn8eDf1/hAf%2BZDU26fTeAzNTFqA5vgXmxBBauBoLHddLg2CHq1tWuGnwRaQBgW0snHge0JI5h8DwdYoNpA5jAuWrzWDcF1vwQ2htKAB1LG%2BsJDt/1QjsGWKEN%2BH8v4/0Gs0gB0tMAGHfQwzgTC9Y/sNuwrR/CUjd17sYgePcNMH1E47ETM8UgDZPp1%2BTKneuSY49vBTYiJP9Gm/3NTZj54H1mFLbA2BhT3T05GlL3A6zYChksfYw7qWlZMxVizVnAELV1G4Ft1KgEgLSwF9aWboE5ukBJYkElwtA54MDoH4a0HFp4FoB6R2fPVr809DL8BG0gBy9gPLv1CvFe4Nd8rZnKv8Gq2QvbfgGva2a9%2Bg2bDBrG3Y1vbrIm%2BMbdHgo4TqmhtjcU0tuT82Vvrzp%2Bos%2BXOBOmNkwt/jm2lrbl23VrQB2DP8Gfids7fxLvMVx6Z8zWg/6kIe42J7BXW02fe/5iB5AvuOZgTtMC6xiRxdByDx3ydOGFvIJD6HFbDNw7ugj%2BzgXgvSD2sSEHWhGw5lTsnZOYFSgSVOrDuzdac1Q4egLdYCvvMJ4CyzTGXXpBAA%3D%3D).
See the
[NVIDIA "Inline PTX Assembly in CUDA" guide](https://docs.nvidia.com/cuda/inline-ptx-assembly/)
for details.
