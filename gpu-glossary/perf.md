---
title: Performance
---

GPUs are used when the performance of an application is inadequate on
general-purpose hardware. That makes programming for them quite different from
most other forms of programming.

For a traditional computer application, like a database management system or a
web server, correctness is the primary concern. If the application loses data or
returns incorrect results, then the application has failed. Performance is often
ignored.

When programming GPUs, correctness is typically poorly-defined. "Correct"
outputs are defined only up to some number of significant bits or only for some
underdetermined subset of "well-behaved" inputs. And correctness is at best
necessary but not sufficient. If the programmers of the application cannot
achieve superior performance (per second, per dollar, or per Watt), then the
application has failed. Programming GPUs is too hard and too limited, and
running them too expensive, for anything else to be the case.

At NVIDIA, this fact is captured in a pithy slogan: "performance is the
product".

This section of the GPU Glossary collects together and defines the key terms
that you need to understand to optimize the performance of programs running on
GPUs.

Roughly speaking, it should cover every term that you run across when using
[NSight Compute](https://developer.nvidia.com/nsight-compute) to debug GPU
[kernel](/gpu-glossary/device-software/kernel) performance issues.
