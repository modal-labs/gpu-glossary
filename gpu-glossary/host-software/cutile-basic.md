---
title: "What is cuTile BASIC?"
---

cuTile BASIC is an implementation of the
[CUDA Tile programming model](/gpu-glossary/device-software/cuda-tile-programming-model)
in the [BASIC programming language](https://modal-cdn.com/BASIC_Oct64.pdf).

BASIC stands for Beginner's All-purpose Symbolic Instruction Code. BASIC is a
programming language designed, in the 1960s, for ease-of-use and interactive
programming. It was popular among early microcomputer programmers like William
Gates III.

cuTile BASIC was released
[as an April Fools' joke](https://developer.nvidia.com/blog/cuda-tile-programming-now-available-for-basic/).
It is a real, if toy, implementation of the programming model and a
demonstration of its generality. You can run the vector-addition cuTile BASIC
kernel below on a B200 GPU using
[this Modal Notebook](https://modal.com/notebooks/modal-labs/examples/nb-151VgRNHYEDuKSfxJRjV5N).
cuTile BASIC was developed, in part, via such Notebooks.

```basic
10 REM Vector Add: C = A + B
20 INPUT N, A(), B()
30 DIM A(N), B(N), C(N)
40 TILE A(128), B(128), C(128)
50 LET C(BID) = A(BID) + B(BID)
60 OUTPUT C
70 END
```
