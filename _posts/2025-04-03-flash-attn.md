---
title: What is Flash Attention?
updated: 2025-04-03 11:11
---

## Introduction
**Flash Attention**[^1] is an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM.

It has been widely used in LLM inference and training, and is the default attention backend in modern serving engines like SGLang, vLLM, etc.

## Naive Attention Calculation

Before we figure out how Flash Attention works, let's first take a look at the naive attention calculation.

$$
\begin{align}
\text{attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}
\end{align}
$$

where:
- $$\mathbf{Q}$$, $$\mathbf{K}$$, $$\mathbf{V}$$ are the query, key, and value matrices, respectively.
- $$\mathbf{d}$$ is the dimensionality of each attention head (used for scaling).


**Key Challenge:** The intermediate result of $$\mathbf{Q}\mathbf{K}^\top$$ can be extremely large. For a sequence length $$n$$, the matrix $$\mathbf{Q}\mathbf{K}^\top$$ has dimensions $$n \times n$$, leading to a memory footprint that scales as $$O(n^2)$$. As sequence lengths increase (common in large language models or multimodal applications), GPU memory usage grows quadratically, making this approach inefficient and resource-intensive.

It's intuitive to think that we can using Tiling to split the large matrix into smaller ones and compute them in parallel. However, this approach is hindered by the fact that the softmax operation is hard to be parallelized.

## Softmax Optimization

### Naive Softmax equation

$$
\text{softmax}(\mathbf{x_i}) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

Above equation is the naive softmax equation. When $$\mathbf{x_i} > 11$$, $$e^{x_i}$$ will overflow since FP16's maximum value is [65504](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html#fp16-overflow) while $$e^{12} = 162754.791419$$

### Safe Softmax: 3-pass

We need to introduce safe softmax to avoid overflow.

$$
\text{safe-softmax}(\mathbf{x_i}) = \frac{e^{x_i-m}}{\sum_{j=1}^{n} e^{x_j-m}}, m = \max(\mathbf{x_i})
$$


We can compute the attention output with Safe Softmax by 3-pass[^2]:
<div style="display: flex; justify-content: center; align-items: center; gap: 0px; flex-wrap: wrap;">
  <div style="flex: 1; min-width: 300px; text-align: center;">
    <img src="/assets/fa3-basics/3-pass-safe-softmax.png" alt="3-pass softmax" style="width: 100%;">
    <p style="font-style: italic; font-size: 0.9em;">3-pass softmax</p>
  </div>
  <div style="flex: 1; min-width: 300px; text-align: center;">
    <img src="/assets/fa3-basics/hbm-sram.png" alt="HBM-SRAM" style="width: 100%;">
    <p style="font-style: italic; font-size: 0.9em;">HBM-SRAM memory hierarchy</p>
  </div>
</div>

With this, we will read $$\mathbf{Q}$$ and $$\mathbf{K}$$ three times given SRAM cannot fit them all at once. It's not efficient since access HBM is expensive.


### Online Softmax: 2-pass

Online softmax[^3] is a technique that can be used to reduce the number of passes required to compute the softmax.

![Online Softmax](/assets/fa3-basics/2-pass-online-softmax.png)

We can clearly see that the first two passes in 3-pass softmax are fused into one pass, which is fairly good, but can we do better?

## Flash Attention V1: 1-pass

The answer is no for fuse 2 passes into 1 pass for the computation of: $$\text{softmax}(\mathbf{Q}\mathbf{K}^\top)$$, however, what we need is $$\text{softmax}\left({\mathbf{Q}\mathbf{K}^\top}\right)\mathbf{V}$$


Flash Attention is the technque which can fuse the computation of: $$\text{softmax}\left({\mathbf{Q}\mathbf{K}^\top}\right)$$ and $$\mathbf{V}$$ into one pass.

Here is the algorithm:
![Flash Attention](/assets/fa3-basics/1-pass-flash-attn.png)

The Flash Attention Author derived the algorithm hence the output only depends on $$d_i'$$, $$d_{i-1}'$$, $$m_i$$, $$m_{i-1}$$ and $$x_i$$, thus we can fuse all computations in Self-Attention in a single loop.

### Flash Attention V1: Tiling

![Flash Attention V1: Tiling](/assets/fa3-basics/fa-tiling.png)

After we fused all computations in Self-Attention in a single loop, we can use Tiling to split the large matrix into smaller ones to fully levarage the speed of GPU SRAM.

From the diagram above, we can see that $$Q$$, $$K$$ and $$V$$ has been split into blocks, and we can load them onto SRAM once to compute the attention output in the kernel.

Hence saved the intermediate results of $$S$$ and $$P$$ in HBM, and also reduced the IO Access from HBM to SRAM.



## Flash Attention V2 and V3



[^1]: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
[^2]: [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
[^3]: [Online Softmax](https://arxiv.org/abs/1805.02867)