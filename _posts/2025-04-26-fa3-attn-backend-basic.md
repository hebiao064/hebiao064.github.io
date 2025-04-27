---
title: Implement Flash Attention Backend in SGLang - Basics and KV Cache
updated: 2025-04-26 11:11
---



<div class="authors-section" style="display: flex; justify-content: center; margin: 40px 0; gap: 40px;">
  <div style="margin: 0 0 5px 0; font-size: 18px; font-weight: 500;"><p>Authored by</p></div>
  <!-- Author 1 -->
  <div class="author-card" style="display: flex; flex-direction: column; align-items: center; max-width: 200px; text-align: center;">
    <h3 style="margin: 0 0 5px 0; font-size: 18px; font-weight: 500;">Biao He</h3>
    <div class="author-social" style="display: flex; gap: 12px; margin-top: 5px;">
      <a href="https://www.linkedin.com/in/biao-he/" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/linkedin.svg" alt="LinkedIn" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
      <a href="https://x.com/hebiao064" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://www.svgrepo.com/show/47722/twitter-black-shape.svg" alt="X" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
      <a href="https://github.com/hebiao064" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/github.svg" alt="GitHub" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
    </div>
  </div>

  <!-- Author 2 -->
  <div class="author-card" style="display: flex; flex-direction: column; align-items: center; max-width: 200px; text-align: center;">
    <h3 style="margin: 0 0 5px 0; font-size: 18px; font-weight: 500;">Qingquan Song</h3>
    <div class="author-social" style="display: flex; gap: 12px; margin-top: 5px;">
      <a href="https://www.linkedin.com/in/qingquan-song-b71167119/" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/linkedin.svg" alt="LinkedIn" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
      <a href="https://x.com/qingquan_song" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://www.svgrepo.com/show/47722/twitter-black-shape.svg" alt="X" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
      <a href="https://github.com/qingquansong" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/github.svg" alt="GitHub" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
    </div>
  </div>
</div>

<div class="divider"></div>


## 0x0. Introduction

### Brief Introduction on FA3 Backend in SGLang
Share the current status of FA3 backend in SGLang which has been turned on as default in the latest release.

### Table of Contents for the series

- **Part 1:** Basics, Paged KV Cache and CUDA Graph Support
- **Part 2:** Speculative Decoding Support (coming soon)
- **Part 3:** MLA, LLama4, Sliding Window and Multimodal Support (coming soon)


<div class="divider"></div>

## 0x1. Background

### Understanding the core logic of FlashAttention

**Flash Attention**[^1] is a technique that significantly speeds up and reduces the memory requirements of attention mechanisms within transformer models.

**Note:** There are already plenty of blog posts and articles explaining FlashAttention, and in most cases, it's fine to treat it as a black box. However, by understanding its core logic, we can use it more intelligently.


#### Naive Attention Calculation

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

#### Softmax Optimization

**Naive Softmax equation:**

$$
\text{softmax}(\mathbf{x_i}) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

Above equation is the naive softmax equation. When $$\mathbf{x_i} > 11$$, $$e^{x_i}$$ will overflow since FP16's maximum value is [65504](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/accuracy-considerations.html#fp16-overflow) while $$e^{12} = 162754.791419$$

**Safe Softmax:**

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


**Online Softmax:**

TBD

#### FlashAttention

One pass



### How Attention Backend works in SGLang

### How KV Cache Allocator works in SGLang




<div class="divider"></div>

## 0x2. FlashAttention3 Backend Basic Implementation

Share the code implementation of the initial version of the FA3 backend in SGLang.



<div class="divider"></div>

## 0x3. CUDA Graph Support

Share the code implementation of the CUDA Graph support in the FA3 backend.




<div class="divider"></div>

## 0x4. Thoughts

### Share some personal experience and thoughts as a new contributor to SGLang.

### Future Work


<div class="divider"></div>

## 0x5. References
- Tri Dao's [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- BBUF's Blog: [BBUF's Zhihu Blog](https://zhuanlan.zhihu.com/p/1888278828897523473)


## 0x6. Footnotes
[^1]: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
[^2]: [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
