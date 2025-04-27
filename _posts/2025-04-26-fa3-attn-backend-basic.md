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

In the past few weeks, we've implemented the Flash Attention Backend end to end in SGLang, which has been turned on by default attention backend in SGLang [`0.4.5` release](https://github.com/sgl-project/sglang/releases).

![Slack Announcement](/assets/fa3-basics/slack-announcement.png)
From this journey, we learned a lot about how Attention Backend works in modern LLM serving engines and Flash Attention itself.

We want to go through the implementation in detail and we hope this series can be helpful for anyone who wants to implement a new backend in serving engines.


<div class="divider"></div>
### Table of Contents for the series

This series will be split into 3 parts:

* **Part 1:** Basics, Paged KV Cache and CUDA Graph Support (this post)
* **Part 2:** Speculative Decoding Support (coming soon)
* **Part 3:** MLA, LLama4, Sliding Window and Multimodal Support (coming soon)

<div class="divider"></div>
### Latest Status of Attention Backend in SGLang

| **Backend**              | **Page Size > 1** | **Spec Decoding** | **MLA** | **Local Attention (Llama4) ** | **MultiModal** | **FP8** |
|--------------------------|-------------------|-------------------|--------|--------------------|------------|--------|
| **FlashAttention**                  | ✅                | ✅ (Top K >= 1)              | ✅     | ✅                 | ✅ | ✅ |
| FlashInfer | ✅                | ✅ (Top K >= 1 for non-MLA)                | ✅     | ❌                 | ✅ | ❌ |
| Triton               | ❌                | ✅                | ✅     | ❌                 | ❌ | ❌ |
| Torch Native         | ❌                | ❌                | ❌     | ❌                 | ❌ | ❌ |


### Benchmark Results

To be added.


<div class="divider"></div>

## 0x1. Background and Motivation

### What is Flash Attention?
**Flash Attention**[^1] is an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM.

![Flash Attention](/assets/fa3-basics/fa-classic.png)
It has been widely used in LLM inference and training, and is the default attention backend in modern serving engines like SGLang, vLLM, etc.

In most cases, it's fine to treat it as a black box. However, by understanding its core logic, we can use it more intelligently. 

I highly recommend this article[^2] to understand the core logic of Flash Attention. And I also have a [**blog post**](https://hebiao064.github.io/flash-attn) on Flash Attention, where I gave a brief introduction from code level.


### How Attention Backend works in SGLang

![SGLang Architecture](/assets/fa3-basics/sglang-architecture.svg)


SGLang, as a modern LLM Serving Engine, has three major components (in logical view):
- **Server Components:** Responsible for handling the incoming requests and sending responses.
- **Scheduler Components:** Responsible for construct batches and send to Worker.
- **Model Components:** Responsible for the model inference using attention backends. **The major model forward pass acceleration comes from the Attention Backend.**





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
[^3]: [Online Softmax](https://arxiv.org/abs/1805.02867)