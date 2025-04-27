---
title: Implement FlashAttention3 Backend in SGLang [Part 1]
updated: 2025-04-26 11:11
---

<div class="subtitle">FlashAttention3 is a highly optimized attention mechanism for large language models. SGLang now adopted our work as the default backend for all mainstream LLM models like Llama, QWen, Deepseek, Gemma, etc.</div>


<div class="divider"></div>

## 0x0. Introduction

### Brief Introduction on FA3 Backend in SGLang
Share the current status of FA3 backend in SGLang which has been turned on as default in the latest release.

### Table of Contents for the series

- **Part 1:** Basic Implementation, Paged KV Cache and CUDA Graph Support
- **Part 2:** Speculative Decoding Support
- **Part 3:** MLA, LLama4, Sliding Window and Multimodal Support


<div class="divider"></div>

## 0x1. Background and Motivation

### What is FlashAttention?

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

