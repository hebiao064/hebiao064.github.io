---
title: Implement FlashAttention3 Backend in SGLang [Part 1]
updated: 2025-04-26 11:11
---

<div class="subtitle">Knowledge distillation is a model compression 
technique whereby a small network (student) is taught by a larger 
trained neural network (teacher).</div>

# 1. Introduction

Share the current status of FA3 backend in SGLang which has been turned on as default in the latest release.

And this post is the first part of the series, I will share the basic implementation of the FA3 backend in SGLang.

- Part 1: Basic Implementation, Paged KV Cache and CUDA Graph Support
- Part 2: Speculative Decoding Support
- Part 3: MLA, LLama4, Sliding Window and Multimodal Support


# 2. Benchmark
Share the benchmark results of FA3 backend in SGLang comparing with other backends


# 3. Background and Motivation

## 3.1 How Attention Backend works in SGLang


## 3.2 How KV Cache Allocator works in SGLang

## 3.3 What is FlashAttention?


# 4. FlashAttention3 Backend Basic Implementation

Share the code implementation of the initial version of the FA3 backend in SGLang.


# 5. CUDA Graph Support

Share the code implementation of the CUDA Graph support in the FA3 backend.



Share some personal experience and thoughts as a new contributor to SGLang.
