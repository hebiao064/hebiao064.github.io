---
title: JAX 是怎么分配内存的(未完成)
updated: 2026-06-11 23:33
lang: zh
---


## 引言
引言：刚刚结束了上一份工作，在开始下一份工作之前刚好有一小段时间可以更新一下我的博客了！

在上一份工作中，我遇到一个问题：Jax Training Stack跑着跑着OOM了。我之前大部分的Infra经验来自于Torch，因此对Torch的OOM问题还算了解，基本就是两套方案，
1. 勤奋点，就通过Torch Memory Snapshot分析出来memory占用的来源
2. 懒狗一点的话，就可以在code的在各种角落加上torch.cuda.empty_cache()。

所以碰到Jax OOM的时候我想到了两个子问题：
1. 为什么我们很少在Jax中遇到OOM
2. 我能不能jax.cuda.empty_cache()

带着这两个子问题我感觉我必须要搞明白Jax是怎样分配内存的，于是就有了这篇blog。


## Overview

![image](/assets/jax-memory-allocator/jax_vs_torch_thiner.png)

首先，假设你已经熟悉torch的memory allocator怎么work（如果没有，请移步这个link：[A guide to PyTorch's CUDA Caching Allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html)）。
Jax是类似的层级架构但是分配逻辑上有很大的不同。

## 什么是BFC Allocator

BFC Allocator是BFC(Best-Fit with Coalescing) Allocator的缩写，他的简化版工作原理是：
1. 一开始会分配比较大的一片内存。
2. 当Jax程序需要不同大小的内存时，我们其中一块大小最接近的Chunk，然后分配给Jax程序。
3. 当Jax程序释放内存时，我们会将这块内存标记为可用，并将其放回Region中。并且我们会通过这个Chunk的prev，next指针来判断它是否可以跟前后Chunk合并从而减少碎片化


### BFC Allocator的详细数据结构
接下来我们细说一下BFC Allocator的详细数据结构。
1. Region：
2. Chunk：
3. Bin：

（次数放一个图表示他们的关系）

### BFC Allocator的API

1. extend
2. split
3. merge

## 举例说明BFC Allocator的分配逻辑

1. 一开始分配一片大的


2. allocate一系列chunk


3. 释放一些chunk


## 结论
比较Torch Memory Allocator

## 引申
BFCAllocator里有一句这只是简化版的dlmalloc，而torch的本质是更接近于xxmalloc（感谢xxx的insight）