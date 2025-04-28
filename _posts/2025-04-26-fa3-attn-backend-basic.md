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

<div class="table-responsive">
  <table class="feature-table">
    <thead>
      <tr>
        <th>Backend</th>
        <th>Page Size > 1</th>
        <th>Spec Decoding</th>
        <th>MLA</th>
        <th>Llama4</th>
        <th>MultiModal</th>
        <th>FP8</th>
      </tr>
    </thead>
    <tbody>
      <tr class="highlight-row">
        <td><strong>FlashAttention</strong></td>
        <td><span class="check">✅</span></td>
        <td><span class="check">✅</span></td>
        <td><span class="check">✅</span></td>
        <td><span class="check">✅</span></td>
        <td><span class="check">✅</span></td>
        <td><span class="check">✅</span></td>
      </tr>
      <tr>
        <td>FlashInfer</td>
        <td><span class="check">✅</span></td>
        <td><span class="check">✅</span> </td>
        <td><span class="check">✅</span></td>
        <td><span class="cross">❌</span></td>
        <td><span class="check">✅</span></td>
        <td><span class="cross">❌</span></td>
      </tr>
      <tr>
        <td>Triton</td>
        <td><span class="cross">❌</span></td>
        <td><span class="check">✅</span></td>
        <td><span class="check">✅</span></td>
        <td><span class="cross">❌</span></td>
        <td><span class="cross">❌</span></td>
        <td><span class="cross">❌</span></td>
      </tr>
      <tr>
        <td>Torch</td>
        <td><span class="cross">❌</span></td>
        <td><span class="cross">❌</span></td>
        <td><span class="cross">❌</span></td>
        <td><span class="cross">❌</span></td>
        <td><span class="cross">❌</span></td>
        <td><span class="cross">❌</span></td>
      </tr>
    </tbody>
  </table>
</div>

### Benchmark Results

To be added.


<div class="divider"></div>

## 0x1. Background and Motivation

### What is Flash Attention?
**Flash Attention**[^1] is an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM.

![Flash Attention](/assets/fa3-basics/fa-classic.png)
It has been widely used in LLM inference and training, and is the default attention backend in modern serving engines like [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), etc.

In most cases, it's fine to treat it as a black box. However, by understanding its core logic, we can use it more intelligently.

I highly recommend this article[^2] to understand the core logic of Flash Attention. And I also have a blog post about [**What is Flash Attention?**](https://hebiao064.github.io/flash-attn), where I gave a brief introduction from code level.


### How Attention Backend works in SGLang

#### SGLang Architecture
![SGLang Architecture](/assets/fa3-basics/sglang-architecture.svg)


[SGLang](https://github.com/sgl-project/sglang), as a modern LLM Serving Engine, has three major components (in logical view)[^3]:
- **Server Components:** Responsible for handling the incoming requests and sending responses.
- **Scheduler Components:** Responsible for construct batches and send to Worker.
- **Model Components:** Responsible for the model inference. 

Let's focus on the model forward pass in the diagram above.

**In step 8:** the `ModelRunner` processes the `ForwardBatch` and calls `model.forward` to execute the model's forward pass.

**In step 9:** `model.forward` will call each layer's `forward` function, and the majority of the time is spent on the self-attention part. Hence the attention backend becomes the bottleneck of the model inference. In addition to performance, there are many different kind of attention variants such as **MHA, MLA, GQA, Sliding Window, Local Attention** which would require very carefully and optimized attention backend implementation.


#### Attention Backend Inheritance
Here is the inheritance relationship of the attention variants:
![Attention Variants](/assets/fa3-basics/attn-backend-inheritance.png)

Let's walk through the method in the `AttentionBackend` class to see what's the backbone of the attention backend in SGLang.

1. `forward()`: When `model.forward()` is called, the `forward` method in the `AttentionBackend` will be called. It will be calling `forward_extend()` and `forward_decode` according to the `forward_batch.forward_mode`. In this blog, we only focus on `EXTEND` and `DECODE` mode.

2. `forward_extend()`: This method will be called for each **layer** when the `forward_mode` is `EXTEND`.

3. `forward_decode()`: This method will be called for each **layer** when the `forward_mode` is `DECODE`.

4. `init_cuda_graph_state()`: This method will be called during the server startup, it will preallocate those tensors which will be used in the CUDA Graph replay.

5. `init_forward_metadata()`: This method will be called when the `model.forward()` is called. It could precalculate some metadata for the entire `model.forward()` call, reused by each **layer**, this is critical for accelerating the model inference. What's ironic is, this metadata is the most complicated part of the attention backend, once we set it up, the call of  $$\text{softmax}\left({\mathbf{Q}\mathbf{K}^\top}\right)\mathbf{V}$$ computation is quite straightforward in this context.

6. `init_forward_metadata_replay_cuda_graph`: This method will be called during the server startup, it is critical to accelerate decoding speed.

7. `init_forward_metadata_replay_cuda_graph`: This method will be called during each layer's `forward_decode` being called. It will setup the metadata for the `forwade_decode` call to make sure the CUDA Graph replay could be done correctly.

By far, we have covered all of the methods we need to implement for the attention backend. We will discuss it in following sections.


### How KV Cache works in SGLang

You might be curious about why there is a `req_to_token` in each Attention Backend class. And I didn't put it there accidentally. Actually, **KV Cache**, as the backbone of all LLM Serving Engines, is also very critical to Attention Backend, so let's briefly take a look at it.

There are two-level memory pools to manage KV cache[^4].
![KV Cache](/assets/fa3-basics/kv_cache.png)

##### req_to_token_pool
A map from a request to its tokens' KV cache indices. And this is the `req_to_token` we mentioned in Attention Backend Diagram.
- **Shape:** Max Allowed Requests Number (being set by argument `max-running-requests` for the maximum number of requests to run concurrently) * Max Context Length for each request (being set by config `model_config.context_len`)
- **Access:** 
    - Dim0: `req_pool_indices` identify the specific request
    - Dim1: token positions in req (starting from 0, 1, 2...), identify the specific token in the request
    - Value: `out_cache_loc` for token, it points to the KV cache indices associated with the token identified by Dim0 and Dim1

##### token_to_kv_pool
`req_to_token_pool` maintained the map between request to tokens KV cache indices, `token_to_kv_pool` further maps token from its KV cache indices to its real KV cache data.  Note that, for different attention implementation, like [`MHA`](https://arxiv.org/abs/1706.03762), [`MLA`](https://arxiv.org/abs/2405.04434), [`Double Sparsity`](https://arxiv.org/abs/2408.07092), `token_to_kv_pool` could have different implementation.
- **Layout:** Number of Layers * Max Allowed Tokens Number * Number of Head * Head Dimension
- **Access:** 
    - Dim0: `layer_id` identify the specific layer
    - Dim1: `out_cache_loc` identify the specific KV cache indices (free slots)
    - Dim2: Head
    - Dim3: Head Dimension
    - Value: `cache_k` for k_buffer and `cache_v` for v_buffer, the real KV Cache Data

    Note that we normally retrieve the KV Cache for entire layer all together, because we would need all prior tokens' KV in a request to do forward.

In attnetion backend, we only need to know what is `req_to_token_pool` and the rest of KV Cache management is transparent to the attention backend.

Let's give an intuitive example of what does `req_to_token_pool` looks like:
1. Assume we have 2 requests, and each request has 7 tokens.
2. Then `req_to_token_pool` is a 2 * 10 tensor, where each element is the KV cache indices for the token in the request.
    ```
    req_to_token_pool = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15]
    ]
    ```
3. After one forward_extend, the `req_to_token_pool` will be updated to:
    ```
    req_to_token_pool = [
        [0, 1, 2, 3, 4, 5, 6, 7, 16],
        [8, 9, 10, 11, 12, 13, 14, 15, 17]
    ]
    ```
4. If the first request already finished, and we ran another decode for second request, the `req_to_token_pool` will be updated to:
    ```
    req_to_token_pool = [
        [0, 1, 2, 3, 4, 5, 6, 7, 16],
        [8, 9, 10, 11, 12, 13, 14, 15, 17, 18]
    ]
    ```

With above prior knowledge, we are already good to implement the attention backend. If you want to know more about the details, please refer to the [Awesome-ML-SYS-Tutorial: KV Cache Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/d4d56dc3ab2260a747964ceb18cb1f69d23146ae/sglang/kvcache-code-walk-through/readme.md).






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

## 0x5. Footnotes
[^1]: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
[^2]: [From Online Softmax to FlashAttention](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
[^3]: [Awesome-ML-SYS-Tutorial: SGLang Code Walk Through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/d4d56dc3ab2260a747964ceb18cb1f69d23146ae/sglang/code-walk-through/readme.md)
[^4]: [Awesome-ML-SYS-Tutorial: KV Cache Code Walkthrough](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/d4d56dc3ab2260a747964ceb18cb1f69d23146ae/sglang/kvcache-code-walk-through/readme.md)