---
title: Draft - Efficient RL Training - Optimizing Weight Sync in slime
updated: 2025-08-27 11:11
---

### Still in draft, ETA: 2025-09-01

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
    <h3 style="margin: 0 0 5px 0; font-size: 18px; font-weight: 500;">Zilin Zhu</h3>
    <div class="author-social" style="display: flex; gap: 12px; margin-top: 5px;">
      <a href="https://www.linkedin.com/in/zilin-zhu/" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/linkedin.svg" alt="LinkedIn" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
      <a href="https://github.com/zhuzilin" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/github.svg" alt="GitHub" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
    </div>
  </div>

  <!-- Author 3 -->
  <div class="author-card" style="display: flex; flex-direction: column; align-items: center; max-width: 200px; text-align: center;">
    <h3 style="margin: 0 0 5px 0; font-size: 18px; font-weight: 500;">Ji Li</h3>
    <div class="author-social" style="display: flex; gap: 12px; margin-top: 5px;">
      <a href="https://www.linkedin.com/in/gelee-q/" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/linkedin.svg" alt="LinkedIn" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
      <a href="https://github.com/GeLee-Q" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/github.svg" alt="GitHub" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
    </div>
  </div>
</div>

<div class="divider"></div>


## 1. What is slime?

slime is a LLM post-training framework aiming for RL Scaling, it was designed to be:

- **Versatile** – with a fully customizable rollout interface and flexible training setups (colocated or decoupled, synchronous or asynchronous, RL or SFT cold start).
- **Performant** - integrating SGLang for inference and Megatron-LM for training, natively.
- **Maintainable** - with a lightweight codebase and smooth transition from Megatron pretraining to SGLang deployment.
<div class="divider"></div>

![What is slime?](/assets/slime/weight_sync/slime_overview.png)

Here is the overview of the slime RL infrastructure. The system consists of three core modules:
**Training (Megatron)** – handles the main training process, reads data from the Data Buffer, and
synchronizes parameters with the rollout module after training; **Rollout (SGLang + Router)** –
generates new data, including rewards and verifier outputs, and writes it to the Data Buffer; **Data
Buffer** – serves as a bridge module that manages prompt initialization, custom data, and rollout
generation strategies.
<div class="divider"></div>


## 2. What is Weight Sync?

![What is weight sync?](/assets/slime/weight_sync/what_is_weight_sync.png)
<br>

**Weight sync** in LLM reinforcement learning (RL) refers to the process of **copying updated model weights from the training side to the inference side** so that inference workers (used for generating samples, evaluations, or environment rollouts) always use up-to-date parameters. 
<div class="divider"></div>


## 3. How weight sync works in slime?

![How weight sync works in slime?](/assets/slime/weight_sync/how_weight_sync_works.png)


The weight sync can be outlined in 5 steps:

1. In Megatron Worker Group, gather all tensors from distributed workers, by gathering tensors which was distibuted by PP/TP/EP/ETP in training process.
2. Serialize the Tensor into CudaIpcHandlers 
3. Call SGLang Server’s update_weight_by_tensor API
4. Scatter the CudaIpcHandlers into SGLang’s TP Workers
5. Deserialize back by rebuilding CUDA Tensors and Load Weights

<div class="divider"></div>


## 4. Our optimization journey

![Our optimization journey](/assets/slime/weight_sync/our_optimization_journey.png)


Through the journey we’ve adopted many optimizion and here we will discuss them in detail.
<div class="divider"></div>

### 4.0 Cross Process Data Transfer on GPU


![CUDA IPC Handler](/assets/slime/weight_sync/cuda_ipc_handler.png)

    1. Normally, to send data between process, we can just serialize the data into base64 and deserailize in the consumer process
    2. Considering the model weight could be huge (e.g: 60GB for a QWen3-30B-A3B model), serialize it into base64 is not a good idea
    3. Glad that we can serailize the cuda tensor to cudaipc and we can rebuild the tensor on  within the same tensor in the consumer process, by doing that we can make it much simpler. 
    4. Since this is the very first version, so we let’s call it baseline.
<div class="divider"></div>

### 4.1 Optimizing the tensor gathering process: *From 120s to 90s*

    1. Async gather tensors scatter by different distributed parallesim paradigm(PP/TP/EP)
    2. Instead of gather them one by one, we choose to run `dist.all_gather(param_list, param, async_op=True)` to maximize the bandwidth, here is the code pointer: https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/update_weight_utils.py#L59-L123


<div class="divider"></div>


### 4.2 Optimizing the SGLang Server Calls by cumulate the tensors into buckets: *From 90s to 30s*


    1. Use bucket size 512mb and send list to avoid CPU Overhead
    2. And introduce the diff between MOE and Dens

<div class="divider"></div>



<div class="divider"></div>

### 4.3 Merge the tensor list into one tensor to reduce cudaipc open and close

    1. Draw a graph


<div class="divider"></div>


### 4.4 Load Weight Optimization

    1. Add PRs




<div class="divider"></div>

## 5. Common Question

- Why we use server not engine?
    - Advantage of server, and decoupling, also router

<div class="divider"></div>
## 6. Next Step

## 

- Overlap the gathering and sending
- Load weight in Async
- Pre-calculate the layout of inference engine and do zero-redundancy copy

<div class="divider"></div>

## 6. Acknowledgments

We extend our gratitude to the Slime Team and  SGLang RL Team and verl Team.

<div class="divider"></div>


## 7. Footnotes

[^1]: [LlamaRL Paper](https://arxiv.org/pdf/2505.24034)
[^2]: [Torch Memory Saver: A PyTorch library that allows tensor memory to be temporarily released and resumed later](https://github.com/fzyzcjy/torch_memory_saver)
[^3]: [CUDA 10.2: Introducing Low-Level GPU Virtual Memory Management](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/)
[^4]: [LD_PRELOAD](https://catonmat.net/simple-ld-preload-tutorial)