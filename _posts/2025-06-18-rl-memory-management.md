---
title: Efficient RL Training - Optimizing Memory Usage in veRL (Draft)
updated: 2025-06-18 11:11
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
    <h3 style="margin: 0 0 5px 0; font-size: 18px; font-weight: 500;">Ata Fatahi </h3>
    <div class="author-social" style="display: flex; gap: 12px; margin-top: 5px;">
      <a href="https://www.linkedin.com/in/atafatahibaarzi/" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/linkedin.svg" alt="LinkedIn" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
      <a href="https://github.com/MrAta" target="_blank" style="text-decoration: none; border: none;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v8/icons/github.svg" alt="GitHub" style="width: 18px; height: 18px; filter: invert(30%);">
      </a>
    </div>
  </div>
</div>

<div class="divider"></div>


## 1. Introduction

As a machine learning enthusiast, I've been diving deep into reinforcement learning (RL) for large language models (LLMs). One of the most fascinating projects I've explored is veRL, a library designed for RL training of LLMs. It combines training strategies like Fully Sharded Data Parallel (FSDP) and Megatron-LM with inference engines like SGLang for rollout generation. RL training is unique because it blends inference and training in every step, which poses significant challenges for scalability and resource efficiency. In this blog post, I’ll share my journey into optimizing memory usage in veRL, focusing on techniques that reduce peak memory demands and enable training larger models on limited GPU resources.

<div class="divider"></div>

## 2. Understanding RL Training in veRL

<br>

#### High-level diagram of the RL training process in RL training:

![An example flow of online RL training](/assets/rl-memory-management/example-flow-diagram.png)

<br>

This is an example flow of **online RL training**. We've ignored the reference model and critic model for simplicity. For reward model, we assume it's a simple reward function not a reward model, as is often the case for code and reasoning applications. The policy model has two instances, implementing based on the Training Framework (FSDP/Megatron-LM) and Inference Engine (SGLang/vLLM) for training and inference optimizations, respectively. [^1]

<br>

#### Simplified example using Proximal Policy Optimization (PPO):

And here's a simplified example using Proximal Policy Optimization (PPO):

```python
for prompts, pretrain_batch in dataloader:
    # Stage 1: Rollout generation (inference)
    batch = actor.generate_sequences(prompts)
    # Stage 2: Prepare experience
    batch = reference.compute_log_prob(batch)
    batch = reward.compute_reward(batch) # could be reward model or a simple reward function
    batch = compute_advantages(batch, algo_type)
    # Stage 3: Actor and critic training
    actor_metrics = actor.update_actor(batch)
```

Each step involves a rollout (inference) using the actor model, followed by training the actor model. A key design choice in veRL is keeping both the rollout and training versions of the actor model on the same GPUs (aka **Co-locate**), which optimizes resource sharing but complicates memory management. My focus here is on the actor model’s memory usage.




## 3. The Memory Challenge

In RL training, memory usage spikes during transitions between rollout and training phases. The challenge is managing these memory-intensive operations on the same GPUs without running into out-of-memory (OOM) errors.

Let's examine the memory footprint for QWen 7B (7.72B parameters) on H200 GPU Node using FSDP and SGLang:

#### Training Phase Memory Breakdown

With FSDP sharding across 8 GPUs, each GPU holds:

- **Model Weights (Sharded)**: ~1.9 GB (15.4 GB ÷ 8 GPUs)
- **Gradients (Sharded)**: ~1.9 GB (same partitioning as weights)
- **Optimizer States (Sharded)**: ~3.9 GB (30.8 GB ÷ 8 GPUs for AdamW)
- **Activations**: ~20-40 GB (not sharded, varies with batch size and sequence length)
- **Temporary All-gather**: ~15.4 GB (during forward/backward, full parameters temporarily reconstructed)

**Peak Training Memory**: ~40-65 GB per GPU (including temporary all-gather)

#### Rollout Phase Memory Breakdown

During inference, the full model is typically loaded (not sharded):

- **Model Weights**: ~15.4 GB (full model for inference efficiency)
- **KV Cache**: ~80-100 GB (dominant factor, can be tuned by `mem-fraction` in SGLang)
- **CUDA Graph**: ~2-4 GB (captures computation graph for inference acceleration)
- **Input/Output Buffers**: ~5-10 GB (request batching and response generation)

**Total Rollout Memory**: ~100-120 GB per GPU


## 4. Memory Optimization Journey

### 4.1: The Naive Approach

Let’s start from a vanilla version, we don’t do any offload, we keep training model weights and serving engine memory in the same GPU.

![v0: The Naive Approach](/assets/rl-memory-management/v0-naive-approach.png)


Given the SGLang will keep occupying the large amount of memory, it's not possible for us to start training.

Note: This is just a vanilla idea, we never actually implemented it.


### 4.2: Offloading Weights to CPU and Relaunch Inference Engine

Intuitively, we can offload the training model weights to the CPU after training and serialize them to disk. During the rollout phase,  we can relaunched the SGLang engine, loading weights from disk. 

![v1: Offloading Weights to CPU and Relaunch Inference Engine](/assets/rl-memory-management/v1-offload-weights-to-cpu.png)

This reduced GPU memory usage during rollout but it will make the training process super sloe:
- Slow Disk I/O: Loading weights from disk was time-consuming.
- Slow Inference Engine relaunch with CUDA Graph re-capture

This approach was a step forward but too slow for practical use.



### 4.3: Sleeping the Inference Engine

Then you might wondered if we could “sleep” the SGLang engine during training, freeing GPU memory without destroying the CUDA graph. The key was to release physical memory while preserving virtual memory addresses. Fortunately, the SGLang RL team (Thanks to [Tom](https://github.com/fzyzcjy)) introduced the [torch_memory_saver](https://github.com/fzyzcjy/torch_memory_saver) library [^2], which does exactly that. 


Here’s a basic example:

```python
import torch_memory_saver

memory_saver = torch_memory_saver.memory_saver

# 1. For tensors that wants to be paused, create them within `region`
with memory_saver.region():
    pauseable_tensor = torch.full((1_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# 2. After `pause`, CUDA memory is released for those tensors.
# For example, check `nvidia-smi`'s memory usage to verify.
memory_saver.pause()

# 3. After `resume`, CUDA memory is re-occupied for those tensors.
memory_saver.resume()
```

And the under the hood, it's customizing the cuda malloc and cuda free api by memorizing the virtual memory address on the GPU. So when we release the memory, the physical memory is actually released and any other process can use it, and when we resume it, the virtual memory address is still the same. For model weights, we will sync the weights from training gpu so we are good, for KV Cache, we just flush it every rollout so we are good as well.

Here is the rough implementation of the CUDA malloc and free api customization:

```cpp
struct _AllocationMetadata {
    size_t size;
    CUdevice device;
    CUmemGenericAllocationHandle allocHandle;
    std::string tag;
};
class TorchMemorySaver {
public:
    TorchMemorySaver() {}
    cudaError_t malloc(void **ptr, size_t size) {
        CUdevice device;
        CURESULT_CHECK(cuCtxGetDevice(&device));

        CUmemGenericAllocationHandle allocHandle;
        CUDAUtils::cu_mem_create(&allocHandle, size, device);

        CURESULT_CHECK(cuMemAddressReserve((CUdeviceptr *) ptr, size, 0, 0, 0));
        CURESULT_CHECK(cuMemMap((CUdeviceptr) * ptr, size, 0, allocHandle, 0));
        CUDAUtils::cu_mem_set_access(*ptr, size, device);

        {
            const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);
            allocation_metadata_.emplace(*ptr, _AllocationMetadata{size, device, allocHandle});
        }

        return cudaSuccess;
    }
    
    void resume() {
        const std::lock_guard <std::mutex> lock(allocator_metadata_mutex_);

        for (auto it = allocation_metadata_.begin(); it != allocation_metadata_.end(); ++it) {
            void *ptr = it->first;
            _AllocationMetadata &metadata = it->second;

            CUmemGenericAllocationHandle newAllocHandle;
            CUDAUtils::cu_mem_create(&newAllocHandle, metadata.size, metadata.device);

            CURESULT_CHECK(cuMemMap((CUdeviceptr) ptr, metadata.size, 0, newAllocHandle, 0));

            CUDAUtils::cu_mem_set_access(ptr, metadata.size, metadata.device);
            metadata.allocHandle = newAllocHandle;
        }
    }
  }

```


To speed up weight reloading, I kept the training model weights on the GPU and copied them to the rollout engine using a tensor copy mechanism inspired by PyTorch’s forking pickle (see PyTorch PR #344). This significantly reduced overhead, making training-to-rollout switches much faster.

Until now, we have a pretty decent solution for the memory challenge.
![v2: Sleeping the Inference Engine](/assets/rl-memory-management/v2-sleeping-inference-engine.png)


### 4.4: Multi-Stage Awake

Despite these improvements, we received a lot of user complain from the verl users: OOM errors during the initial weight synchronization for larger models like QWen 32B. The problem occurred because resuming the entire rollout engine (weights and KV cache) at once consumed too much memory before the training model could be offloaded.

We realized that there are actually a big chunk of memory that being wasted during the resume process, 
we could split the resume process into stages:

- Resume only the model weights for synchronization.
- Offload the training model.
- Resume the KV cache for rollout.


The idea is pretty straghtforward, but the implementation is not.

Because torch_memory_saver’s singleton design didn’t support selective pausing/resuming of different memory regions. After discussions with the community, we implemented a tag-based resume/pause system instead implementing multiple instances. 

We actually implemented both and both works, but the tag-based approach is more flexible and easier to use.
You can checkout all the code in the Option section for the [RFC](https://github.com/sgl-project/sglang/issues/7009)


After the support of the tag-based approach, we can now split the resume process into stages like below:

```python
from torch_memory_saver import torch_memory_saver

# Create tensors with tags
with torch_memory_saver.region(tag="weights"):
  tensor1 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

with torch_memory_saver.region(tag="kv_cache"):
  tensor2 = torch.full((5_000_000_000,), 100, dtype=torch.uint8, device='cuda')

# Pause and resume selectively
torch_memory_saver.pause("weights")
torch_memory_saver.pause("kv_cache")

torch_memory_saver.resume("weights")
# Sync weights and offload training model
torch_memory_saver.resume("kv_cache")
```


And here is the diagram of the multi-stage resume process:

![v3: Multi-Stage Resume](/assets/rl-memory-management/v3-multi-stage-resume.png)







<div class="divider"></div>

## 5. Conclusion and Future work


## 6. Acknowledgments



## 7. Footnotes

[^1]: [LlamaRL Paper](https://arxiv.org/pdf/2505.24034)
[^2]: [Torch Memory Saver: A PyTorch library that allows tensor memory to be temporarily released and resumed later](https://github.com/fzyzcjy/torch_memory_saver)