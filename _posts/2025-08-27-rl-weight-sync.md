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

<br>
![What is slime?](/assets/slime/weight_sync/slime_overview.png)
<br>
The system consists of three core modules:
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


### Why do we need it?

In RL for LLMs (e.g., PPO, GRPO):

1. **Training engine** (on GPUs or distributed nodes) updates weights every optimization step.
2. **Inference engine** (another process or cluster) generates rollouts, samples actions, or evaluates policies, but it must use the **latest policy weights** to stay consistent with training.
3. These two components often run separately (different processes, nodes, or even different frameworks like Megatron/FSDP vs. SGLang/vLLM), so **explicit synchronization is required**.


<div class="divider"></div>



## 3. How weight sync works in slime?

![How weight sync works in slime?](/assets/slime/weight_sync/how_weight_sync_works.png)


The weight sync process involves sophisticated cross-process GPU memory sharing. Here's the detailed 5-step workflow:

1. In Megatron Worker Group, gather all tensors from distributed workers, by gathering tensors which was distibuted by PP/TP/EP/ETP in training process.
2. Serialize the Tensor into CudaIpcHandlers 
3. Call SGLang Server’s update_weight_by_tensor API
4. Scatter the CudaIpcHandlers into SGLang’s TP Workers
5. Deserialize back by rebuilding CUDA Tensors and Load Weights

<div class="divider"></div>



## 4. Our optimization journey: From 120s to 7s

![Our optimization journey](/assets/slime/weight_sync/our_optimization_journey.png)


Through this optimization journey, we've adopted many techniques that we'll discuss in detail below. And we will be using QWen3-30B-A3B model as an example for the following blog.

<div class="divider"></div>

### 4.0 Cross Process Data Transfer on GPU: CUDA IPC Handler Deep Dive

When transferring large model weights between processes, we face a fundamental challenge: how to efficiently share GigaBytes of CUDA tensor data without killing performance or memory usage.

#### Naive Approach vs CUDA IPC

![Traditional Approach vs CUDA IPC](/assets/slime/weight_sync/base64_memraid.png)


#### How CUDA IPC Works: The Magic Behind Zero-Copy Transfer

![How CUDA IPC Works](/assets/slime/weight_sync/cuda_ipc_transfer_memraid.png)


#### Key Advantages:

1. **Zero-Copy Transfer**: No actual data movement - only memory mapping
2. **Minimal Memory Overhead**: Only ~64 bytes for the IPC handle vs GBs for serialized data
3. **GPU-to-GPU Direct**: Avoids CPU-GPU memory copies entirely

This forms our baseline implementation, achieving significant improvements over traditional serialization approaches, however, it still took us 120s to sync the weight.



<div class="divider"></div>



### 4.1 Optimizing the tensor gathering process: *From 120s to 90s*

The first major bottleneck was in gathering tensors scattered across different distributed parallelism paradigms (Pipeline Parallel/Tensor Parallel/Expert Parallel).

#### The Problem
Initially, we were gathering tensors sequentially:
```python
# Slow sequential approach
for param_info in param_infos:
    if distributed.get_rank() == param_info.src_rank:
        param = weights["actor"][param_info.name]
        dist.broadcast(param, src=param_info.src_rank, group=pp_group)
```

This creates a serialization bottleneck where each parameter waits for the previous one to complete.

#### The Solution: Asynchronous Gathering
```python
# Fast async approach
handles = []
for param_info, param in zip(param_infos, params):
    if param_info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
        handle = torch.distributed.broadcast(
            param, 
            src=param_info.src_rank, 
            group=mpu.get_pipeline_model_parallel_group(), 
            async_op=True  # Key optimization!
        )
        handles.append(handle)

# Wait for all operations to complete
for handle in handles:
    handle.wait()
```

#### Performance Impact:
- **Before**: Sequential gathering → 120s
- **After**: Parallel async gathering → 90s  
- **Improvement**: 25% reduction in sync time
- **Key insight**: Maximize network bandwidth utilization by parallelizing communications

Code Reference: [slime/backends/megatron_utils/update_weight_utils.py](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/update_weight_utils.py#L59-L123)

Related PRs: [https://github.com/THUDM/slime/pull/135](https://github.com/THUDM/slime/pull/135)



<div class="divider"></div>



### 4.2 Optimizing SGLang Server Calls with Tensor Bucketing: *From 90s to 30s*

The next bottleneck was in the number of API calls to SGLang servers. Making individual HTTP requests for each tensor was causing significant overhead.

#### The Problem: Too Many Small API Calls
```python
# Inefficient: One API call per tensor
for name, tensor in named_tensors.items():
    response = requests.post(
        f"http://{server_host}/update_weights_from_tensor",
        json={"tensor_name": name, "tensor_data": serialize(tensor)}
    )
```

#### The Solution: Tensor Bucketing
```python
# Efficient: Batch tensors into 512MB buckets
def create_tensor_buckets(named_tensors, bucket_size=512 * 1024 * 1024):
    buckets = []
    current_bucket = {}
    current_size = 0
    
    for name, tensor in named_tensors.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > bucket_size:
            buckets.append(current_bucket)
            current_bucket = {}
            current_size = 0
        
        current_bucket[name] = tensor
        current_size += tensor_size
    
    if current_bucket:
        buckets.append(current_bucket)
    return buckets

# Send buckets instead of individual tensors
for bucket in tensor_buckets:
    serialized_bucket = MultiprocessingSerializer.serialize(bucket)
    response = requests.post(
        f"http://{server_host}/update_weights_from_tensor",
        json={"serialized_named_tensors": [serialized_bucket]}
    )
```

#### Performance Impact:
- **Before**: ~2000 individual API calls → 90s
- **After**: ~120 bucketed calls → 30s
- **Improvement**: 67% reduction by minimizing HTTP overhead

[Code Reference](https://github.com/THUDM/slime/blob/b738d3338aebcdc2875519d3ddeff4991010adf5/slime/backends/megatron_utils/update_weight_utils.py#L277-L293)


<div class="divider"></div>




### 4.3 Merge the tensor list into one tensor to reduce cudaipc open and close

    1. Draw a graph




<div class="divider"></div>



### 4.4 Load Weight Optimization

    1. Add PRs




<div class="divider"></div>

## 5. Key Insights and Lessons Learned

### Why Server-Based Architecture?
We chose SGLang's server-based approach over direct engine integration for several key reasons:

1. **Decoupling**: Clean separation between training and inference processes
2. **Scalability**: Router-based load balancing across multiple inference nodes  
3. **Reliability**: Easier error handling and recovery vs tight process coupling


## 6. Future Optimizations

Several exciting optimization opportunities remain:

- **Overlap Communication**: Pipeline gathering and sending operations
- **Async Weight Loading**: Non-blocking model weight updates  
- **Zero-Redundancy Layout**: Pre-calculate inference engine memory layout
- **Cross-Node CUDA IPC**: Extend beyond single-node limitations

<div class="divider"></div>

## 7. Acknowledgments

We extend our gratitude to:
- The **SLIME Team** for the innovative cross-process weight synchronization framework
- The **SGLang Team** for the high-performance inference engine and CUDA IPC support  

Special thanks to the open-source community for making these advanced ML systems accessible to researchers worldwide.

<div class="divider"></div>


## 8. References

[^1]: [LlamaRL Paper](https://arxiv.org/pdf/2505.24034)
[^2]: [Torch Memory Saver: A PyTorch library that allows tensor memory to be temporarily released and resumed later](https://github.com/fzyzcjy/torch_memory_saver)
[^3]: [CUDA 10.2: Introducing Low-Level GPU Virtual Memory Management](https://developer.nvidia.com/blog/introducing-low-level-gpu-virtual-memory-management/)
[^4]: [LD_PRELOAD](https://catonmat.net/simple-ld-preload-tutorial)