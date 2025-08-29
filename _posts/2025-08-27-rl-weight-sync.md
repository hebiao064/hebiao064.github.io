---
title: Draft - Efficient RL Training - Optimizing Weight Sync in slime
updated: 2025-08-27 11:11
---

### Status: Under Peer Review, ETA: 2025-09-01

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
      <a href="https://www.linkedin.com/in/ji-li-a4892623b/" target="_blank" style="text-decoration: none; border: none;">
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

[slime](https://github.com/THUDM/slime) is a LLM post-training framework aiming for RL Scaling, it was designed to be:

- **Versatile** – with a fully customizable rollout interface and flexible training setups (colocated or decoupled, synchronous or asynchronous, RL or SFT cold start).
- **Performant** - integrating SGLang for inference and Megatron-LM for training, natively.
- **Maintainable** - with a lightweight codebase and smooth transition from Megatron pretraining to SGLang deployment.[^1]

<br>
![What is slime?](/assets/slime/weight_sync/slime_overview.png)
<br>
The system consists of three core modules[^2]:

- **Training (Megatron)** – handles the main training process, reads data from the Data Buffer, and
synchronizes parameters with the rollout module after training
- **Rollout (SGLang + Router)** –
generates new data, including rewards and verifier outputs, and writes it to the Data Buffer
- **Data Buffer** – serves as a bridge module that manages prompt initialization, custom data, and rollout
generation strategies.

<div class="divider"></div>


## 2. What is Weight Sync?

![What is weight sync?](/assets/slime/weight_sync/what_is_weight_sync.png)
<br>

**Weight sync** in LLM reinforcement learning (RL) refers to the process of **copying updated model weights from the training side to the inference side** so that inference workers always use up-to-date parameters. 


### Why do we need it?

In RL for LLMs (e.g., PPO, GRPO):

1. **Training engine** updates weights every optimization step.
2. **Inference engine** generates rollouts, samples actions, but it needs to use the **latest policy weights** to stay consistent with training.
3. These two components often run separately (different processes and different frameworks like Megatron/FSDP vs. SGLang/vLLM), so **explicit synchronization is required**.


<div class="divider"></div>



## 3. How weight sync works in slime?

![How weight sync works in slime?](/assets/slime/weight_sync/how_weight_sync_works.png)


The weight sync process involves sophisticated cross-process GPU memory sharing. Here's the detailed 5-step workflow:

1. **Gather distributed tensors**: Collect model weights from distributed workers across PP/TP/EP/ETP ranks in the Megatron training process. [Code](https://github.com/THUDM/slime/blob/e943681211e2b230f2a34efd9793e1257c2d70c7/slime/backends/megatron_utils/update_weight_utils.py#L334-L399)
2. **Serialize to CUDA IPC**: Convert tensors into CUDA IPC handlers and aggregate them into transfer-ready buckets. [Code](https://github.com/THUDM/slime/blob/e943681211e2b230f2a34efd9793e1257c2d70c7/slime/backends/megatron_utils/update_weight_utils.py#L402-L416)
3. **API communication**: Send serialized tensor data to SGLang server via the `update_weights_from_tensor` endpoint. [Code](https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/sglang_engine.py#L151-L171)
4. **Distribute to workers**: Scatter CUDA IPC handlers across SGLang's tensor parallel workers. [Code](https://github.com/sgl-project/sglang/blob/5343058875a7c07ad62cfef9681f26ffbe359859/python/sglang/srt/managers/tokenizer_manager.py#L1153-L1155)
5. **Reconstruct and load**: Deserialize CUDA IPC handlers back to tensors and load the updated weights into the inference model. [Code](https://github.com/sgl-project/sglang/blob/v0.5.1/python/sglang/srt/model_executor/model_runner.py#L971)

> **Note**: This blog focuses exclusively on Colocated Mode, where we utilize the `update_weights_from_tensor` endpoint throughout our optimization journey. In disaggregated mode, slime uses the `update_weights_from_distributed` endpoint, which transfers weights through NVLink/InfiniBand interconnects.

<br>

### Why Server-Based Architecture?

1.  **Ensuring Consistency Between Training and Inference.** Since online tasks will undoubtedly use a server-based setup, using the exact same configuration for RL training can:
    * Prevent discrepancies between the model's performance during training and its metrics when deployed or evaluated in a standalone test.
    * Allow full reuse of the tests and performance optimizations already made for the server with SGLang.

2.  **Reducing the Mental Burden of Custom Rollouts.**
    * By using server-based engines with a router, writing rollouts becomes similar to calling a regular online service. This makes it easier for users to define custom rollout functions.
    * The router's address can be exposed externally, allowing outside agent environments to freely call the internal SGLang server. This enables a purely asynchronous training approach.


<div class="divider"></div>

## 4. Our optimization journey: From 60s to 7s

![Our optimization journey](/assets/slime/weight_sync/our_optimization_journey.png)


Through this optimization journey, we've adopted many techniques that we'll discuss in detail below. And we will be using QWen3-30B-A3B model as an example for the following blog.

> **Note**: The latency number was simulated according to the series of PRs[^3] to make it easier to understand the logic, in reality, we didn't follow the order of improvement like the graph shown above. And reproducible setup can be found [here](https://gist.github.com/hebiao064/335ac5b44237af8f9514bb37fb216035) with 8 H100 GPUs.

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

This forms our baseline implementation, achieving significant improvements over traditional serialization approaches, however, it still took us 60s to sync the weight.



<div class="divider"></div>



### 4.1 Optimizing the tensor gathering process: *From 60s to 50s*

The first major bottleneck was in gathering tensors scattered across different distributed parallelism paradigms (Pipeline Parallel/Tensor Parallel/Expert Parallel).

#### The Solution: Async Tensor Gathering
```python
def async_tensor_gathering():
    # Phase 1: Start all async operations simultaneously
    handles = []
    for param in tensor_parallel_params:
        handle = dist.all_gather(
            param_partitions, param.data, 
            group=tp_group, async_op=True  # Key: non-blocking
        )
        handles.append(handle)
    
    # Phase 2: Wait for ALL operations to complete
    for handle in handles:
        handle.wait()  # Maximize parallelism by batching waits
    
    # Phase 3: Process all results after all communications are done
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim in gather_tasks:
        param = torch.cat(param_partitions, dim=partition_dim)
        gathered_params.append(param)

    return gathered_params
```

#### Performance Impact:
- **Before**: Sequential gathering → 60s
- **After**: Parallel async gathering → 50s  
- **Improvement**: 17% reduction in sync time
- **Key insight**: Maximize network bandwidth utilization by parallelizing communications

Code Reference: [slime/backends/megatron_utils/update_weight_utils.py](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/update_weight_utils.py#L59-L123)

**Related PRs**: [https://github.com/THUDM/slime/pull/135](https://github.com/THUDM/slime/pull/135)



<div class="divider"></div>



### 4.2 Optimizing SGLang Server Calls with Tensor Bucketing: *From 50s to 30s*

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

The key insight is to intelligently group parameters into optimally-sized buckets before transmission. Here's our production implementation:

```python
def get_param_info_buckets(args, model) -> list[list[ParamInfo]]:
    param_infos = get_param_infos(args, model)
    param_info_buckets = [[]]
    buffer_size = 0
    
    for info in param_infos:
        # Handle different parallelism strategies
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()
        param_size = info.size * tp_size

        # Create new bucket when size limit exceeded
        if buffer_size + param_size > args.update_weight_buffer_size:
            param_info_buckets.append([])
            buffer_size = 0
            
        param_info_buckets[-1].append(info)
        buffer_size += param_size
    
    return param_info_buckets

self.param_info_buckets = get_param_info_buckets(args, model)

# Send buckets instead of individual tensors
for param_infos in tqdm(self.param_info_buckets, disable=rank != 0, desc="Update weights"):
    self._update_bucket_weights_from_tensor(param_infos)
```

> **Note**: From serveral experiments, we found that 512MB is the optimal bucket size for the balance of memory and latency.

#### Performance Impact:
- **Before**: ~2000 individual API calls → 50s
- **After**: ~120 bucketed calls → 30s
- **Improvement**: 40% reduction by minimizing HTTP overhead

[Code Reference](https://github.com/THUDM/slime/blob/b738d3338aebcdc2875519d3ddeff4991010adf5/slime/backends/megatron_utils/update_weight_utils.py#L277-L293)


<div class="divider"></div>



### 4.3 Tensor Flattening: Reducing CUDA IPC Overhead: *From 30s to 20s*

Even with tensor bucketing, we still faced a significant bottleneck: **CUDA IPC handle management overhead**. Each tensor required its own IPC handle creation and cleanup, leading to hundreds of expensive operations.

#### The Problem: Too Many CUDA IPC Operations

![Too Many CUDA IPC Operations](/assets/slime/weight_sync/before_flatten.png)

#### Performance Profiling Analysis

The flame graph above reveals the true bottleneck in our weight synchronization process. Here's the breakdown:

| **Phase** | **Duration** | **Percentage** | **Main Activities** |
|-----------|--------------|----------------|---------------------|
| **IPC Handler Open** | 22ms | 54% | CUDA IPC handle creation and memory mapping |
| **Load Weights** | 8ms | 19% | Actual weight loading and tensor reconstruction |
| **IPC Handler Close** | 11ms | 27% | CUDA IPC cleanup and resource deallocation |
| **Total** | **41ms** | **100%** | Complete weight update cycle in SGLang |


**Critical Finding**: **81% of the time** is spent on CUDA IPC operations (open + close), while only **19%** is used for actual weight loading. This explains why tensor flattening provides such dramatic improvements.

![After Flatten](/assets/slime/weight_sync/after_flatten.png)

#### Performance After Tensor Flattening

| **Phase** | **Duration** | **Percentage** | **Improvement** |
|-----------|--------------|----------------|-----------------|
| **IPC Handler Open** | 3ms | 15% | 86% faster |
| **Rebuild** | 5ms | 25% | New phase for tensor reconstruction |
| **Load Weights** | 12ms | 60% | Small Variance |
| **IPC Handler Close** | 200μs | 1% | 98% faster |
| **Total** | **20ms** | **100%** | **51% improvement vs 41ms without flattening** |

**Key Achievement**: By flattening tensors, we reduced IPC operations from **81%** to **16%** of total time, while weight loading became the dominant phase at **60%** - exactly what we want!


For technical details such as how to implement the tensor flattening, please refer to the following PRs:

Related PRs: 
- [SGLang FlattenedTensorBucket Implementation](https://github.com/sgl-project/sglang/pull/8079)
- [SLIME Integration and Testing](https://github.com/THUDM/slime/pull/156)

<div class="divider"></div>



### 4.4 Load Weight Optimization: Final Performance Gains: *From 20s to 7s*

After optimizing the IPC overhead, we identified additional bottlenecks in the weight loading process itself, particularly for MoE models.

#### Key Optimizations:

**1. Parameter Dictionary Caching**
```python
# Before: Expensive model traversal on every weight update
params_dict = dict(self.named_parameters())

# After: Cache the parameter dictionary
if not hasattr(self, "_cached_params_dict"):
    self._cached_params_dict = dict(self.named_parameters())
params_dict = self._cached_params_dict
```

**2. Expert Map GPU Migration Optimization**  
```python
# Avoid repeated GPU-to-CPU synchronization for expert mapping
if self.expert_map_cpu is not None and self.expert_map_gpu is None:
    # Move expert map to GPU once and cache it
    self.expert_map_gpu = self.expert_map_cpu.to(device="cuda")
```

**3. CUDA Device Caching**
```python
# Cache CUDA device queries to avoid repeated expensive calls
@lru_cache(maxsize=8)
def get_device(device_id: Optional[int] = None) -> str:
    # Cached device lookup eliminates repeated torch.cuda.is_available() calls
```

#### Performance Impact:
- **Before**: Various weight loading bottlenecks → 20s
- **After**: Optimized parameter caching and device handling → 7s
- **Improvement**: 65% reduction in final weight loading time

#### Key Insights:
- Most performance optimizations in SGLang focus on the forward pass, while weight loading optimization receives less attention. This creates abundant low-hanging fruit opportunities, as demonstrated by the PRs above.

Related PRs: 
- [Remove QWen3 MOE Load Weight overhead](https://github.com/sgl-project/sglang/pull/8751)
- [Avoid Expert Map GPU-to-CPU Device Sync](https://github.com/sgl-project/sglang/pull/8753)
- [Cache Cuda Device](https://github.com/sgl-project/sglang/pull/8996)


<div class="divider"></div>


## 5. Future Optimizations

Several exciting optimization opportunities remain:

- **Overlap Communication**: Pipeline gathering and sending operations
- **Async Weight Loading**: Non-blocking model weight updates  
- **Zero-Redundancy Layout**: Pre-calculate inference engine memory layout and do zero-redundancy copy

<div class="divider"></div>

## 6. Acknowledgments

We extend our gratitude to:
- The **slime Team** for the light and power RL training framework
- The **SGLang Team** for the high-performance inference engine and foundation work of weights sync.


<div class="divider"></div>

## 7. References

[^1]: [slime: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/)
[^2]: [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/pdf/2508.06471)
[^3]: [slime Issue #132: Weight Sync Optimization in Colocate Mode](https://github.com/THUDM/slime/issues/132)