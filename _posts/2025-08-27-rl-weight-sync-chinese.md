---
title: 高效强化学习训练 - 优化slime中的权重同步
updated: 2025-08-27 11:11
lang: zh
---

<div class="authors-section" style="display: flex; justify-content: center; margin: 40px 0; gap: 40px;">
  <div style="margin: 0 0 5px 0; font-size: 18px; font-weight: 500;"><p>作者</p></div>
  <!-- Author 1 -->
  <div class="author-card" style="display: flex; flex-direction: column; align-items: center; max-width: 200px; text-align: center;">
    <h3 style="margin: 0 0 5px 0; font-size: 18px; font-weight: 500;">何标</h3>
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
    <h3 style="margin: 0 0 5px 0; font-size: 18px; font-weight: 500;">朱子霖</h3>
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
    <h3 style="margin: 0 0 5px 0; font-size: 18px; font-weight: 500;">李冀</h3>
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


## 1. 什么是slime？

[slime](https://github.com/THUDM/slime) 是一个强化学习大规模训练框架，提供两个核心能力：

- **多功能** – 拥有完全可定制的rollout接口和灵活的训练设置（同卡或分离、同步或异步、强化学习或SFT冷启动）。
- **高性能** - 原生集成SGLang进行推理和Megatron-LM进行训练。
- **易维护** - 具有轻量级代码库，并可从Megatron预训练平滑过渡到SGLang部署。[^1]

<br>
![什么是slime？](/assets/slime/weight_sync/slime_overview.png)
<br>
slime主要由三个核心模块组成[^2]：

- **训练模块（Megatron）** – 处理主要的训练过程，从数据缓冲区读取数据，并在训练后与rollout模块同步参数
- **Rollout模块（SGLang + Router）** – 生成新数据，包括奖励和sampling后的输出结果，并将其写入数据缓冲区
- **数据Buffer模块** – 作为桥接模块，管理prompt初始化、自定义数据和rollout生成策略。

<div class="divider"></div>


## 2. 什么是权重同步？

![什么是权重同步？](/assets/slime/weight_sync/what_is_weight_sync.png)
<br>

在LLM强化学习中，**权重同步**是指**将更新好的训练端的模型权重传输给到推理端**的过程，以确保推理工作节点始终使用最新的参数(On-Policy)。



### 为什么需要权重同步？

在LLM的强化学习（如PPO、GRPO）中：

1. **训练引擎**在每个`optimizer.step()`后更新策略模型权重。
2. **推理引擎**生成rollout、采样动作，但它需要使用**最新的策略模型权重**以与训练保持一致。
3. 这两个组件通常分别运行在不同的进程和不同的框架（如Megatron/FSDP vs. SGLang/vLLM），因此**需要显式同步**。

> **注意**：这篇文章专门关注同卡(Colocate)模式，我们在整个权重更新过程中使用`update_weights_from_tensor` api。在分离(Disaggregate)模式下，slime使用`update_weights_from_distributed` api，通常通过NVLink/InfiniBand互连传输权重。

<div class="divider"></div>



## 3. 权重同步在slime中如何工作？

![权重同步在slime中如何工作？](/assets/slime/weight_sync/how_weight_sync_works.png)

在slime的同卡(Colocate)模式下，**Megatron**的工作进程和**SGLang**的工作进程共同位于相同的物理GPU上。为了实现零拷贝权重传输，Megatron不发送数据本身，而是通过将Tensor序列化成CudaIpcHandlers再将其发送给SGLang的工作进程，而SGLang可以直接通过这些CudaIpcHandlers来访问权重数据进行映射，这样可以极大的提高传输效率。

以下是详细的5步工作流程：

1. **收集分布式张量**：从Megatron训练进程中的PP/TP/EP/ETP等级的分布式工作节点收集，并gather成完整的Tensor。[代码](https://github.com/THUDM/slime/blob/e943681211e2b230f2a34efd9793e1257c2d70c7/slime/backends/megatron_utils/update_weight_utils.py#L334-L399)
2. **序列化为CUDA IPC**：将Tensor转换为CudaIpcHandlers并将其聚合成一个个大约为512MB的bucket tensor中。[代码](https://github.com/THUDM/slime/blob/e943681211e2b230f2a34efd9793e1257c2d70c7/slime/backends/megatron_utils/update_weight_utils.py#L402-L416)
3. **API通信**：通过`update_weights_from_tensor` api将序列化好的CudaIpcHandlers发送到SGLang Server。[代码](https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/sglang_engine.py#L151-L171)
4. **分发到工作节点**：SGLang Server将CudaIpcHandlers分发到SGLang在各个GPU Rank上启动好的TP Worker进程。[代码](https://github.com/sgl-project/sglang/blob/5343058875a7c07ad62cfef9681f26ffbe359859/python/sglang/srt/managers/tokenizer_manager.py#L1153-L1155)
5. **重构和加载**：TP Worker将CudaIpcHandlers反序列化并进行映射，指向Megatron之前聚合好的同一片GPU地址，从而将Megatron的权重加载到SGLang中。[代码](https://github.com/sgl-project/sglang/blob/v0.5.1/python/sglang/srt/model_executor/model_runner.py#L971)



<br>

### 为什么采用基于服务器的架构？

1. **保证训推一致**。
因为线上任务肯定是用的 server based。所以 RL 这里用完全相同的配置，可以
  - 避免训出来的模型上线或者单独评测的指标不匹配
  - 可以充分复用 sglang 对 server 做的测试和性能优化

2. **为了减少用户自定义 rollout 时的心智负担**
  - 通过 server based + router，让写 rollout 就像是常规打线上服务。这样比较好配合让算法老师自定义 rollout funnction 的思路
  - 可以把 router address 对外暴露，从而让外部的 agent  环境可以随意调用 slime 内部的 sglang server，从而实现纯异步训练


<div class="divider"></div>

## 4. 我们的工作：将QWen3-30B-A3B模型的权重更新时间从60秒优化到7秒

![我们的优化之旅](/assets/slime/weight_sync/our_optimization_journey.png)



> **注意**：延迟数字是根据这个Github Issue[^3]里的所有PR做完之后往回捋出来的，以便更容易理解逻辑，实际上，我们没有按照上图所示的改进顺序进行，因为实际工作场景中自然是按照从易到难实现，而不是根据物理上传输过程中的顺序。

<div class="divider"></div>

### 4.0 GPU上的跨进程数据传输：CUDA IPC Handler

在进程间传输大型模型权重时，我们肯定想要避免将整个模型序列化成Base64这种方式然后传输，尤其在同卡情况下，这样传输效率太低，内存和延迟都会爆炸。

#### 不太现实的传统方法

![传统方法 vs CUDA IPC](/assets/slime/weight_sync/base64_memraid.png)


#### 利用CUDA IPC Handler同卡零拷贝传输

![CUDA IPC如何工作](/assets/slime/weight_sync/cuda_ipc_transfer_memraid.png)


#### 主要优势：

1. **零拷贝传输** 没有实际上在进程间传送大量的数据，而是通过内存映射
2. **最小内存开销**：CUDA IPC Handler非常小 vs 序列化数据的GB级别
3. **GPU到GPU直接传输**：避免CPU-GPU内存拷贝

这其实只是我们的Baseline实现，虽然比直接传数据要快得多，但仍然花了60秒，显然有很多优化空间。



<div class="divider"></div>



### 4.1 优化张量收集过程：*从60秒到50秒*

第一个主要瓶颈是收集分散在不同分布式并行范式（管道并行/张量并行/专家并行）中的张量。

#### 按并行类型划分的通信策略

| **并行方式** | **通信方式** | **原因** |
|-----------------|-------------------|------------|
| **张量并行 (TP)** | `all_gather` | 每个rank有部分张量 → 收集所有部分以重构完整张量 |
| **管道并行 (PP)** | `broadcast` | 源rank有完整层 → 分发到其他管道阶段 |
| **专家并行 (EP)** | `broadcast` | 源rank有完整专家 → 分发到其他专家组 |

#### 解决方案：异步张量收集/广播

在下面的代码片段中，我们以TP张量的`all_gather`为例。
```python
def async_tensor_gathering():
    # 阶段1：同时启动所有异步操作
    handles = []
    for param in tensor_parallel_params:
        handle = dist.all_gather(
            param_partitions, param.data, 
            group=tp_group, async_op=True  # 关键：非阻塞
        )
        handles.append(handle)
    
    # 阶段2：等待所有操作完成
    for handle in handles:
        handle.wait()  # 通过批量等待最大化并行性
    
    # 阶段3：所有通信完成后处理所有结果
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim in gather_tasks:
        param = torch.cat(param_partitions, dim=partition_dim)
        gathered_params.append(param)

    return gathered_params
```

#### 性能影响：
- **之前**：顺序收集 → 60秒
- **之后**：并行异步收集 → 50秒  
- **改进**：同步时间减少17%
- **关键洞察**：通过并行化通信最大化网络带宽利用率

代码参考：[slime/backends/megatron_utils/update_weight_utils.py](https://github.com/THUDM/slime/blob/main/slime/backends/megatron_utils/update_weight_utils.py#L59-L123)

**相关PR**：[https://github.com/THUDM/slime/pull/135](https://github.com/THUDM/slime/pull/135)



<div class="divider"></div>



### 4.2 通过张量分桶优化SGLang服务器调用：*从50秒到30秒*

下一个瓶颈是对SGLang服务器的API调用数量。对每个张量进行单独的HTTP请求造成了显著的开销。

#### 问题：太多小的API调用
```python
# 低效：每个张量一个API调用
for name, tensor in named_tensors.items():
    response = requests.post(
        f"http://{server_host}/update_weights_from_tensor",
        json={"tensor_name": name, "tensor_data": serialize(tensor)}
    )
```

#### 解决方案：张量分桶

关键洞察是在传输前将参数智能地分组为最优大小的桶。以下是我们的生产实现：

```python
def get_param_info_buckets(args, model) -> list[list[ParamInfo]]:
    param_infos = get_param_infos(args, model)
    param_info_buckets = [[]]
    buffer_size = 0
    
    for info in param_infos:
        # 处理不同的并行策略
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()
        param_size = info.size * tp_size

        # 当超过大小限制时创建新桶
        if buffer_size + param_size > args.update_weight_buffer_size:
            param_info_buckets.append([])
            buffer_size = 0
            
        param_info_buckets[-1].append(info)
        buffer_size += param_size
    
    return param_info_buckets

self.param_info_buckets = get_param_info_buckets(args, model)

# 发送桶而不是单个张量
for param_infos in tqdm(self.param_info_buckets, disable=rank != 0, desc="Update weights"):
    self._update_bucket_weights_from_tensor(param_infos)
```

> **注意**：通过多次实验，我们发现512MB是在内存和延迟之间平衡的最佳桶大小。

#### 性能影响：
- **之前**：约2000个单独API调用 → 50秒
- **之后**：约120个分桶调用 → 30秒
- **改进**：通过最小化HTTP开销减少40%

[代码参考](https://github.com/THUDM/slime/blob/b738d3338aebcdc2875519d3ddeff4991010adf5/slime/backends/megatron_utils/update_weight_utils.py#L277-L293)


<div class="divider"></div>



### 4.3 张量扁平化：减少CUDA IPC开销：*从30秒到20秒*

即使有了张量分桶，我们仍然面临一个重要瓶颈：**CUDA IPC句柄管理开销**。每个张量都需要自己的IPC句柄创建和清理，导致数百个昂贵的操作。

#### 问题：太多CUDA IPC操作

![太多CUDA IPC操作](/assets/slime/weight_sync/before_flatten.png)

#### 性能分析

上面的火焰图揭示了我们权重同步过程中的真正瓶颈。以下是详细分解：

| **阶段** | **持续时间** | **百分比** | **主要活动** |
|-----------|--------------|----------------|---------------------|
| **IPC句柄打开** | 22ms | 54% | CUDA IPC句柄创建和内存映射 |
| **加载权重** | 8ms | 19% | 实际权重加载和张量重构 |
| **IPC句柄关闭** | 11ms | 27% | CUDA IPC清理和资源释放 |
| **总计** | **41ms** | **100%** | SGLang中完整的权重更新周期 |


**关键发现**：**81%的时间**花费在CUDA IPC操作（打开+关闭）上，而只有**19%**用于实际权重加载。这解释了为什么张量扁平化提供如此显著的改进。

![扁平化后](/assets/slime/weight_sync/after_flatten.png)

#### 张量扁平化后的性能

| **阶段** | **持续时间** | **百分比** | **改进** |
|-----------|--------------|----------------|-----------------|
| **IPC句柄打开** | 3ms | 15% | 快86% |
| **重建** | 5ms | 25% | 张量重构的新阶段 |
| **加载权重** | 12ms | 60% | 轻微变化 |
| **IPC句柄关闭** | 200μs | 1% | 快98% |
| **总计** | **20ms** | **100%** | **相比无扁平化的41ms改进51%** |

**关键成就**：通过扁平化张量，我们将IPC操作从总时间的**81%**减少到**16%**，而权重加载在**60%**时成为主导阶段 - 这正是我们想要的！


有关如何实现张量扁平化等技术细节，请参考以下PR：

相关PR： 
- [SGLang FlattenedTensorBucket实现](https://github.com/sgl-project/sglang/pull/8079)
- [SLIME集成和测试](https://github.com/THUDM/slime/pull/156)

<div class="divider"></div>



### 4.4 加载权重优化：最终性能提升：*从20秒到7秒*

在优化IPC开销后，我们识别了权重加载过程本身的其他瓶颈，特别是对于MoE模型。

#### 关键优化：

**1. 参数字典缓存**
```python
# 之前：每次权重更新时昂贵的模型遍历
params_dict = dict(self.named_parameters())

# 之后：缓存参数字典
if not hasattr(self, "_cached_params_dict"):
    self._cached_params_dict = dict(self.named_parameters())
params_dict = self._cached_params_dict
```

**2. 专家映射GPU迁移优化**  
```python
# 避免专家映射的重复GPU到CPU同步
if self.expert_map_cpu is not None and self.expert_map_gpu is None:
    # 将专家映射移动到GPU一次并缓存
    self.expert_map_gpu = self.expert_map_cpu.to(device="cuda")
```

**3. CUDA设备缓存**
```python
# 缓存CUDA设备查询以避免重复的昂贵调用
@lru_cache(maxsize=8)
def get_device(device_id: Optional[int] = None) -> str:
    # 缓存的设备查找消除了重复的torch.cuda.is_available()调用
```

#### 性能影响：
- **之前**：各种权重加载瓶颈 → 20秒
- **之后**：优化的参数缓存和设备处理 → 7秒
- **改进**：最终权重加载时间减少65%

#### 关键洞察：
- SGLang中的大多数性能优化都专注于前向传播，而权重加载优化受到的关注较少。这创造了大量低垂果实机会，正如上述PR所展示的。

相关PR： 
- [移除QWen3 MOE加载权重开销](https://github.com/sgl-project/sglang/pull/8751)
- [避免专家映射GPU到CPU设备同步](https://github.com/sgl-project/sglang/pull/8753)
- [缓存Cuda设备](https://github.com/sgl-project/sglang/pull/8996)


<div class="divider"></div>


## 5. 未来优化

仍有几个令人兴奋的优化机会：

- **重叠通信**：流水线收集和发送操作
- **异步权重加载**：非阻塞模型权重更新  
- **零冗余布局**：预计算推理引擎内存布局并进行零冗余拷贝

<div class="divider"></div>

## 6. 致谢

我们感谢：
- **slime团队**提供轻量且强大的强化学习训练框架
- **SGLang团队**提供高性能推理引擎和权重同步的基础工作。


<div class="divider"></div>

## 7. 参考文献

[^1]: [slime: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/)
[^2]: [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/pdf/2508.06471)
[^3]: [slime Issue #132: Weight Sync Optimization in Colocate Mode](https://github.com/THUDM/slime/issues/132)
