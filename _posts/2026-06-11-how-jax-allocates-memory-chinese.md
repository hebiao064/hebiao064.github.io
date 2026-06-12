---
title: JAX 是怎么分配内存的
updated: 2026-06-11 23:33
lang: zh
---

第一次在 GPU 上跑 JAX 的时候，很多人都会被这个现象吓一下：

```bash
nvidia-smi
```

里显示 Python 进程一下子占了很大一块显存，好像模型还没多大，显存就已经被吃掉了。

这通常不是内存泄漏，而是 JAX 的 allocator 在按默认策略工作。

这篇文章想建立一个更准确的 mental model：JAX 什么时候只是“预留”显存，什么时候是真的有 live array，占用为什么不会立刻还给系统，`jit` 里面又是怎么复用 buffer 的。

## 一句话版本

理解 JAX 内存，最好把它拆成四层：

1. **Python object**：Python 里的 `jax.Array` 更像一个指向 device buffer 和异步计算结果的 handle。
2. **Device buffer**：真正的数组数据通常放在 GPU/TPU device memory 上。
3. **XLA program**：`jax.jit` 会把计算 lower 给 XLA，由 XLA 规划中间结果和 buffer 复用。
4. **Device allocator**：在 GPU 上，JAX 默认会提前预留一大块显存，然后后续从这个 pool 里分配。

最容易被看到的是第四层：JAX 默认会在第一次 JAX operation 运行时预分配整张 GPU 75% 的显存。这个行为是为了减少 allocation overhead 和 memory fragmentation，不代表此时真的有 75% 的显存都被模型参数或者 activation 用掉了。

## 为什么 JAX 要预分配 GPU 显存

GPU allocation 并不便宜。如果每个中间 tensor 都单独向 CUDA allocator 要显存，训练循环会反复支付分配开销，长期运行还更容易产生碎片。

所以 JAX 在 GPU 上的默认行为更像这样：

```text
第一次 JAX GPU op
      |
      v
预留一大块 GPU memory
      |
      v
后续 array、temporary、compiled computation 从这块 pool 里复用
```

JAX 官方文档明确写到，默认情况下第一次 JAX operation 会预分配 75% 的 GPU memory。因此 `nvidia-smi` 看到的是进程 reservation，而不是 live tensor 的精确大小。

这个区别很重要：

```text
process reserved memory != live model / activation memory
```

当你 `del x` 删除一个 `jax.Array` 后，它背后的 buffer 可以回到 JAX allocator 里继续复用。但默认 allocator 通常不会马上把显存还给操作系统，所以 `nvidia-smi` 里的数字不一定下降。

## 三个常用 allocator 开关

这些环境变量要在 JAX backend 初始化之前设置；实践中最稳妥的方式是在启动 Python 之前设置。

### 1. 关闭预分配

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py
```

JAX 不再启动时抢默认 pool，而是按需申请显存。这个设置适合同一张 GPU 上有多个进程共享显存的场景。代价是更容易出现 memory fragmentation，所以如果任务本身会吃掉大部分显存，后面仍然可能 OOM。

### 2. 调整预分配比例

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.50 python train.py
```

这个设置保留 pool allocator 的行为，但调整 pool 的大小。比如同一张卡上跑两个 JAX process，可以给每个 process 分一个合理比例。

### 3. 使用 platform allocator

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=platform python train.py
```

这个模式会让 JAX 尽量按实际需要申请和释放内存，而不是留在 pool 里复用。它会慢很多，所以不适合日常训练，但很适合 debug：比如确认一个 OOM 到底是预留策略导致的，还是 live array 真的太多。

## `jax.jit` 里面发生了什么

在 `jit` 外面，我们容易把代码想象成逐行分配：

```python
a = x + y
b = a * 2
c = b.sum()
```

好像先分配 `a`，再分配 `b`，最后分配 `c`。但在 `jax.jit` 里面，这段 Python 会先被 trace 成一个 XLA program。XLA 看到的是计算图，不是 Python 赋值语句本身。

这给了 compiler 很多优化空间：

- fuse 多个 op，让一些中间结果不需要完整 materialize；
- 复用 lifetime 不重叠的 temporary buffer；
- 为 compiled computation 选择 layout 和 buffer assignment；
- 只要 Python 还持有输出引用，就保持对应 buffer alive。

这也是为什么 JAX 在 Python API 上是 immutable array，但 compiled program 里面仍然可以做安全的 storage reuse。函数式语义和底层 buffer 复用并不矛盾。

## 异步 dispatch 会改变你对“还活着”的判断

JAX 默认异步 dispatch。比如：

```python
y = jnp.dot(x, x)
```

Python 很可能很快就拿到一个 `jax.Array` 返回值，但 device 上的 matmul 还没真的算完。这个 `jax.Array` 更像一个 future：shape 和 dtype 已经知道了，但数据可能还在 device 上执行。

这会影响 memory debug。如果你在 dispatch 后立刻测量，看到的可能是 queued work，而不是已经完成的 work。需要准确测量时间或者保存 memory profile 时，最好先同步：

```python
y.block_until_ready()
```

## Host Memory、Device Memory 和 Offloading

默认情况下，accelerator 计算使用的数据在 device memory 上。host 和 device 之间的数据移动有时显式，有时隐式：

```python
x = jax.device_put(x_np)  # host -> device
y_np = np.array(y)        # device -> host，会等待 y ready
```

较新的 JAX API 也提供了更明确的 placement 控制。比如 sharding 可以带 `memory_kind`，选择 device memory 或 pinned host memory。这对 host offloading 很重要：大模型训练时，不是所有参数、optimizer state、activation 都必须同时留在 device 上。

一个实用规则是：

```text
device memory 快但稀缺；
host memory 大但搬运有 latency / bandwidth 成本。
```

Offloading 不是免费显存，它是在显存压力和传输开销之间做 tradeoff。

## Buffer Donation

从 Python 视角看，JAX array 是 immutable 的。但训练 step 里，我们经常有这种模式：

```python
params, opt_state = train_step(params, opt_state, batch)
```

旧的 `params` 和 `opt_state` 在 step 之后不会再使用，只需要新的状态。这个时候可以 donation：

```python
train_step = jax.jit(train_step, donate_argnums=(0, 1))
```

Donation 的意思是告诉 JAX/XLA：这些 input buffer 之后不用了，可以拿去装 output。这样能降低 compiled computation 边界处的 peak memory。被 donate 的对象不要再使用，JAX 会把它视为 invalid。

对于大型 training state，donation 往往很有用，因为每一步输入和输出的 shape 基本一致。

## 常见 OOM 模式

### 多个 process 共享同一张 GPU

如果两个 JAX process 都想预分配 75%，第二个 process 很可能启动就 OOM。可以给每个 process 设置更小比例：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.45 python worker_a.py
XLA_PYTHON_CLIENT_MEM_FRACTION=.45 python worker_b.py
```

### JAX 和 TensorFlow / PyTorch 共用 GPU

如果另一个框架也在同一张卡上占显存，JAX 默认 pool 会显得很激进。可以降低 `XLA_PYTHON_CLIENT_MEM_FRACTION`，或者关闭 preallocation。

### Python 容器一直持有 array 引用

这是很常见的“看起来像 leak”的模式：

```python
history = []
for step in range(num_steps):
    loss, activations = f(...)
    history.append(activations)
```

只要 Python 还持有 device array 的引用，对应 buffer 就不会释放。

### Shape 变化导致反复编译

如果输入 shape 经常变，JAX 可能编译很多个 variant。Compiled executable 和常量也会占内存。静态 shape 不只是性能问题，也会让 memory 行为更可预测。

## 我通常怎么 debug JAX 显存

1. **先确认 allocator policy**

   如果是共享 GPU，先试：

   ```bash
   XLA_PYTHON_CLIENT_MEM_FRACTION=.5 python repro.py
   ```

2. **测量前同步**

   在计时或者抓 profile 前加：

   ```python
   result.block_until_ready()
   ```

3. **用 memory profiler**

   JAX 可以保存 device memory profile：

   ```python
   import jax.profiler

   result.block_until_ready()
   jax.profiler.save_device_memory_profile("memory.prof")
   ```

   这个 profile 可以帮助定位 live buffer 来自哪个 Python stack。JAX 文档现在也推荐用 XProf 的 `memory_viewer` 做 device memory 分析。

4. **找 Python reference**

   list、closure、缓存的 metric、debug output 都可能让 array 活得比你想象中更久。

5. **对 training state 使用 donation**

   如果旧 state 在 step 之后真的不会再用，就把大的 state buffer donate 掉。

## 一个实用 mental model

大多数训练和推理场景里，我会这样理解 JAX 显存：

```text
JAX 启动后会较早预留 GPU memory pool。
Python 里的 jax.Array 引用决定 device buffer 是否还 live。
jit 让 XLA 在 compiled program 内部规划和复用 temporary buffer。
异步 dispatch 意味着“Python 返回了”不等于“device 算完了”。
donation 和 offloading 是降低 peak device memory 的显式工具。
```

所以当 `nvidia-smi` 显示 JAX 占了很多显存时，第一个问题不是“我的模型哪里分配了这么多”，而是：

```text
这是 allocator reserved memory，还是 live array memory？
```

把这个问题分清楚，JAX 的很多 OOM 就没那么神秘了。

## 参考

- [JAX: GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
- [JAX: Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)
- [JAX: Buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html)
- [JAX: Profiling device memory](https://docs.jax.dev/en/latest/device_memory_profiling.html)
- [JAX: Memories and host offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html)
