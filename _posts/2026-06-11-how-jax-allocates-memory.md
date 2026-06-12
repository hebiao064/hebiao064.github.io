---
title: How JAX Allocates Memory
updated: 2026-06-11 23:33
---

When people first run a JAX program on GPU, one thing often looks surprising:

```bash
nvidia-smi
```

may show that the Python process has already taken a large chunk of GPU memory, even before the model looks large enough to justify it. That does not necessarily mean the model has a leak. It is usually JAX's allocator doing exactly what it was designed to do.

This post is a mental model for how JAX thinks about memory: what is reserved, what is actually used by arrays, what happens inside `jit`, and how to debug OOMs without being misled by `nvidia-smi`.

## The Short Version

JAX memory behavior is easier to reason about if we separate four layers:

1. **Python objects**: `jax.Array` values in Python are handles to buffers and computations.
2. **Device buffers**: array data usually lives on GPU/TPU device memory.
3. **XLA programs**: `jax.jit` lowers computations to XLA, which plans temporaries and can reuse buffers inside a compiled computation.
4. **The device allocator**: on GPU, JAX normally reserves a large memory pool early and serves later allocations from that pool.

The most user-visible policy is the last one: by default, JAX preallocates 75% of total GPU memory when the first JAX operation runs. The goal is to reduce allocation overhead and fragmentation, not to hold 75% worth of live arrays.

## Why JAX Preallocates GPU Memory

GPU allocation is not free. If every intermediate array had to call into the CUDA allocator independently, a training loop would pay repeated allocation overhead and could fragment memory over time.

So JAX's default GPU behavior is pool-like:

```text
first JAX GPU op
      |
      v
reserve a large GPU memory region
      |
      v
reuse pieces of that region for arrays, temporaries, compiled computations
```

The official JAX docs describe the default as preallocating 75% of total GPU memory on the first JAX operation. That means `nvidia-smi` reports the reservation, while JAX may only be actively using part of it.

This distinction matters:

```text
reserved by process != live model/activation memory
```

If you delete a `jax.Array`, the backing buffer can be returned to JAX's allocator for reuse. But with the default allocator, it usually is not returned to the operating system immediately, so `nvidia-smi` may not go down.

## The Three Common Allocator Knobs

These environment variables should be set before the JAX backend initializes; in practice, set them before running Python.

### 1. Disable preallocation

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train.py
```

JAX allocates GPU memory as needed instead of grabbing the default pool up front. This can help if multiple processes must share one GPU. The tradeoff is that fragmentation becomes more likely, so a job that uses most of the GPU can still OOM later.

### 2. Change the preallocation fraction

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.50 python train.py
```

This keeps the pool behavior but changes its size. It is useful when running several JAX processes on the same GPU, or when another framework also needs GPU memory.

### 3. Use the platform allocator

```bash
XLA_PYTHON_CLIENT_ALLOCATOR=platform python train.py
```

This asks JAX to allocate closer to exactly what it needs and deallocate memory instead of keeping it for reuse. It is much slower and is mainly useful for debugging, measuring a minimal footprint, or confirming whether an OOM is caused by allocator reservation.

## What Happens Inside `jax.jit`

Outside `jit`, it is tempting to imagine each JAX line as:

```python
a = x + y
b = a * 2
c = b.sum()
```

allocating `a`, then `b`, then `c` exactly as written. But under `jax.jit`, that Python program is traced into an XLA program. XLA sees the computation as a graph, not as a sequence of Python assignment statements.

That gives the compiler room to:

- fuse operations so some intermediates never materialize as full arrays;
- reuse temporary buffers whose lifetimes do not overlap;
- choose layouts and buffer assignments for the compiled computation;
- keep outputs alive only if Python still has references to them.

This is one reason JAX can be memory efficient even though arrays are immutable at the Python level. The Python API is functional, but the compiled program can still reuse storage internally when it is safe.

## Asynchronous Dispatch Changes What "Alive" Means

JAX dispatches work asynchronously. When you run:

```python
y = jnp.dot(x, x)
```

Python often gets a `jax.Array` result handle before the device has finished the computation. That array is a future-like value: shape and dtype are known, but the data may still be in flight.

This affects memory debugging. If you measure immediately after launching work, you may be measuring queued work, not completed work. For accurate timing or memory snapshots, force synchronization:

```python
y.block_until_ready()
```

The same idea applies before taking a device memory profile.

## Host Memory, Device Memory, and Offloading

By default, array data used for accelerator computation lives in device memory. Moving values between host and device is explicit or implicit depending on the operation:

```python
x = jax.device_put(x_np)  # host -> device
y_np = np.array(y)        # device -> host, blocks until ready
```

Recent JAX APIs also expose more explicit placement controls. For example, sharding can carry a `memory_kind`, such as device memory or pinned host memory, which is useful for host offloading strategies. That matters for large training jobs where not all parameters, optimizer state, or activations need to be resident on device at the same time.

The high-level rule is:

```text
device memory is fast and scarce;
host memory is larger but transfer costs matter.
```

Offloading is not free memory; it is a latency/bandwidth tradeoff.

## Buffer Donation

JAX arrays are immutable from Python's point of view. But for training steps, we often have values that are logically consumed and replaced:

```python
params, opt_state = train_step(params, opt_state, batch)
```

If the old `params` or `opt_state` will not be used after the call, we can donate their buffers:

```python
train_step = jax.jit(train_step, donate_argnums=(0, 1))
```

Donation tells JAX/XLA that the input buffer may be reused for an output. This can reduce peak memory at the boundary of a compiled computation. After donating an input, do not use the old object again; JAX will treat it as invalid.

Donation is especially useful for large training states where every step returns a new state with the same shapes.

## Common OOM Patterns

### Multiple processes on one GPU

If two JAX processes each try to preallocate 75%, the second one may fail immediately. Set a smaller memory fraction per process:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=.45 python worker_a.py
XLA_PYTHON_CLIENT_MEM_FRACTION=.45 python worker_b.py
```

### JAX plus TensorFlow or PyTorch

If another framework is using the same GPU, the default JAX pool can look too aggressive. Lower `XLA_PYTHON_CLIENT_MEM_FRACTION` or disable preallocation.

### Keeping arrays in Python containers

This is a classic leak-shaped pattern:

```python
history = []
for step in range(num_steps):
    loss, activations = f(...)
    history.append(activations)
```

As long as Python keeps references to device arrays, their buffers stay alive.

### Shape changes causing recompilation

If input shapes change often, JAX may compile many variants. Compiled executables and constants can also consume memory. Static shapes are not just a performance detail; they help keep compilation and memory behavior predictable.

## How I Usually Debug JAX Memory

1. **Start with the allocator policy**

   Check whether the process is using the default preallocation. If sharing a GPU, try:

   ```bash
   XLA_PYTHON_CLIENT_MEM_FRACTION=.5 python repro.py
   ```

2. **Synchronize before measuring**

   Add:

   ```python
   result.block_until_ready()
   ```

   before timing or taking snapshots.

3. **Use the memory profiler**

   JAX can save a device memory profile:

   ```python
   import jax.profiler

   result.block_until_ready()
   jax.profiler.save_device_memory_profile("memory.prof")
   ```

   The profile can show which Python stack allocated live buffers. For newer workflows, the JAX docs recommend using XProf's `memory_viewer` for device memory analysis.

4. **Look for Python references**

   Lists, closures, cached metrics, and debug outputs often keep arrays alive longer than expected.

5. **Use donation for training state**

   Donate large state buffers when the old state is truly dead after the step.

## A Practical Mental Model

For most production training or inference jobs, I think about JAX memory like this:

```text
JAX reserves a GPU pool early.
Python jax.Array objects keep device buffers alive.
jit lets XLA plan and reuse temporary buffers inside compiled programs.
asynchronous dispatch means "returned to Python" does not mean "finished on device".
donation and offloading are explicit tools for reducing peak device pressure.
```

So when `nvidia-smi` says JAX is using a lot of memory, the first question is not "where did my model allocate all of that?" It is:

```text
Is this reserved allocator memory, or live array memory?
```

Once that distinction is clear, most JAX OOMs become much less mysterious.

## References

- [JAX: GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html)
- [JAX: Asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)
- [JAX: Buffer donation](https://docs.jax.dev/en/latest/buffer_donation.html)
- [JAX: Profiling device memory](https://docs.jax.dev/en/latest/device_memory_profiling.html)
- [JAX: Memories and host offloading](https://docs.jax.dev/en/latest/notebooks/host-offloading.html)
