# Adapted from https://zhuanlan.zhihu.com/p/700224642

import torch
from contextlib import contextmanager
import os

@contextmanager
def graph_capture(pool=None, stream=None, capture_error_mode: str = "global", dump_path=None):
    g = torch.cuda.CUDAGraph()
    if dump_path is not None:
        g.enable_debug_mode()
    with torch.cuda.graph(cuda_graph=g, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
        yield g
    if dump_path is not None:
        # Expand the tilde and ensure the directory exists
        expanded_path = os.path.expanduser(dump_path)
        if os.path.isdir(expanded_path):
            # If it's a directory, create a filename
            dump_file = os.path.join(expanded_path, "cuda_graph_debug.dot")
        else:
            dump_file = expanded_path
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
        print(f"Dumping CUDA graph debug info to: {dump_file}")
        g.debug_dump(dump_file)

import ctypes

# Load the CUDA runtime library
cudart = ctypes.CDLL('libcudart.so')

# Define cudaMemcpyKind enumeration as in the CUDA API
cudaMemcpyHostToHost = 0
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaMemcpyDefault = 4

# Setup the prototype of the cudaMemcpyAsync function
cudaMemcpyAsync = cudart.cudaMemcpyAsync
cudaMemcpyAsync.argtypes = [
    ctypes.c_void_p,          # void* dst
    ctypes.c_void_p,          # const void* src
    ctypes.c_size_t,          # size_t count
    ctypes.c_int,             # enum cudaMemcpyKind
    ctypes.c_void_p           # cudaStream_t stream
]
cudaMemcpyAsync.restype = ctypes.c_int


MAX_BATCHSIZE = 128

# Placeholder input used for capture
static_a = torch.zeros((MAX_BATCHSIZE, 1024), device="cpu").pin_memory()
static_b = torch.zeros((MAX_BATCHSIZE, 1024), device="cpu").pin_memory()
static_output = torch.zeros((MAX_BATCHSIZE, 1024), device="cpu").pin_memory()

def compute(batchsize):
    a = static_a[:batchsize].to("cuda", non_blocking=True)
    b = static_b[:batchsize].to("cuda", non_blocking=True)
    output = (a ** 2 + b * 2)
    result = cudaMemcpyAsync(static_output.data_ptr(), output.data_ptr(), output.numel() * output.element_size(), cudaMemcpyDeviceToHost, torch.cuda.current_stream().cuda_stream)
    assert result == 0
    return static_output[:batchsize]

# Warmup before capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(1, MAX_BATCHSIZE + 1):
        compute(i)
torch.cuda.current_stream().wait_stream(s)

def report_memory(prefix):
    free, total = torch.cuda.mem_get_info()
    used = total - free
    print(f"{prefix}: Used: {used / 1024 / 1024} MB, Free: {free / 1024 / 1024} MB, Total: {total / 1024 / 1024} MB")

# Captures the graph
# To allow capture, automatically sets a side stream as the current stream in the context
report_memory("Before capture")
graphs = []  # Fix: Remove the integer placeholder, start with empty list
memory_pool = None
for i in range(1, MAX_BATCHSIZE + 1):
    with graph_capture(pool=memory_pool, dump_path=f"~/cuda_graph_batch_{i}.dot") as g:
        compute(i)
    graphs.append(g)
    memory_pool = g.pool()
report_memory("After capture")
# Run the graph - Fix: Use index 1 since we removed the placeholder at index 0
static_a[:2] += 1
static_b[:2] += 2
graphs[1].replay()  # graphs[1] is now the graph for batch size 2
torch.cuda.current_stream().synchronize()
print(static_output[:2])