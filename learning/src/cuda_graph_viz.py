# Adapted from https://zhuanlan.zhihu.com/p/700224642

import torch
from contextlib import contextmanager
import os

@contextmanager
def graph_capture(pool=None, stream=None, capture_error_mode: str = "global", dump_path="~/"):
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

# Simple test
MAX_BATCHSIZE = 128

# Placeholder input used for capture
static_a = torch.zeros((MAX_BATCHSIZE, 10), device="cpu").pin_memory()
static_b = torch.zeros((MAX_BATCHSIZE, 10), device="cpu").pin_memory()

def compute(batchsize):
    a = static_a[:batchsize].to("cuda", non_blocking=True)
    b = static_b[:batchsize].to("cuda", non_blocking=True)
    output = (a ** 2 + b * 2)
    return output

# Warmup before capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(1, MAX_BATCHSIZE + 1):
        compute(i)
torch.cuda.current_stream().wait_stream(s)

# Test capturing a single graph with debug dump
print("Testing CUDA graph debug dump...")
with graph_capture(dump_path="./test_cuda_graph.dot") as g:
    result = compute(2)

print("Debug dump should have been created at: ./test_cuda_graph.dot")
if os.path.exists("./test_cuda_graph.dot"):
    print("✓ Debug dump file was created successfully!")
    with open("./test_cuda_graph.dot", "r") as f:
        content = f.read()
        if content.strip():
            print(f"✓ File contains {len(content)} characters of debug data")
        else:
            print("⚠ File exists but is empty")
else:
    print("✗ Debug dump file was not created") 