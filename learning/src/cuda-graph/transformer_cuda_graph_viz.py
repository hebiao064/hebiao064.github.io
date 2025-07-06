import torch
import gc
import psutil
import os
from contextlib import contextmanager

@contextmanager
def graph_capture(pool=None, stream=None, capture_error_mode: str = "global", dump_path=None):
    g = torch.cuda.CUDAGraph()
    if dump_path is not None:
        g.enable_debug_mode()
    
    print(f"Before graph capture - GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    with torch.cuda.graph(cuda_graph=g, pool=pool, stream=stream, capture_error_mode=capture_error_mode):
        yield g
    
    print(f"After graph capture - GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    if dump_path is not None:
        expanded_path = os.path.expanduser(dump_path)
        if os.path.isdir(expanded_path):
            dump_file = os.path.join(expanded_path, "cuda_graph_debug.dot")
        else:
            dump_file = expanded_path
        
        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
        print(f"Dumping CUDA graph debug info to: {dump_file}")
        g.debug_dump(dump_file)
        
        # Check file size
        if os.path.exists(dump_file):
            file_size = os.path.getsize(dump_file) / 1024**2
            print(f"Debug dump file size: {file_size:.2f} MB")

def analyze_model_complexity(batch_size, seq_len, hidden_size, num_layers):
    """Simulate a transformer-like model to understand CUDA graph size"""
    print(f"\n=== Analyzing model: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}, layers={num_layers} ===")
    
    # Create a simple transformer-like model
    class SimpleTransformerLayer(torch.nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
            self.norm1 = torch.nn.LayerNorm(hidden_size)
            self.ffn = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, hidden_size * 4),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_size * 4, hidden_size)
            )
            self.norm2 = torch.nn.LayerNorm(hidden_size)
        
        def forward(self, x):
            # Self-attention
            attn_out, _ = self.attention(x, x, x)
            x = self.norm1(x + attn_out)
            
            # FFN
            ffn_out = self.ffn(x)
            x = self.norm2(x + ffn_out)
            return x
    
    class SimpleTransformer(torch.nn.Module):
        def __init__(self, hidden_size, num_layers):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                SimpleTransformerLayer(hidden_size) for _ in range(num_layers)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    model = SimpleTransformer(hidden_size, num_layers).cuda()
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, device='cuda')
    
    # Memory before any operations
    torch.cuda.empty_cache()
    gc.collect()
    initial_memory = torch.cuda.memory_allocated()
    print(f"Initial GPU memory: {initial_memory / 1024**2:.2f} MB")
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()
    
    warmup_memory = torch.cuda.memory_allocated()
    print(f"After warmup GPU memory: {warmup_memory / 1024**2:.2f} MB")
    
    # Capture graph
    print("Capturing CUDA graph...")
    graph_list = []
    
    def compute():
        return model(input_tensor)
    
    # Capture with debug dump
    dump_path = f"/tmp/transformer_graph_b{batch_size}_s{seq_len}_h{hidden_size}_l{num_layers}.dot"
    with graph_capture(dump_path=dump_path) as g:
        output = compute()
    
    graph_list.append(g)
    
    final_memory = torch.cuda.memory_allocated()
    print(f"After graph capture GPU memory: {final_memory / 1024**2:.2f} MB")
    print(f"Graph memory overhead: {(final_memory - warmup_memory) / 1024**2:.2f} MB")
    
    # Analyze memory pool
    if hasattr(g, 'pool') and g.pool() is not None:
        print("Graph has a memory pool")
        
    # Test replay
    print("Testing graph replay...")
    input_tensor.fill_(1.0)  # Change input data
    g.replay()
    torch.cuda.synchronize()
    
    replay_memory = torch.cuda.memory_allocated()
    print(f"After replay GPU memory: {replay_memory / 1024**2:.2f} MB")
    
    return g, model

def main():
    print("CUDA Graph Memory Analysis")
    print("=" * 50)
    
    # Test different model sizes
    test_configs = [
        (1, 128, 512, 2),    # Small model
        (4, 512, 1024, 4),   # Medium model  
        (8, 1024, 2048, 6),  # Large model (similar to sglang scale)
    ]
    
    graphs = []
    for batch_size, seq_len, hidden_size, num_layers in test_configs:
        try:
            graph, model = analyze_model_complexity(batch_size, seq_len, hidden_size, num_layers)
            graphs.append((graph, f"b{batch_size}_s{seq_len}_h{hidden_size}_l{num_layers}"))
            
            # Clean up model but keep graph
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error with config {batch_size}, {seq_len}, {hidden_size}, {num_layers}: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Total graphs captured: {len(graphs)}")
    print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Each graph retains its memory pool
    print("\nMemory pools retained by graphs:")
    for graph, name in graphs:
        if hasattr(graph, 'pool'):
            print(f"Graph {name}: has memory pool")

if __name__ == "__main__":
    main() 