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
        expanded_path = os.path.expanduser(dump_path)
        if os.path.isdir(expanded_path):
            dump_file = os.path.join(expanded_path, "cuda_graph_debug.dot")
        else:
            dump_file = expanded_path
        
        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
        print(f"Dumping CUDA graph debug info to: {dump_file}")
        g.debug_dump(dump_file)

# 模拟SGLang的前向推理
class SGLangForwardSimulator:
    def __init__(self, max_batch_size=128, seq_len=1024, hidden_size=4096):
        self.max_batch_size = max_batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        
        # 预分配静态缓冲区 - 这是关键！
        self.static_input = torch.zeros(max_batch_size, seq_len, hidden_size, device="cuda")
        self.static_output = torch.zeros(max_batch_size, seq_len, hidden_size, device="cuda")
        self.static_logits = torch.zeros(max_batch_size, seq_len, 50000, device="cuda")  # 词汇表大小
        
        # 模拟模型权重
        self.attention_weights = torch.randn(hidden_size, hidden_size, device="cuda")
        self.ffn_weights = torch.randn(hidden_size, hidden_size * 4, device="cuda")
        self.output_weights = torch.randn(hidden_size, 50000, device="cuda")
    
    def forward(self, batch_size):
        """模拟前向推理，使用静态缓冲区"""
        # 使用静态缓冲区的子集
        x = self.static_input[:batch_size]
        
        # 模拟attention计算
        x = torch.matmul(x, self.attention_weights)
        x = torch.relu(x)
        
        # 模拟FFN
        x = torch.matmul(x, self.ffn_weights)
        x = torch.relu(x)
        
        # 模拟输出层
        logits = torch.matmul(x, self.output_weights)
        
        # 写入静态输出缓冲区
        self.static_output[:batch_size] = x
        self.static_logits[:batch_size] = logits
        
        return self.static_logits[:batch_size]

def memory_efficient_batch_capture():
    """内存高效的批量捕获策略"""
    print("SGLang CUDA Graph 内存优化示例")
    print("=" * 50)
    
    # 初始化模拟器
    max_batch_size = 128
    simulator = SGLangForwardSimulator(max_batch_size)
    
    # 定义要捕获的batch sizes（从大到小）
    batch_sizes = [128, 64, 32, 16, 8, 4, 2, 1]
    
    print(f"要捕获的batch sizes: {batch_sizes}")
    print(f"策略: 从大到小捕获，共享内存池")
    
    # 预热
    print("\n预热阶段...")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for batch_size in batch_sizes:
            for _ in range(3):
                simulator.forward(batch_size)
    torch.cuda.current_stream().wait_stream(s)
    
    initial_memory = torch.cuda.memory_allocated()
    print(f"预热后内存: {initial_memory / 1024**2:.2f} MB")
    
    # 捕获所有batch size的图
    graphs = {}
    memory_pool = None
    
    print("\n开始捕获CUDA graphs...")
    for i, batch_size in enumerate(batch_sizes):
        print(f"\n捕获 batch_size={batch_size}...")
        
        before_capture = torch.cuda.memory_allocated()
        
        # 关键：使用共享内存池
        dump_path = f"/mnt/shared-fs/sglang-qiaolin/var/sglang_graph_batch_{batch_size}.dot"
        with graph_capture(pool=memory_pool, dump_path=dump_path) as g:
            result = simulator.forward(batch_size)
        
        after_capture = torch.cuda.memory_allocated()
        memory_increase = (after_capture - before_capture) / 1024**2
        
        graphs[batch_size] = g
        
        # 更新内存池
        if memory_pool is None:
            memory_pool = g.pool()
            print(f"  ✓ 创建内存池，内存增加: {memory_increase:.2f} MB")
        else:
            memory_pool = g.pool()  # 更新为新的内存池
            print(f"  ✓ 复用内存池，内存增加: {memory_increase:.2f} MB")
        
        print(f"  当前总内存: {after_capture / 1024**2:.2f} MB")
        print(f"  累计内存增长: {(after_capture - initial_memory) / 1024**2:.2f} MB")
    
    total_memory = torch.cuda.memory_allocated()
    print(f"\n总内存使用: {total_memory / 1024**2:.2f} MB")
    print(f"平均每个图: {(total_memory - initial_memory) / len(batch_sizes) / 1024**2:.2f} MB")
    
    return graphs, simulator

def smart_batch_selection(graphs, simulator):
    """智能批量选择和推理"""
    print("\n" + "=" * 50)
    print("智能batch size选择测试")
    print("=" * 50)
    
    available_batch_sizes = sorted(graphs.keys(), reverse=True)
    print(f"可用的batch sizes: {available_batch_sizes}")
    
    def select_best_batch_size(actual_batch_size):
        """选择最小的满足条件的batch size"""
        suitable = [bs for bs in available_batch_sizes if bs >= actual_batch_size]
        return min(suitable) if suitable else max(available_batch_sizes)
    
    # 测试用例
    test_cases = [1, 3, 5, 9, 17, 33, 65, 100, 128]
    
    print("\n推理测试:")
    for actual_batch in test_cases:
        selected_batch = select_best_batch_size(actual_batch)
        
        # 模拟输入数据
        input_data = torch.randn(actual_batch, 1024, 4096, device="cuda")
        
        # 复制到静态缓冲区
        simulator.static_input[:actual_batch] = input_data
        
        # 执行推理
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        graphs[selected_batch].replay()
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"  实际batch={actual_batch:3d} -> 使用图batch={selected_batch:3d}, 时间={elapsed_time:.2f}ms")

def memory_comparison():
    """内存策略比较"""
    print("\n" + "=" * 50)
    print("内存策略比较")
    print("=" * 50)
    
    batch_sizes = [32, 16, 8, 4, 2, 1]
    simulator = SGLangForwardSimulator(32)
    
    # 策略1: 独立内存池
    print("\n策略1: 每个图独立内存池")
    torch.cuda.empty_cache()
    
    independent_graphs = {}
    base_memory = torch.cuda.memory_allocated()
    
    for batch_size in batch_sizes:
        with graph_capture() as g:  # 没有共享pool
            simulator.forward(batch_size)
        independent_graphs[batch_size] = g
        
        current_memory = torch.cuda.memory_allocated()
        print(f"  batch={batch_size}: {(current_memory - base_memory) / 1024**2:.2f} MB")
        base_memory = current_memory
    
    independent_total = torch.cuda.memory_allocated()
    
    # 清理
    del independent_graphs, simulator
    torch.cuda.empty_cache()
    
    # 策略2: 共享内存池
    print("\n策略2: 共享内存池")
    simulator2 = SGLangForwardSimulator(32)
    
    shared_graphs = {}
    memory_pool = None
    base_memory = torch.cuda.memory_allocated()
    
    for batch_size in batch_sizes:
        with graph_capture(pool=memory_pool) as g:  # 共享pool
            simulator2.forward(batch_size)
        shared_graphs[batch_size] = g
        memory_pool = g.pool()
        
        current_memory = torch.cuda.memory_allocated()
        print(f"  batch={batch_size}: {(current_memory - base_memory) / 1024**2:.2f} MB")
        base_memory = current_memory
    
    shared_total = torch.cuda.memory_allocated()
    
    print(f"\n结果对比:")
    print(f"  独立内存池总计: {independent_total / 1024**2:.2f} MB")
    print(f"  共享内存池总计: {shared_total / 1024**2:.2f} MB")
    print(f"  内存节省: {(independent_total - shared_total) / 1024**2:.2f} MB")
    print(f"  节省比例: {((independent_total - shared_total) / independent_total) * 100:.1f}%")

if __name__ == "__main__":
    # 主要演示
    graphs, simulator = memory_efficient_batch_capture()
    
    # 智能选择测试
    smart_batch_selection(graphs, simulator)
    
    # 清理
    del graphs, simulator
    torch.cuda.empty_cache()
    
    # 内存策略比较
    memory_comparison()
    
    print(f"\n最终GPU内存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB") 