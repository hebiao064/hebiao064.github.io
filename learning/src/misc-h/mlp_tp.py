import torch
import torch.nn as nn
import torch.distributed as dist
import os

def init_dist():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size

class ColumnParallelLinear(nn.Module):
    def __init__(self, input_dim, output_dim, world_size):
        super().__init__()
        self.W = nn.Parameter(torch.randn(output_dim // world_size, input_dim))
        self.b = nn.Parameter(torch.zeros(output_dim // world_size))
    
    def forward(self, x):
        return torch.matmul(x, self.W.t()) + self.b

class RowParallelLinear(nn.Module):
    def __init__(self, input_dim, output_dim, world_size):
        super().__init__()
        self.W = nn.Parameter(torch.randn(output_dim, input_dim // world_size))
        self.b = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, x):
        output_partial = torch.matmul(x, self.W.t())
        dist.all_reduce(output_partial, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        
        return output_partial  + self.b
    
class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, world_size):
        super().__init__()
        self.fc1 = ColumnParallelLinear(input_dim, hidden_dim, world_size)
        self.fc2 = RowParallelLinear(hidden_dim, output_dim, world_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # x: [B, I]
        x = self.fc1(x)        # -> [B, H/N] (N=world_size)
        x = self.activation(x) # -> [B, H/N]
        x = self.fc2(x)        # -> [B, O]
        return x

def main():
    # 1. 初始化分布式环境
    local_rank, rank, world_size = init_dist()
    print(f"Rank {rank} of {world_size} started on device cuda:{local_rank}")

    # 2. 设置模型参数
    batch_size = 16
    input_dim = 32
    hidden_dim = 64
    output_dim = 8
    
    # 确保维度可以被 world_size 整除
    assert hidden_dim % world_size == 0, "Hidden dimension must be divisible by world_size"

    # 3. 创建模型并移动到 GPU
    model = SimpleModel(input_dim, hidden_dim, output_dim, world_size).to(local_rank)
    
    # 4. 创建输入数据（所有 GPU 上都一样）
    # 在实际应用中，数据通常也会被切分 (数据并行)，但为了单纯演示张量并行，我们用相同的输入
    input_tensor = torch.randn(batch_size, input_dim).to(local_rank)

    # 5. 前向传播
    output = model(input_tensor)

    # 6. 打印结果 (只在 rank 0 上打印，避免重复输出)
    if rank == 0:
        print("Input tensor shape:", input_tensor.shape)
        print("Output tensor shape:", output.shape)
        print("Successfully ran forward pass!")
        # 验证输出维度是否正确
        assert output.shape == (batch_size, output_dim)
        
    # 清理
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    
# torchrun /workspace/mlp_tp.py --nproc_per_node=2 --nnodes=1