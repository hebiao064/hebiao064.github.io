import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.distributed as dist

def init_dist():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 检查是否有GPU
    if torch.cuda.is_available():
        # 有GPU：使用nccl后端
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{local_rank}")
        print(f"Process {rank}: Using GPU {local_rank}")
    else:
        # 没有GPU：使用gloo后端
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        device = torch.device("cpu")
        print(f"Process {rank}: Using CPU")
    
    return rank, local_rank, world_size, device

class ColumnParallelLinear(nn.Module):
    def __init__(self, input_dim, output_dim, world_size):
        super().__init__()
        # 权重矩阵：每个进程处理 output_dim // world_size 个输出
        self.W = nn.Parameter(torch.randn(output_dim // world_size, input_dim))
        self.b = nn.Parameter(torch.zeros(output_dim // world_size))

    def forward(self, input):
        # input: (..., input_dim)
        # W: (output_dim // world_size, input_dim)
        # 输出: (..., output_dim // world_size)
        return torch.matmul(input, self.W.T) + self.b

class RowParallelLinear(nn.Module):
    def __init__(self, input_dim, output_dim, world_size):
        super().__init__()
        # 权重矩阵：每个进程处理 input_dim // world_size 个输入
        self.W = nn.Parameter(torch.randn(output_dim, input_dim // world_size))
        self.b = nn.Parameter(torch.zeros(output_dim))

    def forward(self, input):
        # input: (..., input_dim // world_size)
        # W: (output_dim, input_dim // world_size)
        # 先计算部分输出
        output_partial = torch.matmul(input, self.W.T)
        
        # 然后进行 all_reduce 合并所有进程的结果
        dist.all_reduce(output_partial, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        return output_partial + self.b

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, world_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.world_size = world_size
        self.hidden_dim = hidden_dim
        
        # 简化处理：假设输入总是 hidden_dim // world_size
        self.q_proj = ColumnParallelLinear(hidden_dim // world_size, hidden_dim, world_size)
        self.k_proj = ColumnParallelLinear(hidden_dim // world_size, hidden_dim, world_size)
        self.v_proj = ColumnParallelLinear(hidden_dim // world_size, hidden_dim, world_size)
        self.o_proj = RowParallelLinear(hidden_dim, hidden_dim, world_size)

    def forward(self, input: torch.Tensor):
        bsz, seq_len = input.shape[:2]
        
        # Q, K, V 投影
        q_value = self.q_proj(input).view(bsz, seq_len, self.num_heads, (self.head_dim // self.world_size)).transpose(1, 2)
        k_value = self.k_proj(input).view(bsz, seq_len, self.num_heads, (self.head_dim // self.world_size)).transpose(1, 2)
        v_value = self.v_proj(input).view(bsz, seq_len, self.num_heads, (self.head_dim // self.world_size)).transpose(1, 2)

        # 注意力计算
        qk = torch.matmul(q_value, k_value.transpose(-1, -2)) / ((self.head_dim // self.world_size) ** 0.5)
        mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, dtype=bool, device=qk.device))
        masked_qk = torch.where(mask, qk, torch.ones_like(qk) * -torch.inf)
        softmax_qk = torch.softmax(masked_qk, dim=-1)

        attn = torch.matmul(softmax_qk, v_value)

        y = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_dim // self.world_size)
        
        # 输出投影
        o = self.o_proj(y)
        return o

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, world_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.world_size = world_size
        
        # 处理的是 hidden_dim // world_size
        self.pre_norm = nn.LayerNorm(hidden_dim // world_size)
        self.attn = SelfAttention(hidden_dim, num_heads, world_size)
        self.post_norm = nn.LayerNorm(hidden_dim // world_size)
        
        # FFN 使用张量并行
        self.ffn = nn.Sequential(
            ColumnParallelLinear(hidden_dim // world_size, 4 * hidden_dim, world_size),
            nn.ReLU(),
            RowParallelLinear(4 * hidden_dim, hidden_dim, world_size)
        )

    def forward(self, input):
        # input: (bsz, seq_len, hidden_dim // world_size)
        # attn 输出: (bsz, seq_len, hidden_dim // world_size) 经过 all_reduce
        # 但我们需要确保维度匹配
        
        # 注意力分支
        attn_input = self.pre_norm(input)
        attn_out = self.attn(attn_input)
        
        # 注意：attn_out 经过 RowParallelLinear 后维度应该是 hidden_dim // world_size
        # 但实际上 RowParallelLinear 会输出完整的 hidden_dim，然后我们需要分割
        rank = dist.get_rank()
        chunk_size = self.hidden_dim // self.world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size
        attn_out_chunk = attn_out[..., start_idx:end_idx]
        
        x = input + attn_out_chunk
        
        # FFN 分支
        ffn_input = self.post_norm(x)
        ffn_out = self.ffn(ffn_input)
        
        # 同样处理 FFN 输出
        ffn_out_chunk = ffn_out[..., start_idx:end_idx]
        x = x + ffn_out_chunk
        
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size=1024, hidden_dim=256, num_layers=12, num_heads=8, max_context_len=256, world_size=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.world_size = world_size
        
        # 嵌入层 - 列并行
        # 输入: vocab_size -> 输出: hidden_dim // world_size
        self.emb_token = ColumnParallelLinear(vocab_size, hidden_dim, world_size)
        self.pos_emb = nn.Parameter(torch.randn(1, max_context_len, hidden_dim // world_size))

        self.layers = nn.ModuleList(TransformerBlock(hidden_dim, num_heads, world_size) for _ in range(num_layers))

        self.post_norm = nn.LayerNorm(hidden_dim // world_size)
        # 输出层 - 行并行
        # 输入: hidden_dim // world_size -> 输出: vocab_size
        self.lm_head = RowParallelLinear(hidden_dim, vocab_size, world_size)

    def forward(self, input):
        bsz, seq_len = input.shape
        
        # 输入嵌入
        input_onehot = F.one_hot(input, num_classes=self.vocab_size).float()
        x = self.emb_token(input_onehot) + self.pos_emb[:, :seq_len, :]
        
        # 通过所有层
        for layer in self.layers:
            x = layer(x)
        x = self.post_norm(x)
        x = self.lm_head(x)
        return x

    def loss(self, tokens):
        input = tokens[:, :-1]
        target = tokens[:, 1:]
        output = self.forward(input)
        return F.cross_entropy(output.view(-1, self.vocab_size), target.contiguous().view(-1))

def main():
    # 初始化分布式训练
    rank, local_rank, world_size, device = init_dist()
    
    # 创建模型并移动到设备
    model = GPT(world_size=world_size).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    
    # 创建训练数据（每个进程使用不同的数据）
    torch.manual_seed(42 + rank)  # 不同进程使用不同种子
    input_data = torch.randint(0, 1024, (4, 20)).to(device)
    
    # 训练循环
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        loss = model.loss(input_data)
        loss.backward()
        
        # 张量并行不需要手动梯度同步，因为已经在forward中处理了
        optimizer.step()
        
        if rank == 0:  # 只在主进程打印
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # 训练完成，打印结果
    if rank == 0:
        print("\n" + "="*50)
        print("Training completed successfully!")
        print(f"Final loss: {loss.item():.4f}")
        print("="*50)
    
    # 尝试正常清理，但设置超时
    try:
        # 同步所有进程
        dist.barrier()
        if rank == 0:
            print("All processes synchronized. Cleaning up...")
        
        # 清理分布式进程组
        dist.destroy_process_group()
        print(f"Process {rank} finished successfully.")
        
    except Exception as e:
        print(f"Process {rank} cleanup failed: {e}")
        print(f"Process {rank} exiting anyway...")
    
    # 强制退出
    import sys
    sys.exit(0)

if __name__ == "__main__":
    main()