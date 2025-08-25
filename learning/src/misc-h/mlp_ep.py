import torch
import torch.nn as nn
import torch.distributed as dist
import os
import math

def init_dist():
    """初始化分布式环境"""
    dist.init_process_group(backend='nccl')
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE"))
    local_rank = int(os.environ.get("LOCAL_RANK"))
    
    torch.cuda.set_device(local_rank)
    
    return local_rank, rank, world_size

class MoEExpert(nn.Module):
    """一个简单的专家模型"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.ffn(x)

class MoE_AllToAll_Layer(nn.Module):
    """修正版的专家并行MoE层"""
    def __init__(self, input_dim, hidden_dim, output_dim, world_size):
        super().__init__()
        self.rank = dist.get_rank()
        self.world_size = world_size
        self.num_experts = world_size
        
        # 路由器
        self.router = nn.Linear(input_dim, self.num_experts)
        
        # 本地专家
        self.expert = MoEExpert(input_dim, hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size, input_dim = x.shape
        
        # 1. 路由决策
        router_logits = self.router(x)
        router_scores = torch.softmax(router_logits, dim=-1)
        
        # Top-1 专家选择
        top_k_scores, top_k_indices = torch.topk(router_scores, k=1, dim=-1)
        expert_indices = top_k_indices.squeeze(-1)  # [batch_size]
        
        # 2. 按专家分组并排序
        sorted_expert_indices, sort_indices = torch.sort(expert_indices)
        sorted_x = x[sort_indices]  # 按专家分配重排输入
        sorted_scores = top_k_scores[sort_indices]  # 对应的分数
        
        # 3. 统计每个专家需要处理的token数量
        tokens_per_expert = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        for i in range(self.num_experts):
            tokens_per_expert[i] = (sorted_expert_indices == i).sum()
        
        print(f"Rank {self.rank}: tokens_per_expert = {tokens_per_expert}")
        
        # 4. 通过all_gather收集所有rank的token分布信息
        all_tokens_per_expert = [torch.zeros_like(tokens_per_expert) for _ in range(self.world_size)]
        dist.all_gather(all_tokens_per_expert, tokens_per_expert)
        
        # 计算每个rank要接收多少tokens (从所有rank的第rank个专家)
        tokens_to_recv = sum(tpe[self.rank] for tpe in all_tokens_per_expert)
        
        # 5. 准备all_to_all的发送和接收张量
        # 发送：按专家分组的tokens
        send_splits = tokens_per_expert.tolist()
        send_tensors = list(torch.split(sorted_x, send_splits))
        
        # 接收：为每个rank预分配空间
        recv_tensors = []
        for rank in range(self.world_size):
            recv_size = all_tokens_per_expert[rank][self.rank].item()
            if recv_size > 0:
                recv_tensors.append(torch.empty(recv_size, input_dim, device=x.device))
            else:
                recv_tensors.append(torch.empty(0, input_dim, device=x.device))
        
        # 6. 第一次all_to_all：分发tokens到对应专家
        dist.all_to_all(recv_tensors, send_tensors)
        
        # 7. 本地专家计算
        local_input = torch.cat([t for t in recv_tensors if t.numel() > 0])
        if local_input.numel() > 0:
            expert_output = self.expert(local_input)
        else:
            expert_output = torch.empty(0, output_dim, device=x.device)
        
        # 8. 准备发送回去的结果
        output_dim = expert_output.shape[-1] if expert_output.numel() > 0 else input_dim
        
        send_result_tensors = []
        start_idx = 0
        for rank in range(self.world_size):
            size = all_tokens_per_expert[rank][self.rank].item()
            if size > 0:
                send_result_tensors.append(expert_output[start_idx:start_idx + size])
                start_idx += size
            else:
                send_result_tensors.append(torch.empty(0, output_dim, device=x.device))
        
        # 接收结果张量
        recv_result_tensors = []
        for i in range(self.world_size):
            recv_size = send_splits[i]
            if recv_size > 0:
                recv_result_tensors.append(torch.empty(recv_size, output_dim, device=x.device))
            else:
                recv_result_tensors.append(torch.empty(0, output_dim, device=x.device))
        
        # 9. 第二次all_to_all：收集结果
        dist.all_to_all(recv_result_tensors, send_result_tensors)
        
        # 10. 重新组装并恢复原始顺序
        gathered_output = torch.cat([t for t in recv_result_tensors if t.numel() > 0])
        
        # 创建逆排序索引来恢复原始顺序
        final_output = torch.empty(batch_size, output_dim, device=x.device)
        final_output[sort_indices] = gathered_output
        
        # 11. 应用路由权重
        final_weighted_output = final_output * sorted_scores.view(-1, 1)
        
        return final_weighted_output

def main():
    local_rank, rank, world_size = init_dist()
    print(f"Rank {rank} of {world_size} started on device cuda:{local_rank}")

    if world_size < 2:
        raise RuntimeError("需要至少2个GPU才能运行专家并行示例")

    batch_size = 8  # 减小batch size便于调试
    input_dim = 32
    hidden_dim = 64
    output_dim = 16

    model = MoE_AllToAll_Layer(input_dim, hidden_dim, output_dim, world_size).to(local_rank)
    
    # 所有GPU都创建相同的输入
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, input_dim).to(local_rank)

    print(f"Rank {rank}: input shape = {input_tensor.shape}")
    
    output = model(input_tensor)

    print(f"Rank {rank}: output shape = {output.shape}")
    
    if rank == 0:
        print("-" * 20)
        print(f"Final output shape: {output.shape}")
        print("Successfully ran forward pass!")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
    
# 运行命令: torchrun --nproc_per_node=2 --nnodes=1 learning/src/misc-h/fixed_mlp_ep.py
