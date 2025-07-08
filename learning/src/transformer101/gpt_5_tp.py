import torch
from torch import nn
from math import sqrt
from transformers import AutoTokenizer
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp


class ColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_rank, tp_size, process_group=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.output_size_per_rank = output_size // tp_size
        
        # 权重形状：(output_size_per_rank, input_size)
        self.weight = nn.Parameter(torch.randn(self.output_size_per_rank, input_size))
        self.bias = nn.Parameter(torch.randn(self.output_size_per_rank))
        self.process_group = process_group or torch.distributed.group.WORLD
        
        if tp_rank == 0:  # 只在rank 0上打印
            print(f"ColumnParallelLinear: {input_size} -> {output_size} (per_rank: {self.output_size_per_rank})")
    
    def forward(self, inputs):
        # 计算当前rank的部分输出
        output_partial = torch.matmul(inputs, self.weight.t()) + self.bias
        
        # All-gather收集所有rank的输出
        output_list = [torch.empty_like(output_partial) for _ in range(self.tp_size)]
        torch.distributed.all_gather(output_list, output_partial, group=self.process_group)
        
        # 在最后一个维度上连接
        return torch.cat(output_list, dim=-1)
    
class RowParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, tp_rank, tp_size, process_group=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.input_size_per_rank = input_size // tp_size
        
        # 权重形状：(output_size, input_size_per_rank)
        self.weight = nn.Parameter(torch.randn(output_size, self.input_size_per_rank))
        self.bias = nn.Parameter(torch.randn(output_size))
        self.process_group = process_group or torch.distributed.group.WORLD
        
        if tp_rank == 0:  # 只在rank 0上打印
            print(f"RowParallelLinear: {input_size} -> {output_size} (per_rank: {self.input_size_per_rank})")
    
    def forward(self, inputs: torch.Tensor, is_input_paralled: bool = False):
        if not is_input_paralled:
            # 切分输入张量
            start_idx = self.input_size_per_rank * self.tp_rank
            end_idx = self.input_size_per_rank * (self.tp_rank + 1)
            inputs_partial = inputs[:, :, start_idx:end_idx]  # 3D张量切片
        else:
            inputs_partial = inputs
        
        # 矩阵乘法
        outputs = torch.matmul(inputs_partial, self.weight.t())
        
        # All-reduce求和
        torch.distributed.all_reduce(outputs, torch.distributed.ReduceOp.SUM, group=self.process_group)
        
        return outputs + self.bias
                                      
class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, tp_rank=0, tp_size=1):
        super().__init__() 

        self.norm_1 = nn.LayerNorm(hidden_dim) # layer norm is learnable
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim=hidden_dim, tp_rank=tp_rank, tp_size=tp_size)

        # FFN is a special MLP used by Transformer Models like GPT, LLama
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), # up proj
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim) # down proj
        )
    
    def forward(self, x: torch.Tensor, attention_mask=None, past_kv_cache_tuple=None):
        attn_out, new_kv_cache_tuple = self.attn(self.norm_1(x), attention_mask=attention_mask, past_kv_cache_tuple=past_kv_cache_tuple)
        x = x + attn_out
        x = x + self.mlp(self.norm_2(x))

        return x, new_kv_cache_tuple
        
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, tp_rank=0, tp_size=1):
        super().__init__()

        # self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        if tp_size > 1:
            # 使用tensor parallel
            self.q_proj = ColumnParallelLinear(hidden_dim, hidden_dim, tp_rank=tp_rank, tp_size=tp_size, process_group=dist.group.WORLD)
            self.k_proj = ColumnParallelLinear(hidden_dim, hidden_dim, tp_rank=tp_rank, tp_size=tp_size, process_group=dist.group.WORLD)
            self.v_proj = ColumnParallelLinear(hidden_dim, hidden_dim, tp_rank=tp_rank, tp_size=tp_size, process_group=dist.group.WORLD)
            self.o_proj = RowParallelLinear(hidden_dim, hidden_dim, tp_rank=tp_rank, tp_size=tp_size, process_group=dist.group.WORLD)
        else:
            # 单GPU情况，使用普通Linear
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = nn.Linear(hidden_dim, hidden_dim)
            self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
    
    def forward(self, x: torch.Tensor, attention_mask=None, past_kv_cache_tuple=None):
        batch_size, seq_len, hidden_dim = x.shape

        if past_kv_cache_tuple is not None:
            # forward decode
            assert seq_len == 1, f"seq_len should be 1 for forward decode, but got {seq_len}"
            new_k_value = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            new_v_value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k_from_cache, v_from_cache = past_kv_cache_tuple
            k_value = torch.cat([k_from_cache, new_k_value], dim=2) # cat k within the seq dim
            v_value = torch.cat([v_from_cache, new_v_value], dim=2) # cat v within the seq dim
        else:
            # forward extend
            k_value = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v_value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # We need to transpose from the activations
        # from (batch, seq, hid) 
        # to (batch, num_heads, seq, head_dim)
        # to faciliate attention calculation
        q_value = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # q with shape (batch, num_heads, seq, head_dim) 
        # cannot muliply with k (batch, num_heads, seq, head_dim)
        # so we need to transpose k to be (batch, num_heads, head_dim, seq)
        attn = torch.matmul(q_value, k_value.transpose(-1, -2)) / sqrt(self.head_dim)

        # causal mask: (1, 1, seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)).view(1, 1, seq_len, seq_len)

        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Expand to (batch, 1, seq_len, seq_len) for broadcasting
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
            # Convert to boolean before combining with causal_mask
            attention_mask = attention_mask.bool()
            # Apply mask: set padded positions to -inf
            combined_mask = causal_mask & attention_mask
        else:
            combined_mask = causal_mask

        attn = torch.where(combined_mask, attn, torch.zeros_like(attn) - torch.inf)
        attn = torch.softmax(attn, -1)

        y = torch.matmul(attn, v_value)

        # Transpose back since attn calculation is done, we don't need to split it by heads
        y = y.transpose(1, 2)
        y = y.reshape(batch_size, seq_len, hidden_dim)
        
        # Apply output projection
        # 对于RowParallelLinear，需要传递is_input_paralled=True，因为QKV的输出已经是并行的
        if hasattr(self.o_proj, 'tp_size') and self.o_proj.tp_size > 1:
            o = self.o_proj(y, is_input_paralled=False)  # y is full tensor, not parallel
        else:
            o = self.o_proj(y)
            
        return o, (k_value, v_value)

class GPT(nn.Module):
    def __init__(self, context_len=1024, vocab_size=50257, hidden_dim=768, tp_rank=0, tp_size=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.rand(1, context_len, hidden_dim))
        # 或者 self.pos_emb = nn.Embedding(context_len, hidden_dim)
        self.module_list = nn.ModuleList([TransformerBlock(hidden_dim=hidden_dim, tp_rank=tp_rank, tp_size=tp_size) for _ in range(12)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, attention_mask=None, past_kv_cache=None, enable_kv_cache=False):
        # x: (batch, seq_len)
        # kv_cache: (layer, batch, seq_len, num_heads, head_dim)

        if past_kv_cache is None and enable_kv_cache:
            past_kv_cache = [None] * len(self.module_list)

        offset = past_kv_cache[0][0].size(2) if (past_kv_cache is not None and past_kv_cache[0] is not None) else 0
        # print(f"offset: {offset}")
        seq_len = x.size(1) if x.ndim == 2 else 1
        x = self.emb(x) + self.pos_emb[:, offset:seq_len + offset, :]

        new_kv_cache = []
        for i, block in enumerate(self.module_list):
            if enable_kv_cache:
                x, new_kv_cache_tuple = block(x, attention_mask=attention_mask, past_kv_cache_tuple=past_kv_cache[i])
                new_kv_cache.append(new_kv_cache_tuple)
            else:
                x, _ = block(x, attention_mask=attention_mask)
                new_kv_cache.append(None)
        
        x = self.norm(x)
        output_vocab = self.lm_head(x)
        
        return output_vocab, new_kv_cache
    
    '''
    ==============================================================================================
                                        Add Loss Function for Training
    ==============================================================================================
    '''

    def loss(self, tokens, attention_mask=None):
        inputs = tokens[:, :-1] # (batch, seq_len - 1) 
        labels = tokens[:, 1:] # (batch, seq_len - 1)
        logits, _ = self.forward(inputs, enable_kv_cache=False) # (batch, seq_len - 1, vocab_size)

        if attention_mask is not None:
            valid_mask_indices = attention_mask[:, 1:].reshape(-1)
            valid_logits = logits.view(-1, logits.size(-1))[valid_mask_indices]
            valid_labels = labels.reshape(-1)[valid_mask_indices]
            cross_entropy = F.cross_entropy(valid_logits, valid_labels)
        else:
            # logits: (batch, seq_len - 1, vocab_size) -> (batch * (seq_len - 1), vocab_size)
            # labels: (batch, seq_len - 1) -> (batch * (seq_len - 1))
            # we need to calculate the cross entropy for each token in the sequence
            cross_entropy = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.reshape(-1))

        return cross_entropy
    

'''
==============================================================================================
                                    Run Single Token Forward
==============================================================================================
'''

def run_single_token_forward(model):
    # 创建输入数据（每个rank都需要相同的输入）
    torch.manual_seed(42)  # 确保所有rank生成相同的随机数据
    input_ids = torch.randint(0, 10000, (2, 128))  # batch=2, seq_len=128
    
    # 前向传播（每个rank都参与计算）
    output, _ = model(input_ids)
    
    # 只在rank 0上打印结果
    if dist.get_rank() == 0:
        print(f"Output shape: {output.shape}")  # (2, 128, vocab_size)

def run_single_token_forward_with_tokenizer(model, tokenizer):
    # 这个函数只在rank 0上运行，因为它主要用于展示
    input_texts = ["Once upon a time, there was a magical forest",
                    "Once upon a time, there was a magical forest"]
    
    input_batch = tokenizer(input_texts, 
                      return_tensors="pt", 
                      padding="longest")

    attention_mask = input_batch["attention_mask"]
    real_lens = attention_mask.sum(dim=1)

    # 使用已有的model，不要重新创建
    outputs, _ = model(input_batch.input_ids, attention_mask=attention_mask)
    print(f"Output shape: {outputs.shape}")
    
    # 获取预测结果
    for i in range(len(real_lens)):
        predicted_id = torch.argmax(outputs[i, real_lens[i] - 1])
        print(f"Original text + Predicted token: {input_texts[i] + tokenizer.decode([predicted_id.item()])}")


'''
==============================================================================================
                                    GPT 2 Auto-regressive Inference
==============================================================================================
'''
def generate(model: nn.Module, tokenizer: AutoTokenizer, texts: list[str], max_new_tokens: int):
    # kv_cache: (batch, num_heads, seq_len, head_dim)
    past_kv_cache = None
    new_token_ids = tokenizer(texts, return_tensors="pt", padding="longest").input_ids
    for _ in range(max_new_tokens):
        logits, past_kv_cache = model(new_token_ids, past_kv_cache=past_kv_cache, enable_kv_cache=True)
        new_token_ids = torch.distributions.Categorical(logits=logits[:, -1, :]).sample()
        new_token_ids = new_token_ids.unsqueeze(1)
        texts = [text + tokenizer.decode([next_token_id.item()]) for text, next_token_id in zip(texts, new_token_ids)]

    for i in range(len(texts)):
        print(f"Generated text: {texts[i]}")

'''
==============================================================================================
                                    GPT 2 Training
==============================================================================================
'''
def train(model: nn.Module, tokenizer: AutoTokenizer):
    training_data = [
        "Once upon a time, there was a magical forest",
        "She was a beautiful princess",
        "America is a country in North America",
        "China is a country in Asia",
        "Japan is a country in Asia",
        "France is a country in Europe",
        "Germany is a country in Europe",
        "Italy is a country in Europe",
        "Spain is a country in Europe",
    ]

    training_tokens = tokenizer(training_data, return_tensors="pt", padding="max_length", max_length=128)
    input_ids = training_tokens.input_ids
    attention_mask = training_tokens.attention_mask

    epochs = 100
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        optimizer.zero_grad()

        loss = model.loss(input_ids, attention_mask)
        loss.backward() # calculate the gradient

        optimizer.step() # update the weight

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


    return model

def main(rank, world_size):
    # 1. 初始化分布式进程组
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",  # 本地单机
        rank=rank,
        world_size=world_size,
    )

    # 2. 测试分布式通信
    tensor = torch.ones(2)
    if rank == 0:
        print(f"[Rank {rank}] before all_reduce: {tensor}")
    
    # 3. all-reduce（求和）
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"[Rank {rank}] after all_reduce: {tensor}")

    # 4. 只在rank 0上加载tokenizer
    if rank == 0:
        print("Loading tokenizer and creating model...")
        
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 5. 创建模型（传递正确的tp参数）
    model = GPT(context_len=128, vocab_size=tokenizer.vocab_size, hidden_dim=512, tp_rank=rank, tp_size=world_size)
    
    # 6. 同步所有进程
    dist.barrier()
    
    if rank == 0:
        print("="*100)
        print("Run Simple Forward Before Training")
        
    # 7. 运行测试（每个rank都需要参与计算）
    run_single_token_forward(model)
    
    dist.barrier()
    # 只在rank 0上打印详细信息
    if rank == 0:
        print("="*100)
        print("Run Simple Forward Before Training")

    # 8. 运行测试（每个rank都需要参与计算）
    run_single_token_forward_with_tokenizer(model, tokenizer)
    dist.barrier()

    # 9. Run Auto-regressive Inference before training
    print("="*100)
    print("Run Auto-regressive Inference before training")
    generate(model, tokenizer, ["China is a country in ", "China is a country in "], max_new_tokens=10)

    # 10. 清理
    dist.destroy_process_group()


if __name__ == "__main__":
    world_size = 2
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)


"""
This script added support of:
1. Support KV Cache for Inference
"""