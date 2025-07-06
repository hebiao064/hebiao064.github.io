import torch
from torch import nn
from math import sqrt

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__() 

        self.norm_1 = nn.LayerNorm(hidden_dim) # layer norm is learnable
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.attn = SelfAttention(hidden_dim=hidden_dim)

        # FFN is a special MLP used by Transformer Models like GPT, LLama
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), # up proj
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim) # down proj
        )
    
    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.norm_1(x))
        x = x + self.mlp(self.norm_2(x))

        return x
        
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.shape

        # We need to transpose from the activations
        # from (batch, seq, hid) 
        # to (batch, num_heads, seq, head_dim)
        # to faciliate attention calculation
        q_value = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_value = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_value = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # q with shape (batch, num_heads, seq, head_dim) 
        # cannot muliply with k (batch, num_heads, seq, head_dim)
        # so we need to transpose k to be (batch, num_heads, head_dim, seq)
        attn = torch.matmul(q_value, k_value.transpose(-1, -2)) / sqrt(self.head_dim)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)).view(1, 1, seq_len, seq_len)
        attn = torch.where(mask, attn, torch.zeros_like(attn) - torch.inf)
        attn = torch.softmax(attn, -1)

        y = torch.matmul(attn, v_value)

        # Transpose back since attn calculation is done, we don't need to split it by heads
        y = y.transpose(1, 2)
        y = y.reshape(batch_size, seq_len, hidden_dim)
        
        o = self.o_proj(y)
        return o

class GPT(nn.Module):
    def __init__(self, context_len=1024, vocab_size=50257, hidden_dim=768):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Parameter(torch.rand(1, context_len, hidden_dim))
        self.module_list = nn.ModuleList([TransformerBlock(hidden_dim=hidden_dim) for _ in range(12)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):

        x = self.emb(x) + self.pos_emb[:, :x.size(1), :]

        for block in self.module_list:
            x = block(x)
        
        x = self.norm(x)
        output_vocab = self.lm_head(x)
        
        return output_vocab

def run_single_token_forward():
    model = GPT(context_len=128, vocab_size=10000, hidden_dim=512)
    input_ids = torch.randint(0, 10000, (2, 128))  # batch=2, seq_len=128
    output = model(input_ids)
    print("Output shape:", output.shape)  # (2, 128, vocab_size)

def run_single_token_forward_with_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer = AutoTokenizer.from_pretrained("/home/jobuser/.cache/huggingface/hub/models--openai-community--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2/")
    input = tokenizer("Once upon a time, there was a magical forest", return_tensors="pt", padding="max_length", max_length=128)
    print(input)

    attention_mask = input["attention_mask"]
    # 获取真实输入的长度（即非 pad 的 token 数）
    real_len = attention_mask[0].sum().item()

    model = GPT(context_len=128, vocab_size=tokenizer.vocab_size, hidden_dim=512)
    outputs = model(input.input_ids)
    print("Output shape:", outputs.shape)
    
    # 获取该位置的预测
    predicted_id = torch.argmax(outputs[0, real_len - 1])
    print("Predicted token:", tokenizer.decode([predicted_id.item()]))


if __name__ == "__main__":
    # Run Simple Forward Before Training
    run_single_token_forward()
    run_single_token_forward_with_tokenizer()


"""
This script added support of:
1. Add naive GPT Model
2. Support Single Token Forward
3. Support Single Token Forward with Tokenizer


TODO:
1. Add a loss function to the GPT Model
2. Add a training loop to the GPT Model
3. Support Attention Masking for Training
4. Support Attention Masking for Inference
5. Support Auto-regressive Inference
6. Support KV Cache for Inference
"""