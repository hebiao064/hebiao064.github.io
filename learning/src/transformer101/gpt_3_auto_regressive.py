import torch
from torch import nn
from math import sqrt
from transformers import AutoTokenizer
from torch.nn import functional as F

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
    
    def forward(self, x: torch.Tensor, attention_mask=None):
        x = x + self.attn(self.norm_1(x), attention_mask=attention_mask)
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
    
    def forward(self, x: torch.Tensor, attention_mask=None):
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

        # causal mask: (1, 1, seq_len, seq_len)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)).view(1, 1, seq_len, seq_len)

        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Expand to (batch, 1, seq_len, seq_len) for broadcasting
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
            # Apply mask: set padded positions to -inf
            combined_mask = causal_mask & attention_mask
        else:
            combined_mask = causal_mask

        attn = torch.where(causal_mask, attn, torch.zeros_like(attn) - torch.inf)
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
    
    def forward(self, x, attention_mask=None):

        x = self.emb(x) + self.pos_emb[:, :x.size(1), :]

        for block in self.module_list:
            x = block(x, attention_mask=attention_mask)
        
        x = self.norm(x)
        output_vocab = self.lm_head(x)
        
        return output_vocab
    
    '''
    ==============================================================================================
                                        Add Loss Function for Training
    ==============================================================================================
    '''

    def loss(self, tokens, attention_mask=None):
        inputs = tokens[:, :-1] # (batch, seq_len - 1) 
        labels = tokens[:, 1:] # (batch, seq_len - 1)
        logits = self.forward(inputs) # (batch, seq_len - 1, vocab_size)

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
    
    input_ids = torch.randint(0, 10000, (2, 128))  # batch=2, seq_len=128
    output = model(input_ids)
    print("Output shape:", output.shape)  # (2, 128, vocab_size)

def run_single_token_forward_with_tokenizer(model, tokenizer):
    # tokenizer = AutoTokenizer.from_pretrained("/home/jobuser/.cache/huggingface/hub/models--openai-community--gpt2-xl/snapshots/15ea56dee5df4983c59b2538573817e1667135e2/")
    input_texts = ["Once upon a time, there was a magical forest",
                      "She was a beautiful princess"]
    input = tokenizer(input_texts, 
                      return_tensors="pt", 
                      padding="max_length", 
                      max_length=128)

    attention_mask = input["attention_mask"]
    # 获取真实输入的长度（即非 pad 的 token 数）
    real_lens = attention_mask.sum(dim=1)

    model = GPT(context_len=128, vocab_size=tokenizer.vocab_size, hidden_dim=512)
    outputs = model(input.input_ids, attention_mask=attention_mask)
    print("Output shape:", outputs.shape)
    
    # 获取该位置的预测
    predicted_ids = []
    for i in range(len(real_lens)):
        predicted_id = torch.argmax(outputs[i, real_lens[i] - 1])
        print(f"Original text + Predicted token: {input_texts[i] + tokenizer.decode([predicted_id.item()])}")


'''
==============================================================================================
                                    GPT 2 Auto-regressive Inference
==============================================================================================
'''
def generate(model: nn.Module, tokenizer: AutoTokenizer, text: str, max_new_tokens: int):
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token = torch.distributions.Categorical(logits=logits[:, -1, :]).sample()
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

    for i in range(input_ids.shape[0]):
        print(f"Generated text: {tokenizer.decode(input_ids[i])}")

'''
==============================================================================================
                                    GPT 2 Training
==============================================================================================
'''
def train(model: nn.Module, tokenizer: AutoTokenizer):
    training_data = [
        "Once upon a time, there was a magical forest",
        "She was a beautiful princess",
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


if __name__ == "__main__":
    # A Simple GPT Model with random weights
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-xl")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT(context_len=128, vocab_size=tokenizer.vocab_size, hidden_dim=512)

    # Run Simple Forward Before Training
    run_single_token_forward(model)
    run_single_token_forward_with_tokenizer(model, tokenizer)

    # Run Training
    model = train(model, tokenizer)

    # Run Single Token Forward After Training
    run_single_token_forward(model)
    run_single_token_forward_with_tokenizer(model, tokenizer)

    # Run Auto-regressive Inference
    generate(model, tokenizer, "Once upon a time, there was a magical forest", max_new_tokens=10)



"""
This script added support of:
1. Support Auto-regressive Inference
"""