import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to('cuda:0')
        
        self.stage2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        ).to('cuda:1')

    def forward(self, x):
        # assume x is on cuda:0
        x = self.stage1(x)  # computation on cuda:0
        
        x = x.to('cuda:1') # transfer to cuda:1
        
        x = self.stage2(x) # computation on cuda:1
        
        x = x.to('cuda:0') # transfer back to cuda:0 if needed
        return x

def main():
    input_dim = 16
    hidden_dim = 64
    output_dim = 32
    print("Model parameters:")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Output dimension: {output_dim}")

    model = SimpleModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    x = torch.randn(2, 1, 16).to('cuda:0')
    output = model(x)
    print(output)
    print(output.shape)
    print(output.device)

if __name__ == "__main__":
    main()
