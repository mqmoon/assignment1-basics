import torch
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, device=device, dtype=dtype))
    
    def reset_parameters(self):
        std = math.sqrt(2 / (self.in_features + self.out_features))
        lower_bound = -3.0 * std
        upper_bound = 3.0 * std
        torch.nn.init.trunc_normal_(self.weight, mean=0.0,std=std,a=lower_bound,b=upper_bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
    
    def extra_repr(self):
        return f'input_dim={self.in_features}, output_dim={self.out_features}, bias=False'