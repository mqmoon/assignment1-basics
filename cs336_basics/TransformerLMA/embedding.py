import torch
import math

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0
        lower_bound = -3.0
        upper_bound = 3.0
        torch.nn.init.trunc_normal_(self.weight, mean=0.0,std=std,a=lower_bound,b=upper_bound)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]