import torch
from torch import nn
from typing import Iterable

def clip_grad(parameters: Iterable[nn.Parameter], max_norm: float, eps: float = 1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]

    if len(grads) == 0: 
        return
    grad_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]))
    
    if grad_norm < max_norm:
        return
    else:
        clip_rate =  max_norm / (grad_norm + eps)
        
        for g in grads:
            g.detach().mul_(clip_rate)