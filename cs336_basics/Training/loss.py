import torch
from torch import Tensor
from jaxtyping import Float, Int

def run_cross_entropy_loss(logits: Float[Tensor, "... vocab_size"],
                           targets: Int[Tensor, "..."]) -> Float[Tensor, ""]:
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    subtracted_logits = logits - max_logits
    sum_exp = torch.sum(torch.exp(subtracted_logits), dim=-1)
    log_sum_exp = torch.log(sum_exp) + max_logits.squeeze(-1)
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss_per_token = log_sum_exp - target_logits
    return loss_per_token.mean()