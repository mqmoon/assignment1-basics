import torch
from torch.nn import Module as Module
from torch.nn import Parameter as Parameter
from einops import einsum as einsum
from .linear import Linear as Linear
from .embedding import Embedding as Embedding

class RMSNorm(Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        squared = x.pow(2)
        mean_squared = squared.mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_squared + self.eps)
        x = x / rms
        x = x * self.weight
        return x.to(in_dtype)


class SiLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # self.w1 = Parameter(torch.randn(d_ff, d_model, device=device, dtype=dtype))
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        # self.w3 = Parameter(torch.randn(d_ff, d_model, device=device, dtype=dtype))
        # self.w2 = Parameter(torch.randn(d_model, d_ff, device=device, dtype=dtype))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x1 = x1 * torch.sigmoid(x1)
        x3 = self.w3(x)
        return  self.w2(x1 * x3)

class RoPE(Module):
    cos_cache: torch.Tensor
    sin_cache: torch.Tensor
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "d_k must be even."
        dims = torch.arange(0, d_k, 2, device=device)
        inv_freq = 1.0 / (theta ** (dims.float() / d_k))
        t = torch.arange(max_seq_len, device=device)
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # print(token_positions.shape)
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        # print(self.cos_cache.shape)
        # print(self.sin_cache.shape)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        # print(x_even.shape)
        # print(x_odd.shape)

        # 1. 计算 x_even 和 cos 之间的维度差异
        #    x_even.ndim = 4, cos.ndim = 2, 所以 diff = 2
        # diff = x_even.ndim - cos.ndim
        # print(f'diff: {diff}')
        # 2. 构造一个新的形状，在 cos 的前面加上 diff 个大小为 1 的维度
        #    这会生成一个元组，例如 (1, 1, 256, 16)
        # broadcast_shape = (1,) * diff + cos.shape
        # print(f'broadcast_shape: {broadcast_shape}')
        # 3. 使用 view() 或 reshape() 将 cos 和 sin 变为可广播的形状
        # cos_broadcastable = cos.view(broadcast_shape)
        # sin_broadcastable = sin.view(broadcast_shape)
        # print(f'cos_broadcastable.shape: {cos_broadcastable.shape}')
        # print(f'sin_broadcastable.shape: {sin_broadcastable.shape}')

        # --- 现在乘法就可以安全地进行了 ---
        # PyTorch 会自动将 [1, 1, 256, 16] 广播到 [32, 16, 256, 16]
        # x_rotated_even = x_even * cos_broadcastable - x_odd * sin_broadcastable
        # x_rotated_odd  = x_odd  * cos_broadcastable + x_even * sin_broadcastable
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd = x_even * sin + x_odd * cos
        x_rotated = torch.empty_like(x)
        x_rotated[..., 0::2] = x_rotated_even
        x_rotated[..., 1::2] = x_rotated_odd
        return x_rotated


class SafeSoftmax(Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_value = torch.max(x, dim=self.dim, keepdim=True).values
        subtracted_x = x - max_value
        exponential = torch.exp(subtracted_x)
        sum_values = torch.sum(exponential, dim=self.dim, keepdim=True)
        return exponential / sum_values


class ScaledDotProductAttention(Module):
    def __init__(self):
        super().__init__()
        self.safesoftmax = SafeSoftmax(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal_mask = None):
        dot_value = q @ k.transpose(-2, -1)
        pre_score = dot_value / (q.shape[-1] ** 0.5)
        if causal_mask is not None:
            attn_mask = (causal_mask == False)
            pre_score = pre_score.masked_fill(attn_mask, -1e9)
        attn_score = self.safesoftmax(pre_score)

        attn_output = attn_score @ v
        return attn_output


class MultiHeadSelfAttention(Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int | None, theta: float | None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.d_h = self.d_model // self.num_heads
        if max_seq_len is not None and theta is not None:
            self.rope = RoPE(max_seq_len=max_seq_len, theta=theta, d_k=d_model // num_heads)
        self.q_proj = Linear(in_features=d_model, out_features=d_model)
        self.k_proj = Linear(in_features=d_model, out_features=d_model)
        self.v_proj = Linear(in_features=d_model, out_features=d_model)
        self.output_proj = Linear(in_features=d_model, out_features=d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # # split to heads, apply RoPE, multihead_attention, concat and output
        B, S, D = x.shape
        q = q.view(B, S, self.num_heads, self.d_h)
        k = k.view(B, S, self.num_heads, self.d_h)
        v = v.view(B, S, self.num_heads, self.d_h)

        # (B, S, H, d_k) -> (B, H, S, d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if token_positions is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        causal_mask = torch.tril(torch.ones(S, S, device=x.device, dtype=torch.bool))
        attn_output = self.attention(q, k, v, causal_mask)

        # (B, H, S, d_k) -> (B, S, H, d_k)
        attn_output = attn_output.transpose(1, 2)
        
        # .contiguous() 确保张量在内存中是连续的，这是在 view/reshape 前的好习惯
        # (B, S, H, d_k) -> (B, S, D)
        attn_output = attn_output.contiguous().view(B, S, D)
        final_output = self.output_proj(attn_output)
        return final_output


class TransformerBlock(Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model,
            num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
    
    def forward(self, x: torch.Tensor):
        residual = x
        x = self.ln1(x)
        token_positions = torch.arange(x.shape[-2])
        x = self.attn(x, token_positions)
        x = x + residual
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        return x + residual


class TransformerLM(Module):
    def __init__(self, d_model: int, num_heads: int, vocab_size: int, context_length: int, 
        num_layers: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = torch.nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor :
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


@torch.no_grad()
def generate(model: torch.nn.Module, 
             prompt_tokens: torch.Tensor, 
             max_new_tokens: int, 
             temperature: float = 1.0, 
             top_p: float = 1.0,
             end_of_text_token: int =  256):
    """
    Generates text completions from a model given a prompt.

    Args:
        model (nn.Module): The language model.
        prompt_tokens (torch.Tensor): A tensor of shape (B, T) with prompt token IDs.
        max_new_tokens (int): The maximum number of new tokens to generate.
        temperature (float): Controls randomness. Higher values ( > 1.0) make output more random,
                             lower values ( < 1.0) make it more deterministic. 1.0 is neutral.
        top_p (float): Nucleus sampling threshold. Keeps the smallest set of tokens whose
                       cumulative probability is >= top_p. 1.0 disables it.
        end_of_text_token (int): The token ID that signals the end of a sequence.

    Returns:
        torch.Tensor: The generated sequence of token IDs, including the prompt.
    """
    model.eval() # Set the model to evaluation mode
    # The sequence to be generated, starting with the prompt
    idx = prompt_tokens
    for _ in range(max_new_tokens):
        # --- Step 1: Prepare the input ---
        # If the sequence gets too long, crop it to the model's context length
        context_length = model.context_length
        idx_cond = idx if idx.size(1) <= context_length else idx[:, -context_length:]
        # --- Step 2: Forward pass to get logits ---
        logits, _ = model(idx_cond)
        # --- Step 3: Focus on the last logit (prediction for the next token) ---
        logits = logits[:, -1, :]  # Becomes (B, C)
        # --- Step 4: Apply temperature scaling ---
        if temperature <= 0: # Handle temperature=0 as greedy sampling
            temperature = 1.0 # Avoid division by zero, logic below handles it
            probs = torch.zeros_like(logits)
            probs.scatter_(1, torch.argmax(logits, dim=-1, keepdim=True), 1)
        else:
            logits = logits / temperature
            # --- Step 5: Apply softmax to get probabilities ---
            softmax = SafeSoftmax(dim=-1)
            probs = softmax(logits) # (B, C)
        # --- Step 6: Apply Top-p (nucleus) sampling ---
        # This part is only active if top_p is less than 1.0
        if top_p < 1.0 and temperature > 0:
            # Sort probabilities in descending order
            probs_sorted, indices_sorted = torch.sort(probs, descending=True, dim=-1)
            # Calculate cumulative probabilities
            cumulative_probs = torch.cumsum(probs_sorted, dim=-1)            
            # Create a mask for tokens to remove (those that are not in the nucleus)
            # Find the first index where cumulative probability exceeds top_p
            indices_to_remove = cumulative_probs > top_p
            # Shift the mask to the right to keep the first token that exceeds the threshold
            indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
            indices_to_remove[..., 0] = 0 # Never remove the most likely token
            # Create a mask for the original unsorted probabilities
            # We scatter `True` to the indices that should be removed
            remove_mask = torch.zeros_like(probs, dtype=torch.bool).scatter_(
                dim=-1, index=indices_sorted, src=indices_to_remove
            )
            # Set the probability of removed tokens to 0
            probs[remove_mask] = 0           
            # Re-normalize the probabilities so they sum to 1 again
            # Add a small epsilon to avoid division by zero in case all probabilities become zero
            probs = probs / (torch.sum(probs, dim=-1, keepdim=True) + 1e-9)
        # --- Step 7: Sample from the final probability distribution ---
        idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
        # --- Step 8: Append the new token and check for end-of-text ---
        if idx_next.item() == end_of_text_token:
            break # Stop if end-of-text token is generated
        idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

    model.train() # Set model back to training mode
    return idx