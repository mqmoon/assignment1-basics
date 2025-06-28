import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str):
    max_start_index = len(x) - context_length - 1
    start_indices = np.random.randint(0, max_start_index + 1, size=(batch_size,))
    input_seqs = [x[i:i+context_length] for i in start_indices]
    target_seqs = [x[i+1:i+context_length+1] for i in start_indices]
    inputs_np = np.stack(input_seqs)
    targets_np = np.stack(target_seqs)
    inputs = torch.from_numpy(inputs_np).to(torch.long).to(device)
    targets = torch.from_numpy(targets_np).to(torch.long).to(device)
    return inputs, targets