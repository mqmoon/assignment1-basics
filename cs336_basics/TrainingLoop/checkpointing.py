import torch
import os
from typing import Union, BinaryIO, IO

def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    torch.save(checkpoint, out)
    print(f'Checkpoint saved at iteration: {iteration}.')

def load_checkpoint(src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']