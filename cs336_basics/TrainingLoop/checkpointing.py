import torch
import os
from typing import Union, BinaryIO, IO

def save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    iteration: int, 
                    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]):
    # checkpoint = {
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'iteration': iteration,
    # }
    # torch.save(checkpoint, out)
    # print(f'Checkpoint saved at iteration: {iteration}.')

    final_path = out
    # 我们只在 'out' 是一个路径字符串时才修改文件名
    if isinstance(out, (str, os.PathLike)):
        # 将路径分解为目录、主文件名和扩展名
        directory = os.path.dirname(out)
        base_filename, extension = os.path.splitext(os.path.basename(out))
        # 创建包含迭代次数的新文件名
        # 例如: "ckpt.pth" -> "ckpt-1000.pth"
        new_filename = f"{base_filename}-{iteration}{extension}"
        # 重新组合成最终的完整路径
        final_path = os.path.join(directory, new_filename)
        # 确保目录存在
        if directory:
            os.makedirs(directory, exist_ok=True)
    # 创建要保存的状态字典
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
    }
    # 使用最终路径保存检查点
    torch.save(checkpoint, final_path)
    # 根据我们是否有路径字符串来调整打印信息
    if isinstance(final_path, (str, os.PathLike)):
        print(f'Checkpoint saved to: {final_path}')
    else:
        print(f'Checkpoint saved at iteration: {iteration}.')

def load_checkpoint(src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']