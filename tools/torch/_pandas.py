import torch
import pandas as pd
from torch import Tensor

__all__ = [
'to_csv',
]

def to_csv(tensor: torch.Tensor, path: str):
    assert type(tensor)==torch.Tensor
    assert len(tensor.shape) <= 2, f'Must pass 2-d input. shape={tensor.shape}'
    pd.DataFrame(tensor.detach().cpu().numpy()).to_csv(path, index=False)
