from typing import Union, Tuple, List
import torch
from ..core.tensor import Tensor

def check_tensor(x: Union[torch.Tensor, Tensor], shape: Tuple[int, ...] = None, 
                dim: int = None, dtype: torch.dtype = None) -> None:
    """Check tensor properties."""
    if isinstance(x, Tensor):
        x = x.data
    
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
    if shape is not None and x.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {x.shape}")
        
    if dim is not None and len(x.shape) != dim:
        raise ValueError(f"Expected {dim} dimensions, got {len(x.shape)}")
        
    if dtype is not None and x.dtype != dtype:
        raise ValueError(f"Expected dtype {dtype}, got {x.dtype}")