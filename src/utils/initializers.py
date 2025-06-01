import math
from typing import Tuple
import torch

def xavier_uniform(tensor: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Xavier uniform initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return torch.nn.init.uniform_(tensor, -a, a)

def kaiming_normal(tensor: torch.Tensor, mode: str = 'fan_in') -> torch.Tensor:
    """Kaiming normal initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == 'fan_in' else fan_out
    std = math.sqrt(2.0 / float(fan))
    return torch.nn.init.normal_(tensor, 0, std)

def _calculate_fan_in_and_fan_out(tensor: torch.Tensor) -> Tuple[int, int]:
    """Calculate fan in and fan out."""
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can't be computed for tensor with < 2 dimensions")
        
    if dimensions == 2:  # Linear
        fan_in, fan_out = tensor.shape[1], tensor.shape[0]
    else:  # Convolution
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        for s in tensor.shape[2:]:
            receptive_field_size *= s
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out