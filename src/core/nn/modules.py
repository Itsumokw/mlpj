from typing import Optional, List, Dict
from ..tensor import Tensor
from ...utils.type_checks import check_tensor
from ...utils.initializers import kaiming_normal, xavier_uniform
from . import functional as F  # Add this import
from ..base import BaseModel  # Add this import
import torch
import numpy as np
import math

class Module:
    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self._initialized: bool = False
        
    def parameters(self) -> List[Tensor]:
        """Return all parameters of the module."""
        return list(self._parameters.values())
        
    def zero_grad(self) -> None:
        """Zero out gradients of all parameters."""
        for param in self.parameters():
            if param.grad is not None:
                param.grad = None
                
    def train(self) -> None:
        """Set the module in training mode."""
        self.training = True
        
    def eval(self) -> None:
        """Set the module in evaluation mode."""
        self.training = False
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using Kaiming initialization
        self.weight = Tensor(
            kaiming_normal(torch.empty(out_features, in_features)),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(
                torch.zeros(out_features),
                requires_grad=True
            )
        else:
            self.bias = None
            
        self._parameters = {'weight': self.weight}
        if self.bias is not None:
            self._parameters['bias'] = self.bias
        
    def forward(self, x: Tensor) -> Tensor:
        # Input shape check
        check_tensor(x, dim=3)  # (batch_size, seq_len, in_features)
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Expected input features {self.in_features}, got {x.shape[-1]}")
            
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out

class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = Tensor(
            torch.randn(out_channels, in_channels, kernel_size) / np.sqrt(in_channels * kernel_size),
            requires_grad=True
        )
        self.bias = Tensor(
            torch.zeros(out_channels),
            requires_grad=True
        )
        self._parameters = {'weight': self.weight, 'bias': self.bias}
        
    def forward(self, x):
        batch_size, in_channels, seq_len = x.shape
        if self.padding:
            pad_size = self.padding if isinstance(self.padding, int) else self.padding[0]
            x = F.pad(x, (pad_size, pad_size))
            
        out_len = (seq_len + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = torch.zeros(batch_size, self.out_channels, out_len)
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    for i in range(0, seq_len - self.kernel_size + 1, self.stride):
                        out[b, c_out, i//self.stride] += torch.sum(
                            x[b, c_in, i:i+self.kernel_size] * self.weight[c_out, c_in]
                        )
        return Tensor(out + self.bias.view(1, -1, 1))

class BatchNorm1d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Parameters
        self.gamma = Tensor(torch.ones(num_features), requires_grad=True)
        self.beta = Tensor(torch.zeros(num_features), requires_grad=True)
        
        # Running estimates
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)
        
        self._parameters = {'gamma': self.gamma, 'beta': self.beta}
        
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = x.mean(dim=(0, 2))
            var = x.var(dim=(0, 2), unbiased=False)
            
            # Update running estimates
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        # Normalize
        x_norm = (x - mean[None, :, None]) / torch.sqrt(var[None, :, None] + self.eps)
        return self.gamma[None, :, None] * x_norm + self.beta[None, :, None]

class LayerNorm(Module):
    """Layer Normalization."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.weight = Tensor(torch.ones(normalized_shape), requires_grad=True)
        self.bias = Tensor(torch.zeros(normalized_shape), requires_grad=True)
        
        self._parameters = {'weight': self.weight, 'bias': self.bias}
        
    def forward(self, x: Tensor) -> Tensor:
        # Calculate mean and variance along last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize and scale
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias

class ResidualBlock(Module):
    """Residual block with skip connection."""
    
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = Conv1d(channels, channels, kernel_size, padding='same')
        self.conv2 = Conv1d(channels, channels, kernel_size, padding='same')
        
    def forward(self, x):
        identity = x
        out = self.conv1(x).relu()
        out = self.conv2(out)
        return (out + identity).relu()

class Sequential(Module):
    """Enhanced Sequential container with improved functionality."""
    
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for idx, module in enumerate(args):
            self.add_module(f'module_{idx}', module)
            
    def add_module(self, name, module):
        """Add a module to the container."""
        self.modules.append(module)
        setattr(self, name, module)
        
    def insert(self, index, module):
        """Insert a module at specified position."""
        self.modules.insert(index, module)
        for idx, mod in enumerate(self.modules):
            setattr(self, f'module_{idx}', mod)
            
    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x
        
    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

class GlobalAveragePool(Module):
    """Global Average Pooling layer."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, x: Tensor) -> Tensor:
        # Input shape: (batch_size, channels, seq_len)
        # Output shape: (batch_size, channels)
        return Tensor(torch.mean(x.data, dim=2))

class ResNet(BaseModel):
    """Residual Network for time series."""
    
    def __init__(self, input_size=1, feature_size=0, hidden_size=32, num_blocks=3):
        super().__init__()
        total_input = input_size + feature_size
        
        # Initial convolution
        self.conv1 = Conv1d(total_input, hidden_size, kernel_size=3, padding='same')
        
        # Residual blocks
        self.blocks = Sequential(*[
            ResidualBlock(hidden_size)
            for _ in range(num_blocks)
        ])
        
        # Final layers
        self.global_pool = GlobalAveragePool()
        self.fc = Linear(hidden_size, 1)
        
    def forward(self, x, features=None):
        if features is not None:
            x = torch.cat([x, features], dim=-1)
        x = x.transpose(1, 2)  # (batch_size, channels, seq_len)
        
        x = self.conv1(x).relu()
        x = self.blocks(x)
        x = self.global_pool(x)
        return self.fc(x)

class LSTM(Module):
    """Long Short-Term Memory layer"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Gates weights
        self.weights = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            w_ih = Tensor(
                xavier_uniform(torch.empty(4 * hidden_size, layer_input_size)),
                requires_grad=True
            )
            w_hh = Tensor(
                xavier_uniform(torch.empty(4 * hidden_size, hidden_size)),
                requires_grad=True
            )
            self.weights.extend([w_ih, w_hh])
            
    def forward(self, x, initial_states=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        if initial_states is None:
            h = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]
        else:
            h, c = initial_states
            
        output = []
        for t in range(seq_len):
            x_t = x[:, t]
            for layer in range(self.num_layers):
                w_ih = self.weights[layer * 2]
                w_hh = self.weights[layer * 2 + 1]
                
                gates = (x_t @ w_ih.T + h[layer] @ w_hh.T)
                i, f, g, o = gates.chunk(4, dim=1)
                
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)
                
                c[layer] = f * c[layer] + i * g
                h[layer] = o * torch.tanh(c[layer])
                x_t = h[layer]
                
            output.append(h[-1])
            
        return torch.stack(output, dim=1), (h, c)

class Attention(Module):
    """Multi-head attention layer"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections and reshape for multi-head
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = attention @ v
        
        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        return self.out_proj(out)

class ReLU(Module):
    """Rectified Linear Unit activation function."""
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        if self.inplace and isinstance(x.data, torch.Tensor):
            x.data = torch.relu(x.data)
            return x
        return Tensor(torch.relu(x.data))

class Dropout(Module):
    """Randomly zeroes some of the elements with probability p."""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, got {p}")
        self.p = p
        self.training = True
        
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
            
        # Create dropout mask
        mask = torch.bernoulli(torch.full_like(x.data, 1 - self.p))
        # Scale the remaining values
        scale = 1.0 / (1 - self.p)
        return Tensor(x.data * mask * scale)

class ModuleList(Module):
    """Holds submodules in a list."""
    
    def __init__(self, modules=None):
        super().__init__()
        self.modules = []
        if modules is not None:
            self.extend(modules)
            
    def append(self, module: Module) -> None:
        """Append a module to the list."""
        self.modules.append(module)
        
    def extend(self, modules) -> None:
        """Extend the list with a sequence of modules."""
        for module in modules:
            self.append(module)
            
    def __getitem__(self, idx: int) -> Module:
        return self.modules[idx]
        
    def __len__(self) -> int:
        return len(self.modules)
        
    def __iter__(self):
        return iter(self.modules)
        
    def parameters(self) -> List[Tensor]:
        """Return parameters from all modules."""
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params