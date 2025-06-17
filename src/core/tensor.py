import numpy as np
import torch
# from .autograd import ReLU, Transpose, ElementwiseMultiply, MatMul, View

class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, Tensor):
            self.data = data.data
        elif isinstance(data, torch.Tensor):
            self.data = data
        else:
            self.data = torch.tensor(data, dtype=torch.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None

    def __sub__(self, other):
        """Subtraction operation."""
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = torch.tensor(other, dtype=self.data.dtype)
        return Tensor(self.data - other_data, requires_grad=self.requires_grad)
    
    def __add__(self, other):
        """Addition operation."""
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = torch.tensor(other, dtype=self.data.dtype)
        return Tensor(self.data + other_data, requires_grad=self.requires_grad)
    
    def __mul__(self, other):
        """Multiplication operation."""
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = torch.tensor(other, dtype=self.data.dtype)
        return Tensor(self.data * other_data, requires_grad=self.requires_grad)
    
    def __truediv__(self, other):
        """Division operation."""
        if isinstance(other, Tensor):
            other_data = other.data
        else:
            other_data = torch.tensor(other, dtype=self.data.dtype)
        return Tensor(self.data / other_data, requires_grad=self.requires_grad)
    
    @property
    def shape(self):
        """Get tensor shape."""
        return self.data.shape

    def reshape(self, *shape):
        """Reshape the tensor."""
        return Tensor(self.data.reshape(*shape), requires_grad=self.requires_grad)
    
    def __getitem__(self, idx):
        """Support indexing operations."""
        return Tensor(self.data[idx], requires_grad=self.requires_grad)
        
    def __setitem__(self, idx, value):
        """Support assignment operations."""
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = value
            
    def numpy(self):
        """Convert to numpy array."""
        return self.data.cpu().numpy() if isinstance(self.data, torch.Tensor) else self.data
        
    def backward(self, gradient=None):
        if gradient is None:
            gradient = torch.ones_like(self.data)
            
        if self.requires_grad:
            if self.grad is None:
                self.grad = gradient
            else:
                self.grad += gradient  # Gradient accumulation
            self._backward()
            
    def __matmul__(self, other):
        from .autograd import MatMul
        return MatMul.apply(self, other)
        
    def transpose(self, dim0, dim1):
        from .autograd import Transpose
        return Transpose.apply(self, dim0, dim1)
        
    def relu(self):
        from .autograd import ReLU
        return ReLU.apply(self)
        
    def view(self, *shape):
        from .autograd import View
        return View.apply(self, shape)
        
    def item(self):
        """Get the single value from a scalar tensor."""
        return self.data.item()
        
    def squeeze(self, dim=None):
        """Remove single-dimensional entries from the shape of the tensor.
        
        Args:
            dim (int, optional): Dimension to squeeze. If None, squeezes all
                single-dimensional entries.
                
        Returns:
            Tensor: A new tensor with all or a specific dimension of size 1 removed
        """
        if dim is None:
            return Tensor(self.data.squeeze(), requires_grad=self.requires_grad)
        return Tensor(self.data.squeeze(dim), requires_grad=self.requires_grad)
    
    def unsqueeze(self, dim):
        """Add a singleton dimension at the specified position.
        
        Args:
            dim: Position where to insert singleton dimension
            
        Returns:
            Tensor with added dimension
        """
        return Tensor(self.data.unsqueeze(dim), requires_grad=self.requires_grad)