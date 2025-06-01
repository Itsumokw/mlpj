import numpy as np
import torch

class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, torch.Tensor):
            self.data = data
        else:
            self.data = torch.tensor(data, dtype=torch.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self.shape = self.data.shape
        
    def backward(self, gradient=None):
        if gradient is None:
            gradient = torch.ones_like(self.data)
            
        if self.requires_grad:
            if self.grad is None:
                self.grad = gradient
            else:
                self.grad += gradient  # Gradient accumulation
            self._backward()
            
    def __add__(self, other):
        from .autograd import Add
        return Add.apply(self, other)
        
    def __mul__(self, other):
        from .autograd import Multiply
        return Multiply.apply(self, other)
        
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