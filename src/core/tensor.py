import numpy as np
import torch
from .autograd import Add, Multiply, ReLU, Transpose, ElementwiseMultiply, MatMul, View

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
        return Add.apply(self, other)
        
    def __mul__(self, other):
        return Multiply.apply(self, other)
        
    def __matmul__(self, other):
        return MatMul.apply(self, other)
        
    def transpose(self, dim0, dim1):
        return Transpose.apply(self, dim0, dim1)
        
    def relu(self):
        return ReLU.apply(self)
        
    def view(self, *shape):
        return View.apply(self, shape)