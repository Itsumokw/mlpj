import torch

class Function:
    @staticmethod
    def apply(*args):
        raise NotImplementedError

class Add(Function):
    @staticmethod
    def apply(a, b):
        # Import Tensor at runtime to avoid circular import
        from .tensor import Tensor
        
        out = Tensor(a.data + b.data)
        if a.requires_grad or b.requires_grad:
            out.requires_grad = True
            def _backward():
                if a.requires_grad:
                    a.grad = out.grad
                if b.requires_grad:
                    b.grad = out.grad
            out._backward = _backward
        return out

class Multiply(Function):
    @staticmethod
    def apply(a, b):
        # Import Tensor at runtime to avoid circular import
        from .tensor import Tensor
        
        out = Tensor(a.data * b.data)
        if a.requires_grad or b.requires_grad:
            out.requires_grad = True
            def _backward():
                if a.requires_grad:
                    a.grad = b.data * out.grad
                if b.requires_grad:
                    b.grad = a.data * out.grad
            out._backward = _backward
        return out

class ReLU(Function):
    @staticmethod
    def apply(x):
        # Import Tensor at runtime to avoid circular import
        from .tensor import Tensor
        
        out = Tensor(torch.maximum(x.data, torch.tensor(0.)))
        
        if x.requires_grad:
            out.requires_grad = True
            def _backward():
                grad = out.grad * (x.data > 0).float()
                if x.grad is None:
                    x.grad = grad
                else:
                    x.grad += grad
            out._backward = _backward
        return out

class Transpose(Function):
    @staticmethod
    def apply(x, dim0, dim1):
        # Import Tensor at runtime to avoid circular import
        from .tensor import Tensor
        
        out = Tensor(x.data.transpose(dim0, dim1))
        
        if x.requires_grad:
            out.requires_grad = True
            def _backward():
                if x.grad is None:
                    x.grad = out.grad.transpose(dim1, dim0)
                else:
                    x.grad += out.grad.transpose(dim1, dim0)
            out._backward = _backward
        return out

class ElementwiseMultiply(Function):
    @staticmethod
    def apply(a, b):
        # Import Tensor at runtime to avoid circular import
        from .tensor import Tensor
        
        if not isinstance(b, Tensor):
            b = Tensor(b)
        out = Tensor(a.data * b.data)
        
        if a.requires_grad or b.requires_grad:
            out.requires_grad = True
            def _backward():
                if a.requires_grad:
                    grad = b.data * out.grad
                    if a.grad is None:
                        a.grad = grad
                    else:
                        a.grad += grad
                if b.requires_grad:
                    grad = a.data * out.grad
                    if b.grad is None:
                        b.grad = grad
                    else:
                        b.grad += grad
            out._backward = _backward
        return out

class MatMul(Function):
    @staticmethod
    def apply(a, b):
        # Import Tensor at runtime to avoid circular import
        from .tensor import Tensor
        
        out = Tensor(torch.matmul(a.data, b.data))
        
        if a.requires_grad or b.requires_grad:
            out.requires_grad = True
            def _backward():
                if a.requires_grad:
                    if a.grad is None:
                        a.grad = torch.matmul(out.grad, b.data.transpose(-1, -2))
                    else:
                        a.grad += torch.matmul(out.grad, b.data.transpose(-1, -2))
                if b.requires_grad:
                    if b.grad is None:
                        b.grad = torch.matmul(a.data.transpose(-1, -2), out.grad)
                    else:
                        b.grad += torch.matmul(a.data.transpose(-1, -2), out.grad)
            out._backward = _backward
        return out

class View(Function):
    @staticmethod
    def apply(x, shape):
        # Import Tensor at runtime to avoid circular import
        from .tensor import Tensor
        
        out = Tensor(x.data.view(*shape))
        
        if x.requires_grad:
            out.requires_grad = True
            def _backward():
                if x.grad is None:
                    x.grad = out.grad.view(x.shape)
                else:
                    x.grad += out.grad.view(x.shape)
            out._backward = _backward
        return out