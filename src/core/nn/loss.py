from ..tensor import Tensor
import torch

class MSELoss:
    """Mean Squared Error Loss."""
    
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        error = pred.data - target.data
        loss = Tensor((error ** 2).mean())
        
        if pred.requires_grad:
            def _backward():
                grad = 2 * error / error.numel()
                if pred.grad is None:
                    pred.grad = grad
                else:
                    pred.grad += grad
            loss._backward = _backward
            
        return loss