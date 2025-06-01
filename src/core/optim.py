import torch
from .tensor import Tensor

class Optimizer:
    """Base class for all optimizers."""
    
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr
        
    def zero_grad(self):
        """Zero out the gradients of all parameters."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None
                
    def step(self):
        """Update parameters using their gradients."""
        raise NotImplementedError

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(self, parameters, lr=0.01, momentum=0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.velocity = [torch.zeros_like(p.data) for p in self.parameters]
        
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            if self.momentum > 0:
                self.velocity[i] = (self.momentum * self.velocity[i] + 
                                  param.grad)
                param.data -= self.lr * self.velocity[i]
            else:
                param.data -= self.lr * param.grad

class Adam(Optimizer):
    """Adam optimizer."""
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        
        # Initialize momentum and velocity terms
        self.m = [torch.zeros_like(p.data) for p in self.parameters]
        self.v = [torch.zeros_like(p.data) for p in self.parameters]
        
    def step(self):
        self.t += 1
        beta1, beta2 = self.betas
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
                
            # Update biased first and second moment estimates
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * param.grad
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * param.grad ** 2
            
            # Compute bias-corrected first and second moment estimates
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            
            # Update parameters
            param.data -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

class AdaGrad(Optimizer):
    """Adaptive Gradient optimizer."""
    def __init__(self, parameters, lr=0.01, eps=1e-10):
        super().__init__(parameters, lr)
        self.eps = eps
        self.square_grads = [torch.zeros_like(p.data) for p in self.parameters]
        
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            self.square_grads[i] += param.grad ** 2
            param.data -= self.lr * param.grad / (torch.sqrt(self.square_grads[i]) + self.eps)

class RMSprop(Optimizer):
    """Root Mean Square Propagation optimizer."""
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.square_avg = [torch.zeros_like(p.data) for p in self.parameters]
        
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            self.square_avg[i] = self.alpha * self.square_avg[i] + \
                                (1 - self.alpha) * param.grad ** 2
            param.data -= self.lr * param.grad / \
                         (torch.sqrt(self.square_avg[i]) + self.eps)

class AdaDelta(Optimizer):
    """Adaptive Delta optimizer."""
    def __init__(self, parameters, rho=0.95, eps=1e-7):
        super().__init__(parameters, lr=1.0)  # Learning rate is adaptive
        self.rho = rho
        self.eps = eps
        self.square_avg = [torch.zeros_like(p.data) for p in self.parameters]
        self.acc_delta = [torch.zeros_like(p.data) for p in self.parameters]
        
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            # Accumulate gradient
            self.square_avg[i] = self.rho * self.square_avg[i] + \
                                (1 - self.rho) * param.grad ** 2
            
            # Compute update
            std = torch.sqrt(self.acc_delta[i] + self.eps)
            delta = torch.sqrt(self.square_avg[i] + self.eps)
            update = param.grad * std / delta
            
            # Accumulate updates
            self.acc_delta[i] = self.rho * self.acc_delta[i] + \
                               (1 - self.rho) * update ** 2
            
            # Apply update
            param.data -= update