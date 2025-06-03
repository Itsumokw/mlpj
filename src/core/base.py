import torch
import torch.nn as nn
from typing import Optional, List, Dict
import numpy as np


class BaseModel(nn.Module):
    """Base class for all models with enhanced parameter tracking"""

    def __init__(self):
        super().__init__()
        self.is_fitted = False
        self._parameters: Dict[str, nn.Parameter] = {}
        self.param_history: List[Dict] = []  # Track parameter evolution
        self.batch_losses: List[float] = []  # Track batch-level losses

    def register_parameter(self, name: str, param: Optional[nn.Parameter]) -> None:
        if param is not None:
            self._parameters[name] = param

    def forward(self, x, features=None):
        raise NotImplementedError("Forward method not implemented.")

    def fit(self, data, labels):
        raise NotImplementedError("Fit method not implemented.")

    def predict(self, data):
        raise NotImplementedError("Predict method not implemented.")

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        self.load_state_dict(torch.load(filepath))

    def record_parameters(self, epoch: int):
        """Record model parameters for visualization"""
        param_snapshot = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                param_data = param.detach().cpu().numpy()
                param_snapshot[name] = {
                    'mean': float(np.mean(param_data)),
                    'std': float(np.std(param_data)),
                    'min': float(np.min(param_data)),
                    'max': float(np.max(param_data)),
                    'values': param_data.flatten()[:100]  # Sample first 100 values
                }
        self.param_history.append(param_snapshot)

    def record_batch_loss(self, loss: float):
        """Record batch-level loss"""
        self.batch_losses.append(loss)