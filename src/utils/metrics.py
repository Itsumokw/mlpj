import numpy as np
import torch
from src.core.tensor import Tensor

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Squared Error between true and predicted values."""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Absolute Error between true and predicted values."""
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared score."""
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total) if ss_total > 0 else 0

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the accuracy of predictions."""
    return np.mean(y_true == y_pred)

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    return Tensor(torch.mean(torch.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    """Root Mean Square Error"""
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    return Tensor(torch.sqrt(torch.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    return Tensor(torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100)

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    if isinstance(y_true, Tensor):
        y_true = y_true.data
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.data
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2
    return Tensor(torch.mean(numerator / denominator) * 100)