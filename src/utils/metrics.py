import numpy as np

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