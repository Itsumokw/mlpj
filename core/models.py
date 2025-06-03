import numpy as np
import torch
import torch.nn as nn

class SimpleARIMA:
    """简化版ARIMA模型（仅实现AR部分）"""
    def __init__(self, p=3):
        self.p = p  # 自回归阶数
        self.coef_ = None

    def fit(self, X: np.ndarray):
        # 生成滞后矩阵
        X = np.asarray(X)
        X_lag = np.lib.stride_tricks.sliding_window_view(X, self.p)[:-1]
        y = X[self.p:]
        
        # 最小二乘求解
        self.coef_ = np.linalg.lstsq(X_lag, y, rcond=None)[0]

    def predict(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        predictions = []
        current = X[-self.p:].copy()
        for _ in range(steps):
            pred = np.dot(current, self.coef_)
            predictions.append(pred)
            current = np.roll(current, -1)
            current[-1] = pred
        return np.array(predictions)

class LiteTCN(nn.Module):
    """极简时序卷积网络"""
    def __init__(self, input_size=1, hidden_size=8, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(hidden_size, 1, kernel_size, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: (batch, seq_len)
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x.squeeze(1)  # (batch, seq_len)