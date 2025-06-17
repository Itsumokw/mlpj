import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator
import logging

# 配置日志
logger = logging.getLogger(__name__)

class NeuralARIMA(nn.Module, BaseEstimator):
    def __init__(self, input_size, hidden_size, output_size, lr=0.005, epochs=500, alpha=0.1, q=1, p=12):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        self.lr = lr
        self.epochs = epochs
        self.alpha = alpha
        self.q = q
        self.p = p
        if q > 0:
            self.register_buffer('recent_errors', torch.zeros(q))
        else:
            self.recent_errors = None
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized NeuralARIMA with input_size={input_size}")
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def fit(self, X, y):
        """训练模型"""
        # 检查输入数据
        if X.size == 0:
            logger.error("Received empty input data")
            return [0.0] * self.epochs  # 返回空损失历史
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            logger.info(f"Reshaped input from {X.shape} to 2D array")
        
        # 验证特征维度
        if X.shape[1] != self.input_size:
            logger.warning(
                f"Input feature size {X.shape[1]} does not match model input size {self.input_size}. "
                f"Adjusting model input size to {X.shape[1]}"
            )
            self.fc1 = nn.Linear(X.shape[1], self.fc1.out_features)
            self.input_size = X.shape[1]
        
        self.to(self.device)
        
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        # 初始化误差张量
        model_errors = torch.tensor([], device=self.device)
        
        loss_history = []
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self(X_tensor)
            
            # 计算当前批次的误差
            batch_errors = (outputs - y_tensor).detach().squeeze()
            
            # 更新全局误差记录
            if model_errors.numel() > 0:
                model_errors = torch.cat((model_errors, batch_errors))
            else:
                model_errors = batch_errors
                
            # 计算损失
            mse_loss = self.criterion(outputs, y_tensor)
            
            # MA损失 - 基于历史误差
            ma_loss = torch.tensor(0.0, device=self.device)
            if len(model_errors) >= self.q:
                recent_errors = model_errors[-self.q:]
                ma_loss = torch.mean(recent_errors ** 2)
            
            total_loss = mse_loss + self.alpha * ma_loss
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            loss_history.append(total_loss.item())
            
            # 限制历史误差大小
            if len(model_errors) > 1000:
                model_errors = model_errors[-1000:]
            
            # 每50轮记录一次
            if (epoch + 1) % 50 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss.item():.6f}")
        if self.q > 0:
            with torch.no_grad():
                outputs = self(X_tensor)
                errors = (outputs - y_tensor).squeeze()
                if len(errors) >= self.q:
                    self.recent_errors = errors[-self.q:].clone().detach()
                else:
                    # 样本不足时用0填充
                    self.recent_errors = torch.zeros(self.q, device=self.device)
        return loss_history
    
    def predict(self, X):
        """预测"""
        if X.size == 0:
            logger.error("Received empty input for prediction")
            return np.array([])
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self(X_tensor).cpu().numpy().flatten()
        return predictions