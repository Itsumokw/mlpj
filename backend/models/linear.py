import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator

class TimeLinear(nn.Module, BaseEstimator):
    def __init__(self, input_size, output_size, lr=0.01, epochs=100):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        
        self.input_size = input_size
        self.output_size = output_size
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        return self.linear(x)
    
    def fit(self, X, y):
        """训练模型"""
        self.to(self.device)
        
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        loss_history = []
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            
            outputs = self(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            
            loss.backward()
            optimizer.step()
            
            loss_history.append(loss.item())
        
        return loss_history
    
    def predict(self, X):
        """预测"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            predictions = self(X_tensor).cpu().numpy().flatten()
        return predictions