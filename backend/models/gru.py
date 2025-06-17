import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator

class TimeGRU(nn.Module, BaseEstimator):
    def __init__(self, input_size, hidden_size, num_layers, output_size, 
                 dropout=0.1, lr=0.001, epochs=200, batch_size=16):
        super().__init__()
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size, output_size)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        # x形状: (batch_size, seq_len, input_size)
        out, _ = self.gru(x)
        # 只取最后一个时间步的输出
        out = self.linear(out[:, -1, :])
        return out
    
    def fit(self, X, y):
        """训练模型"""
        self.to(self.device)
        
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)
        
        if X_tensor.dim() == 2:
            # 添加特征维度: (batch, seq_len) -> (batch, seq_len, 1)
            X_tensor = X_tensor.unsqueeze(-1)
        # 创建数据集
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        
        loss_history = []
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            self.train()
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = self(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_x.size(0)
            
            epoch_loss /= len(dataloader.dataset)
            loss_history.append(epoch_loss)
        
        return loss_history
    
    def predict(self, X):
        """预测"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(-1)
            predictions = self(X_tensor).cpu().numpy().flatten()
        return predictions