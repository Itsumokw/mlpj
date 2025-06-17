import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super().__init__()
        # 计算正确的填充量以保持序列长度不变
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1,
            self.conv2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
        # 添加裁剪层以保持序列长度不变
        self.crop = kernel_size - 1
        
    def forward(self, x):
        out = self.net(x)
        
        # 裁剪序列两端以保持长度不变
        if self.crop > 0:
            out = out[:, :, self.crop:-self.crop]
        
        res = x if self.downsample is None else self.downsample(x)
        
        # 确保残差连接尺寸匹配
        if res.size(2) > out.size(2):
            res = res[:, :, :out.size(2)]
        elif res.size(2) < out.size(2):
            out = out[:, :, :res.size(2)]
            
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, 
                stride=1, dilation=dilation_size, 
                dropout=dropout
            ))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class TCN(nn.Module, BaseEstimator):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, 
                 lr=0.001, epochs=200, batch_size=16):
        super().__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        
        self.input_size = input_size
        self.output_size = output_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 转换为 (batch_size, input_size, sequence_length)
        y = self.tcn(x)
        y = self.linear(y[:, :, -1])
        return y
    
    def fit(self, X, y):
        """训练模型"""
        self.to(self.device)
        
        # 转换为PyTorch张量
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(self.device)
        
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
            # 直接转换为张量，不添加额外维度
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            
            # 如果输入是2D (batch_size, seq_len)，添加通道维度
            if X_tensor.dim() == 2:
                X_tensor = X_tensor.unsqueeze(-1)
                
            predictions = self(X_tensor).cpu().numpy().flatten()
        return predictions