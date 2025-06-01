import sys
import os

# 获取项目根目录路径
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到Python路径
sys.path.append(project_dir)

import numpy as np
import torch
from src.core.models import SimpleARIMA, LiteTCN
from src.core.features import FeatureExtractor
from src.core.optim import Adam  # 导入自定义优化器
from src.data.dataset import Dataset
from src.data.dataloader import DataLoader
from src.utils.metrics import mean_squared_error
from src.visualization.plotter import plot_predictions

# Hyperparameters
window_size = 30
batch_size = 16
epochs = 100
learning_rate = 0.001

# Load dataset
data = np.sin(np.linspace(0, 20, 1000)) + np.random.normal(0, 0.1, 1000)
dataset = Dataset(data[:-window_size], data[window_size:])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize feature extractor
feature_extractor = FeatureExtractor(window_size)

# Initialize model - 修改模型输入维度，原始维度 + 特征数量
model = SimpleARIMA(p=5, feature_size=feature_extractor.feature_size())  # 新增feature_size参数

# 确保模型参数启用梯度跟踪
for param in model.parameters():
    param.requires_grad = True  # 关键修复：启用梯度跟踪

# 使用自定义Adam优化器
optimizer = Adam(model.parameters(), lr=learning_rate)  # 使用自定义优化器

# Training loop
for epoch in range(epochs):
    for raw_data, targets in dataloader:
        # 提取特征
        features = feature_extractor.transform(raw_data.numpy())
        
        # 合并原始数据和特征作为模型输入
        combined_input = np.concatenate([
            raw_data.numpy(),       # 原始数据: (batch_size, window_size)
            features                # 提取的特征: (batch_size, num_features)
        ], axis=1)
        
        # 转换为张量
        inputs = torch.tensor(combined_input, dtype=torch.float32)
        targets = torch.tensor(targets.numpy(), dtype=torch.float32)

        # 前向传播
        preds = model.predict(inputs)
        
        # 计算损失
        loss = mean_squared_error(targets, preds)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 可视化结果
plot_predictions(targets.numpy(), preds.detach().numpy())