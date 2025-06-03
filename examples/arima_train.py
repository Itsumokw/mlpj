import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 加载数据
data = pd.read_csv('C:\\Users\\Administrator\\.cache\\kagglehub\\datasets\\rakannimer\\air-passengers\\versions\\1\\AirPassengers.csv')
passengers = data['#Passengers'].values.astype(float)
months = [int(m.split('-')[1]) for m in data['Month']]  # 提取月份信息

# 差分操作函数 - 实现I组件
def difference(data, interval=1):
    return [data[i] - data[i - interval] for i in range(interval, len(data))]

# 逆差分函数
def inverse_difference(last_ob, diff):
    return last_ob + diff

# 一阶差分 + 季节性差分 (12个月周期)
d = 1
D = 1
s = 12

# 原始数据
original = passengers.copy()

# 一阶差分
diff1 = difference(passengers, 1)
# 季节性差分
diff_seasonal = difference(diff1[s:], s)

# 更新月份信息以匹配差分后的数据
months = months[s+1:]  # 去掉前13个月（1阶差分去1，季节性差分去12）

# 创建月份特征矩阵 (one-hot编码)
def create_month_features(months):
    month_features = np.zeros((len(months), 12))
    for i, month in enumerate(months):
        month_features[i, month-1] = 1
    return month_features

month_features = create_month_features(months)

# 数据归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
diff_seasonal_normalized = scaler.fit_transform(np.array(diff_seasonal).reshape(-1, 1)).flatten()

# 创建数据集函数
def create_dataset(data, month_features, p=12, q=1):
    X, y = [], []
    errors = []  # 存储预测误差
    for i in range(p, len(data)):
        # AR特征: 过去p个值
        ar_features = data[i-p:i]
        
        # MA特征: 过去q个预测误差
        ma_features = errors[-q:] if len(errors) >= q else [0]*q
        
        # 季节性特征: 当前月份
        seasonal_feature = month_features[i]
        
        # 合并所有特征
        features = np.concatenate((ar_features, ma_features, seasonal_feature))
        X.append(features)
        y.append(data[i])
        
        # 计算当前预测误差 (使用简单线性模型估计)
        if len(errors) < 10:  # 初始阶段
            errors.append(0)
        else:
            # 实际值减去特征的平均值作为误差估计
            errors.append(data[i] - np.mean(features[:p]))
            
    return np.array(X), np.array(y)

# 设置AR和MA阶数
p = 12  # AR阶数
q = 1   # MA阶数
X, y = create_dataset(diff_seasonal_normalized, month_features, p, q)

# 转换为PyTorch张量
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# 数据集划分
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 定义ARIMA模型
class ARIMAModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 模型参数
input_size = p + q + 12  # AR特征 + MA特征 + 月份特征
hidden_size = 64
output_size = 1
model = ARIMAModel(input_size, hidden_size, output_size)

# 自定义损失函数 (包含MA组件)
class MALoss(nn.Module):
    def __init__(self, q=1, alpha=0.1):
        super().__init__()
        self.q = q
        self.alpha = alpha  # MA项的权重
        self.mse = nn.MSELoss()
        
    def forward(self, predictions, targets, model_errors):
        # 基础MSE损失
        mse_loss = self.mse(predictions, targets)
        
        # MA损失 - 基于历史误差
        ma_loss = torch.tensor(0.0, device=predictions.device)
        
        if len(model_errors) >= self.q:
            # 获取最近q个误差
            recent_errors = model_errors[-self.q:]
            ma_loss = torch.mean(recent_errors ** 2)
        
        # 组合损失
        total_loss = mse_loss + self.alpha * ma_loss
        return total_loss

# 训练参数
criterion = MALoss(q=q, alpha=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
num_epochs = 500

# 训练循环 初始化误差张量（确保在正确的设备上）
device = X_train.device
model_errors = torch.tensor([], device=device)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # 前向传播
    outputs = model(X_train)
    
    # 计算当前批次的误差（张量）
    batch_errors = (outputs - y_train).detach().squeeze()
    
    # 更新全局误差记录
    model_errors = torch.cat((model_errors, batch_errors)) if model_errors.numel() > 0 else batch_errors
    
    # 计算损失
    loss = criterion(outputs, y_train, model_errors)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    # 每50轮打印损失
    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

# 评估模式
model.eval()
with torch.no_grad():
    train_predict = model(X_train)
    test_predict = model(X_test)

# 反归一化差分值
def inverse_transform(data):
    return scaler.inverse_transform(data.numpy().reshape(-1, 1)).flatten()

train_diff = inverse_transform(train_predict)
test_diff = inverse_transform(test_predict)

# 逆差分重建原始值
# 修改逆差分重建函数
def reconstruct_original(diff_values, original_data, start_idx, d=1, D=1, s=12):
    """
    从差分值重建原始数据
    
    参数:
    diff_values: 季节性差分后的预测值
    original_data: 原始数据
    start_idx: 在原始数据中的起始索引
    d: 一阶差分次数
    D: 季节性差分次数
    s: 季节性周期
    """
    reconstructed = []
    
    # 获取重建所需的历史原始数据
    raw_history = list(original_data[start_idx-s-1:start_idx])
    
    # 计算一阶差分历史值
    diff1_history = [raw_history[i] - raw_history[i-1] for i in range(1, len(raw_history))]
    
    # 初始化历史记录
    history_original = list(raw_history)  # 原始数据历史
    history_diff1 = list(diff1_history)   # 一阶差分历史
    
    for i, val in enumerate(diff_values):
        # 1. 逆季节性差分：使用一阶差分历史值
        # val是季节性差分值，加上s步前的一阶差分值
        diff1_current = val + history_diff1[-s]
        
        # 2. 逆一阶差分：使用原始数据历史值
        # 当前原始值 = 上一个原始值 + 当前一阶差分值
        current_original = history_original[-1] + diff1_current
        
        reconstructed.append(current_original)
        
        # 更新历史记录
        history_original.pop(0)
        history_original.append(current_original)
        
        history_diff1.pop(0)
        history_diff1.append(diff1_current)
    
    return np.array(reconstructed)

# 重建原始数据
train_start_idx = s + 1 + p  # 差分开始索引 + AR阶数
train_reconstructed = reconstruct_original(train_diff, original, train_start_idx, d, D, s)

test_start_idx = train_start_idx + len(train_reconstructed)
test_reconstructed = reconstruct_original(test_diff, original, test_start_idx, d, D, s)

# 获取实际值用于比较
y_train_actual = original[train_start_idx:train_start_idx+len(train_reconstructed)]
y_test_actual = original[test_start_idx:test_start_idx+len(test_reconstructed)]

# 可视化结果
plt.figure(figsize=(15, 8))

# 原始数据
plt.plot(original, label='Actual Data', alpha=0.7)

# 训练集预测
train_indices = range(train_start_idx, train_start_idx + len(train_reconstructed))
plt.plot(train_indices, train_reconstructed, 'g-', label='Train Predictions')

# 测试集预测
test_indices = range(test_start_idx, test_start_idx + len(test_reconstructed))
plt.plot(test_indices, test_reconstructed, 'r-', label='Test Predictions')

plt.axvline(x=train_indices[-1], color='k', linestyle='--', label='Train/Test Split')
plt.title('ARIMA Model Predictions with Seasonal Features')
plt.xlabel('Time Steps')
plt.ylabel('Passengers')
plt.legend()
plt.grid(True)
plt.show()

# 计算RMSE
def rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted)**2))

train_rmse = rmse(y_train_actual, train_reconstructed)
test_rmse = rmse(y_test_actual, test_reconstructed)

print(f'Train RMSE: {train_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')
print(f'Test RMSE as % of mean: {test_rmse/np.mean(original)*100:.1f}%')

# 预测未来12个月
def forecast_future(model, last_values, months_ahead, month_features, p, q):
    """
    预测未来值
    
    参数:
    model: 训练好的模型
    last_values: 最后的p个差分值
    months_ahead: 要预测的未来月数
    month_features: 未来月份的one-hot编码
    p: AR阶数
    q: MA阶数
    """
    forecast = []
    history = list(last_values)
    errors = [0] * q  # 初始误差设为0
    
    for i in range(months_ahead):
        # 准备特征
        ar_features = history[-p:]
        ma_features = errors[-q:]
        seasonal_feature = month_features[i]
        
        features = np.concatenate((ar_features, ma_features, seasonal_feature))
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        
        # 预测
        with torch.no_grad():
            pred = model(features_tensor).item()
        
        forecast.append(pred)
        
        # 更新历史记录
        history.append(pred)
        
        # 更新误差 (使用预测值本身作为估计)
        errors.append(pred - np.mean(ar_features))
    
    return forecast

# 修改未来预测的重建部分
last_original_values = original[-s-1:]  # 最后s+1个原始值
# 计算最后的一阶差分值
last_diff1_values = [last_original_values[i] - last_original_values[i-1] 
                    for i in range(1, len(last_original_values))]
future_months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])  # 未来12个月
future_month_features = create_month_features(future_months)

# 预测未来差分值
future_diff = forecast_future(model, last_diff1_values, 12, future_month_features, p, q)

# 重建原始预测值
last_original_values = original[-s-1:]  # 最后s+1个原始值
future_reconstructed = []
history_original = list(last_original_values)  # 原始数据历史
history_diff1 = list(last_diff1_values)        # 一阶差分历史

for val in future_diff:
    # 1. 逆季节性差分
    diff1_current = val + history_diff1[-s]
    
    # 2. 逆一阶差分
    current_original = history_original[-1] + diff1_current
    
    future_reconstructed.append(current_original)
    
    # 更新历史记录
    history_original.pop(0)
    history_original.append(current_original)
    
    history_diff1.pop(0)
    history_diff1.append(diff1_current)

# 可视化未来预测
plt.figure(figsize=(15, 6))
plt.plot(original, 'b-', label='Historical Data')

# 未来预测
future_indices = range(len(original), len(original) + len(future_reconstructed))
plt.plot(future_indices, future_reconstructed, 'r--', label='Forecast')

# 添加最后12个月作为参考
plt.plot(range(len(original)-12, len(original)), original[-12:], 'g-', alpha=0.7)

plt.title('12-Month Forecast')
plt.xlabel('Time Steps')
plt.ylabel('Passengers')
plt.legend()
plt.grid(True)
plt.show()

print("Future Forecast:")
for i, (month, value) in enumerate(zip(future_months, future_reconstructed)):
    print(f"1961-{month:02d}: {int(round(value))} passengers")