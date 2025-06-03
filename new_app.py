import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time

# 设置页面布局
st.set_page_config(layout="wide", page_title="ARIMA Time Series Forecasting")
st.title("✈️ Air Passengers Forecasting with Neural ARIMA")
st.markdown("""
This application demonstrates a neural network-based ARIMA model for forecasting monthly airline passengers.
The model incorporates both autoregressive (AR) and moving average (MA) components, along with seasonal features.
""")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ Model Configuration")

    # 模型参数
    p = st.slider("AR Order (p)", 1, 24, 12, 1,
                  help="Number of lag observations included in the model (autoregressive part)")
    q = st.slider("MA Order (q)", 1, 12, 1, 1,
                  help="Size of the moving average window")
    hidden_size = st.slider("Hidden Layer Size", 16, 128, 64, 16)
    lr = st.slider("Learning Rate", 0.001, 0.1, 0.005, 0.001)
    epochs = st.slider("Training Epochs", 100, 2000, 500, 100)
    alpha = st.slider("MA Loss Weight (α)", 0.01, 0.5, 0.1, 0.01,
                      help="Weight for moving average component in loss function")

    # 数据选项
    show_data = st.checkbox("Show Raw Data", True)
    show_diff = st.checkbox("Show Differenced Data", False)

    st.subheader("Forecasting Options")
    forecast_months = st.slider("Months to Forecast", 1, 24, 12, 1)

    train_button = st.button("🚀 Train Model")


# 加载数据
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('datasets/rakannimer/air-passengers/versions/1/AirPassengers.csv')
        passengers = data['#Passengers'].values.astype(float)
        months = [int(m.split('-')[1]) for m in data['Month']]
        return data, passengers, months
    except:
        st.error("Failed to load data. Using sample data instead.")
        # Generate sample data if file not found
        t = np.arange(0, 100)
        passengers = 100 + 50 * np.sin(0.1 * t) + 30 * np.random.randn(100)
        months = [i % 12 + 1 for i in range(100)]
        data = pd.DataFrame({'Month': [f"1949-{m:02d}" for m in months], '#Passengers': passengers})
        return data, passengers, months


data, passengers, months = load_data()

# 显示原始数据
if show_data:
    st.subheader("📊 Air Passengers Data")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(data.head(10))
    with col2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['Month'], data['#Passengers'], 'b-')
        ax.set_title("Monthly Air Passengers (1949-1960)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Passengers")
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)


# 数据预处理函数
def preprocess_data(passengers, months, p, q, s=12):
    """Preprocess data for ARIMA model"""

    # 差分操作函数
    def difference(data, interval=1):
        return [data[i] - data[i - interval] for i in range(interval, len(data))]

    # 原始数据
    original = passengers.copy()

    # 一阶差分
    diff1 = difference(passengers, 1)
    # 季节性差分
    diff_seasonal = difference(diff1[s:], s)

    # 更新月份信息以匹配差分后的数据
    months_diff = months[s + 1:]

    # 创建月份特征矩阵 (one-hot编码)
    def create_month_features(months):
        month_features = np.zeros((len(months), 12))
        for i, month in enumerate(months):
            month_features[i, month - 1] = 1
        return month_features

    month_features = create_month_features(months_diff)

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    diff_seasonal_normalized = scaler.fit_transform(np.array(diff_seasonal).reshape(-1, 1)).flatten()

    # 创建数据集函数
    def create_dataset(data, month_features, p, q):
        X, y = [], []
        errors = []  # 存储预测误差
        for i in range(p, len(data)):
            # AR特征: 过去p个值
            ar_features = data[i - p:i]

            # MA特征: 过去q个预测误差
            ma_features = errors[-q:] if len(errors) >= q else [0] * q

            # 季节性特征: 当前月份
            seasonal_feature = month_features[i]

            # 合并所有特征
            features = np.concatenate((ar_features, ma_features, seasonal_feature))
            X.append(features)
            y.append(data[i])

            # 计算当前预测误差
            if len(errors) < 10:  # 初始阶段
                errors.append(0)
            else:
                # 实际值减去特征的平均值作为误差估计
                errors.append(data[i] - np.mean(features[:p]))

        return np.array(X), np.array(y)

    X, y = create_dataset(diff_seasonal_normalized, month_features, p, q)

    # 转换为PyTorch张量
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

    # 数据集划分
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return {
        'original': original,
        'months': months,
        'months_diff': months_diff,
        'scaler': scaler,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_size': train_size,
        'p': p,
        'q': q
    }


# 定义ARIMA模型
class ARIMAModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 自定义损失函数
class MALoss(torch.nn.Module):
    def __init__(self, q=1, alpha=0.1):
        super().__init__()
        self.q = q
        self.alpha = alpha
        self.mse = torch.nn.MSELoss()

    def forward(self, predictions, targets, model_errors):
        mse_loss = self.mse(predictions, targets)
        ma_loss = torch.tensor(0.0, device=predictions.device)

        if len(model_errors) >= self.q:
            recent_errors = model_errors[-self.q:]
            ma_loss = torch.mean(recent_errors ** 2)

        total_loss = mse_loss + self.alpha * ma_loss
        return total_loss


# 训练模型
def train_model(model, X_train, y_train, criterion, optimizer, epochs):
    losses = []
    device = X_train.device
    model_errors = torch.tensor([], device=device)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_train)

        # 计算当前批次的误差
        batch_errors = (outputs - y_train).detach().squeeze()

        # 更新全局误差记录
        model_errors = torch.cat((model_errors, batch_errors)) if model_errors.numel() > 0 else batch_errors

        # 计算损失
        loss = criterion(outputs, y_train, model_errors)

        # 反向传播
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        # 更新进度
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    return model, losses


# 评估模型
def evaluate_model(model, X_test, y_test, scaler, preprocessed_data):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)

    # 反归一化差分值
    def inverse_transform(data):
        return scaler.inverse_transform(data.numpy().reshape(-1, 1)).flatten()

    test_diff = inverse_transform(outputs)

    # 逆差分重建原始值
    def reconstruct_original(diff_values, original_data, start_idx, d=1, D=1, s=12):
        reconstructed = []
        raw_history = list(original_data[start_idx - s - 1:start_idx])
        diff1_history = [raw_history[i] - raw_history[i - 1] for i in range(1, len(raw_history))]

        history_original = list(raw_history)
        history_diff1 = list(diff1_history)

        for i, val in enumerate(diff_values):
            diff1_current = val + history_diff1[-s]
            current_original = history_original[-1] + diff1_current
            reconstructed.append(current_original)

            history_original.pop(0)
            history_original.append(current_original)

            history_diff1.pop(0)
            history_diff1.append(diff1_current)

        return np.array(reconstructed)

    # 重建原始数据
    s = 12
    d = 1
    D = 1
    train_start_idx = s + 1 + preprocessed_data['p']
    test_start_idx = train_start_idx + preprocessed_data['train_size']

    test_reconstructed = reconstruct_original(test_diff, preprocessed_data['original'],
                                              test_start_idx, d, D, s)

    # 获取实际值
    y_test_actual = preprocessed_data['original'][test_start_idx:test_start_idx + len(test_reconstructed)]

    # 计算RMSE
    def rmse(actual, predicted):
        return np.sqrt(np.mean((actual - predicted) ** 2))

    test_rmse = rmse(y_test_actual, test_reconstructed)

    return {
        'predictions': test_reconstructed,
        'actuals': y_test_actual,
        'rmse': test_rmse,
        'test_start_idx': test_start_idx
    }


# 预测未来
def forecast_future(model, last_values, months_ahead, month_features, p, q, scaler, original_data):
    forecast = []
    history = list(last_values)
    errors = [0] * q

    for i in range(months_ahead):
        ar_features = history[-p:]
        ma_features = errors[-q:]
        seasonal_feature = month_features[i]

        features = np.concatenate((ar_features, ma_features, seasonal_feature))
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            pred = model(features_tensor).item()

        forecast.append(pred)
        history.append(pred)
        errors.append(pred - np.mean(ar_features))

    # 反归一化
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()

    # 重建原始值
    s = 12
    last_original_values = original_data[-s - 1:]
    last_diff1_values = [last_original_values[i] - last_original_values[i - 1]
                         for i in range(1, len(last_original_values))]

    future_reconstructed = []
    history_original = list(last_original_values)
    history_diff1 = list(last_diff1_values)

    for val in forecast:
        diff1_current = val + history_diff1[-s]
        current_original = history_original[-1] + diff1_current
        future_reconstructed.append(current_original)

        history_original.pop(0)
        history_original.append(current_original)

        history_diff1.pop(0)
        history_diff1.append(diff1_current)

    return future_reconstructed


# 主应用逻辑
if train_button:
    # 预处理数据
    with st.spinner("Preprocessing data..."):
        preprocessed_data = preprocess_data(passengers, months, p, q)

    # 显示差分数据
    if show_diff:
        st.subheader("Differenced Data")
        fig, ax = plt.subplots(figsize=(10, 4))
        diff_data = preprocessed_data['scaler'].transform(
            np.array(preprocessed_data['original'][13:]).reshape(-1, 1)
        ).flatten()
        ax.plot(preprocessed_data['months_diff'], diff_data)
        ax.set_title("Differenced Air Passengers Data")
        ax.set_xlabel("Month")
        ax.set_ylabel("Differenced Passengers")
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # 初始化模型
    input_size = p + q + 12
    model = ARIMAModel(input_size, hidden_size, 1)
    criterion = MALoss(q=q, alpha=alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    st.subheader("Model Training")
    model, losses = train_model(
        model,
        preprocessed_data['X_train'],
        preprocessed_data['y_train'],
        criterion,
        optimizer,
        epochs
    )

    # 显示训练损失
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    st.pyplot(fig)

    # 评估模型
    with st.spinner("Evaluating model..."):
        results = evaluate_model(
            model,
            preprocessed_data['X_test'],
            preprocessed_data['y_test'],
            preprocessed_data['scaler'],
            preprocessed_data
        )

    # 显示评估结果
    st.subheader("Model Evaluation")
    st.metric("Test RMSE", f"{results['rmse']:.2f}")
    st.metric("Test RMSE as % of mean",
              f"{results['rmse'] / np.mean(preprocessed_data['original']) * 100:.1f}%")

    # 显示预测结果
    fig, ax = plt.subplots(figsize=(12, 6))

    # 原始数据
    ax.plot(preprocessed_data['original'], 'b-', label='Historical Data', alpha=0.7)

    # 测试集预测
    test_indices = range(results['test_start_idx'],
                         results['test_start_idx'] + len(results['predictions']))
    ax.plot(test_indices, results['predictions'], 'r-', label='Test Predictions', alpha=0.8)

    # 添加最后12个月作为参考
    last_year_indices = range(len(preprocessed_data['original']) - 12,
                              len(preprocessed_data['original']))
    ax.plot(last_year_indices, preprocessed_data['original'][-12:], 'g-', alpha=0.7)

    ax.axvline(x=results['test_start_idx'], color='k', linestyle='--', label='Train/Test Split')
    ax.set_title("ARIMA Model Predictions with Seasonal Features")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Passengers")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # 预测未来
    st.subheader(f"{forecast_months}-Month Forecast")

    # 创建月份特征
    last_month = int(data['Month'].iloc[-1].split('-')[1])
    future_months = [(last_month + i) % 12 or 12 for i in range(1, forecast_months + 1)]


    def create_month_features(months):
        month_features = np.zeros((len(months), 12))
        for i, month in enumerate(months):
            month_features[i, month - 1] = 1
        return month_features


    future_month_features = create_month_features(future_months)

    # 获取最后的值
    last_diff1_values = [
                            passengers[-1] - passengers[-2],
                            passengers[-2] - passengers[-3]
                        ][-p:]

    # 预测
    with st.spinner("Generating forecast..."):
        future_reconstructed = forecast_future(
            model,
            last_diff1_values,
            forecast_months,
            future_month_features,
            p, q,
            preprocessed_data['scaler'],
            passengers
        )

    # 显示预测结果
    fig, ax = plt.subplots(figsize=(12, 6))

    # 原始数据
    ax.plot(preprocessed_data['original'], 'b-', label='Historical Data')

    # 未来预测
    future_indices = range(len(preprocessed_data['original']),
                           len(preprocessed_data['original']) + len(future_reconstructed))
    ax.plot(future_indices, future_reconstructed, 'r--', label='Forecast')

    # 添加最后12个月作为参考
    ax.plot(last_year_indices, preprocessed_data['original'][-12:], 'g-', alpha=0.7)

    ax.set_title(f"{forecast_months}-Month Forecast")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Passengers")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # 显示预测表格
    forecast_df = pd.DataFrame({
        'Month': [f"1961-{m:02d}" for m in future_months],
        'Forecasted Passengers': [int(round(x)) for x in future_reconstructed]
    })
    st.dataframe(forecast_df.style.format({"Forecasted Passengers": "{:,.0f}"}))

# 添加说明
st.sidebar.markdown("""
**Model Notes:**
- This neural ARIMA model combines traditional time series concepts with deep learning
- AR (Autoregressive) component uses past observations
- MA (Moving Average) component models forecast errors
- Seasonal features are incorporated via month one-hot encoding
""")

# 添加作者信息
st.sidebar.markdown("---")
st.sidebar.info("""
**Developed by:** [Your Name]  
**Dataset:** Monthly Air Passengers (1949-1960)  
**Model:** Neural ARIMA with Seasonal Features
""")