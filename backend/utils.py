import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple, Dict, Any
import os
import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_default_dataset() -> pd.DataFrame:
    """加载内置的AirPassengers数据集"""
    data_path = '../datasets/rakannimer/air-passengers/versions/1/AirPassengers.csv'
    df = pd.read_csv(data_path)
        # 确保列名正确
    
    df.columns = ['Month', '#Passengers']
    logger.info(f"Loaded default dataset with {len(df)} records")
    return df

def parse_date(date_str: str) -> int:
    """解析日期字符串返回月份(1-12)"""
    try:
        # 标准化日期字符串
        date_str = str(date_str).strip()
        
        # 尝试匹配 "1949-01" 格式
        if re.match(r"\d{4}-\d{1,2}", date_str):
            parts = date_str.split('-')
            month = int(parts[1])
            return month
        
        # 尝试匹配 "01/1949" 格式
        if re.match(r"\d{1,2}/\d{4}", date_str):
            parts = date_str.split('/')
            month = int(parts[0])
            return month
        
        # 尝试匹配月份缩写
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        for abbr, month_num in month_map.items():
            if abbr in date_str.lower():
                return month_num
    except Exception as e:
        logger.error(f"Error parsing date '{date_str}': {str(e)}")
    
    # 默认返回1月
    return 1

def extract_month_features(dates: List[str]) -> np.ndarray:
    """从日期字符串中提取月份特征（one-hot编码）"""
    month_features = np.zeros((len(dates), 12))
    
    for i, date_str in enumerate(dates):
        month = parse_date(date_str)
        month = max(1, min(12, month))  # 确保在1-12范围内
        month_features[i, month-1] = 1
    
    logger.info(f"Created month features for {len(dates)} dates")
    return month_features

def preprocess_data(
    dataset_name: str, 
    time_col: str, 
    value_col: str, 
    custom_data: List[Dict] = None,
    p: int = 12,
    q: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, List[float], List[str], np.ndarray]:
    """
    数据预处理函数
    返回: (X_train, y_train, X_test, y_test, scaler, last_values, dates, full_normalized)
    """
    try:
        logger.info(f"Preprocessing data for {dataset_name}")
        
        if dataset_name == "Air Passengers (Default)":
            df = load_default_dataset()
            dates = df['Month'].astype(str).tolist()
            values = df['#Passengers'].values.astype(float)
        else:
            df = pd.DataFrame(custom_data)
            dates = df[time_col].astype(str).tolist()
            values = df[value_col].values.astype(float)
        
        logger.info(f"Loaded dataset with {len(values)} records")
        
        # 数据归一化
        scaler = MinMaxScaler(feature_range=(0, 1))
        values_normalized = scaler.fit_transform(values.reshape(-1, 1)).flatten()
        full_normalized = values_normalized.copy()  # 保存完整归一化数据
        
        # 创建标准数据集 (滑动窗口)
        X, y = [], []
        window_size = p
        
        # 确保有足够的数据创建特征
        if len(values_normalized) > window_size:
            for i in range(len(values_normalized) - window_size):
                X.append(values_normalized[i:i+window_size])
                y.append(values_normalized[i+window_size])
        
        if len(X) == 0:
            logger.warning("Not enough data to create features. Using fallback method.")
            # 回退方法：使用所有数据作为特征
            if len(values_normalized) > 0:
                X = [values_normalized]
                y = [values_normalized[-1]] if len(values_normalized) > 1 else [0]
        
        X = np.array(X)
        y = np.array(y)
        
        # 划分训练集和测试集 (80/20)
        train_size = int(len(X) * 0.8) if len(X) > 0 else 0
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        # 获取最后p个值用于预测
        last_values = values_normalized[-p:].tolist() if len(values_normalized) >= p else []
        
        logger.info(f"Created dataset - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        return X_train, y_train, X_test, y_test, scaler, last_values, dates, full_normalized
    
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        # 返回空数据集但保留维度信息
        return (
            np.empty((0, p)), 
            np.empty(0), 
            np.empty((0, p)), 
            np.empty(0), 
            MinMaxScaler(), 
            [], 
            [],
            np.empty(0)
        )

def create_arima_features(data: np.ndarray, month_features: np.ndarray, p: int, q: int) -> Tuple[np.ndarray, np.ndarray]:
    """为ARIMA模型创建特征"""
    logger.info(f"Creating ARIMA features with p={p}, q={q}")
    
    if len(data) == 0 or len(month_features) == 0:
        logger.error("Cannot create ARIMA features from empty data")
        return np.array([]), np.array([])
    
    # 确保数据长度匹配
    min_length = min(len(data), len(month_features))
    if min_length < p + 1:
        logger.error(f"Insufficient data for ARIMA features. Need at least {p+1} points, got {min_length}")
        return np.array([]), np.array([])
    
    data = data[:min_length]
    month_features = month_features[:min_length]
    
    X, y = [], []
    errors = [0.0] * min_length  # 初始化误差列表
    
    try:
        for i in range(p, min_length):
            # AR特征: 过去p个值
            ar_features = data[i-p:i]
            
            # MA特征: 过去q个预测误差
            if i >= q:
                ma_features = errors[i-q:i]
            else:
                ma_features = [0.0] * q
            
            # 季节性特征: 当前月份
            seasonal_feature = month_features[i]
            
            # 合并所有特征
            features = np.concatenate([
                ar_features,
                ma_features,
                seasonal_feature
            ])
            
            X.append(features)
            y.append(data[i])
            
            # 估计预测误差 (实际值 - 特征均值)
            if i >= 10:  # 有足够历史数据后开始更新误差
                predicted = np.mean(features[:p])  # 简单预测
                errors[i] = data[i] - predicted
    except Exception as e:
        logger.error(f"Error creating ARIMA features: {str(e)}")
        return np.array([]), np.array([])
    
    if len(X) == 0:
        logger.error("No ARIMA features created")
        return np.array([]), np.array([])
    
    logger.info(f"Created {len(X)} ARIMA samples with {len(X[0])} features each")
    return np.array(X), np.array(y)

# preprocess_data(
#     dataset_name="Air Passengers (Default)",
#     time_col="Month",
#     value_col="#Passengers",
#     p=12,
#     q=1
# )