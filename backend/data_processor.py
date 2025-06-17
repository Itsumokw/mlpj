from schemas import PreprocessRequest, PreprocessResponse
from backend.utils import preprocess_data, create_arima_features, extract_month_features
import numpy as np
from typing import Dict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_data(request: PreprocessRequest) -> PreprocessResponse:
    """处理数据预处理请求"""
    try:
        
        
        # 调用预处理函数
        X_train, y_train, X_test, y_test, scaler, last_values, dates, full_normalized = preprocess_data(
            request.dataset_name,
            request.time_col,
            request.value_col,
            request.custom_data,
            request.p,
            request.q
        )
        
        # 如果是ARIMA模型，需要特殊处理特征
        if X_train is not None:
            logger.info("Creating ARIMA features")
            
            # 提取月份特征
            month_features = extract_month_features(dates)
            logger.info(f"Month features shape: {month_features.shape}")
            
            # 为完整数据集创建ARIMA特征
            X_full, y_full = create_arima_features(full_normalized, month_features, request.p, request.q)
            
            if len(X_full) > 0:
                # 重新划分训练集和测试集
                train_size = int(len(X_full) * 0.8)
                X_train = X_full[:train_size]
                y_train = y_full[:train_size]
                X_test = X_full[train_size:]
                y_test = y_full[train_size:]
                
                # 记录特征维度
                logger.info(f"ARIMA features created - Train: {X_train.shape}, Test: {X_test.shape}")
            else:
                logger.error("Failed to create ARIMA features")
                # 不继续使用ARIMA特征，但保持原始特征
                logger.info("Using standard features instead of ARIMA features")
        
        # 确保所有数组都是numpy数组
        X_train = np.array(X_train) if not isinstance(X_train, np.ndarray) else X_train
        X_test = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test
        y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
        y_test = np.array(y_test) if not isinstance(y_test, np.ndarray) else y_test
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        # 准备响应数据
        response = PreprocessResponse(
            X_train=X_train.tolist(),
            y_train=y_train.tolist(),
            X_test=X_test.tolist(),
            y_test=y_test.tolist(),
            scaler={
                "type": "MinMaxScaler",
                "min": scaler.min_.tolist(),
                "scale": scaler.scale_.tolist(),
                "data_min": scaler.data_min_.tolist(),
                "data_max": scaler.data_max_.tolist(),
                "feature_range": scaler.feature_range
            },
            last_values=last_values,
            dates=dates
        )
        print(response.json())
        logger.info("Data processing completed successfully")
        return response
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        # 返回一个带有错误信息的响应
        return PreprocessResponse(
            X_train=[],
            y_train=[],
            X_test=[],
            y_test=[],
            scaler={
                "type": "MinMaxScaler",
                "min": [0],
                "max": [1],
                "scale": [1],
                "data_min": [0],
                "data_max": [1],
                "feature_range": [0, 1]
            },
            last_values=[],
            dates=[],
            error=str(e)
        )
    
process_data(PreprocessRequest(
    dataset_name="Air Passengers (Default)",
    time_col="Month",
    value_col="#Passengers",
    custom_data=None,
    p=12,
    q=1
))