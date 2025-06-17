import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from models.arima import NeuralARIMA
from models.tcn import TCN, EnhancedTCN, AdvancedTCN
from models.gru import TimeGRU
from models.linear import TimeLinear
from typing import Any, Dict, List

class ForecastPredictor:
    def __init__(self):
        self.models = {
            "Neural ARIMA": NeuralARIMA,
            "Lite TCN": TCN,
            "Enhanced TCN": EnhancedTCN,  # 映射到新类
            "Advanced TCN": AdvancedTCN,  # 映射到新类
            "Time GRU": TimeGRU,
            "Time CNN": TCN,  # 暂时使用TCN代替
            "Time Linear": TimeLinear
        }
    
    def create_model(self, model_state: Dict) -> Any:
        """从状态字典创建模型"""
        model_type = model_state["type"]
        params = model_state["params"]
        
        # 创建模型实例
        model_class = self.models.get(model_type)
        if not model_class:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = model_class(**params)
        
        # 加载模型状态
        
        model.load_state_dict(model_state["state_dict"])
        
        return model
    
    def forecast(self, request: Dict) -> Dict:
        """执行预测"""
        # 从请求中获取数据
        model_state = request["model_state"]
        serialized_state = model_state["state_dict"]
        state_dict = self._restore_state_dict(serialized_state)
        model_state["state_dict"] = state_dict
        forecast_months = request["forecast_months"]
        last_values = request["last_values"]
        output_steps = request["output_steps"]
        dates = request["dates"]
        scaler_info = request["scaler"]
        
        # 恢复归一化器
        scaler = MinMaxScaler()
        scaler.scale_ = np.array(scaler_info["scale"])
        scaler.min_ = np.array(scaler_info["min"])
        scaler.data_min_ = np.array(scaler_info["data_min"])
        scaler.data_max_ = np.array(scaler_info["data_max"])
        scaler.feature_range = tuple(scaler_info["feature_range"])
        
        # 创建模型
        model = self.create_model(model_state)
        
        # 准备历史数据
        last_values_2d = np.array(last_values).reshape(-1, 1)
        history_values = scaler.inverse_transform(last_values_2d).flatten().tolist()
        history_dates = dates[-len(history_values):]

        last_date = pd.to_datetime(dates[-1])
        forecast_dates = [
            str(last_date + pd.DateOffset(months=i+1)) 
            for i in range(forecast_months)
    ]
        
        # 预测未来值
        if model_state["type"] in ["Lite TCN", "Enhanced TCN", "Advanced TCN", "Time CNN"]:
            forecast_values = self._forecast_tcn(model, last_values, forecast_months, scaler)
            print(f"TCN forecast values: {forecast_values}")
            print(f"Forecast dates: {forecast_months}")
        elif model_state["type"] == "Neural ARIMA":
            forecast_values = self._forecast_arima(model, last_values, forecast_months, scaler, forecast_dates)
        elif model_state["type"] == "Time GRU":
            forecast_values = self._forecast_gru(model, last_values, forecast_months, scaler)
        else:
            forecast_values = self._forecast_recursive(model, last_values, forecast_months, scaler)
        
        # 生成预测日期
        last_date = pd.to_datetime(history_dates[-1])
        forecast_dates = [str(last_date + pd.DateOffset(months=i+1)) for i in range(forecast_months)]
        
        # 计算预测统计信息
        forecast_stats = {
            "mean": np.mean(forecast_values),
            "min": np.min(forecast_values),
            "max": np.max(forecast_values)
        }
        
        return {
            "forecast_values": forecast_values,
            "forecast_dates": forecast_dates,
            "history_values": history_values,
            "history_dates": history_dates,
            "forecast_stats": forecast_stats
        }
    
    def _forecast_recursive(self, model, last_values, steps, scaler):
        """递归预测未来值"""
        predictions = []
        current_seq = last_values.copy()
        
        for _ in range(steps):
            # 预测下一个值
            pred = model.predict(np.array([current_seq]))[0]
            predictions.append(pred)
            
            # 更新序列：移除第一个值，添加新预测值
            current_seq = current_seq[1:] + [pred]
        
        # 反归一化
        predictions = scaler.inverse_transform(np.array(predictions).flatten().tolist())
        return predictions
    

    def _forecast_gru(self, model, last_values, steps, scaler):
        """GRU模型的多步预测方法（递归预测）"""
        predictions = []
        current_seq = last_values.copy()  # 复制初始序列
        
        for _ in range(steps):
            # 准备输入数据：形状为 (1, sequence_length, 1)
            input_seq = np.array(current_seq).reshape(1, -1, 1)
            
            # 预测下一步
            pred = model.predict(input_seq)[0]
            predictions.append(pred)
            
            # 更新序列：移除第一个值，添加新预测值
            current_seq = current_seq[1:] + [pred]
        
        # 反归一化所有预测值
        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)).flatten().tolist()
        return predictions
    
    def _forecast_tcn(self, model, last_values, steps, scaler):
        """TCN模型的多步预测方法（递归预测）"""
        predictions = []
        current_seq = last_values.copy()  # 复制初始序列

        data_min = scaler.data_min_[0]
        data_max = scaler.data_max_[0]
        
        for _ in range(steps):
            # 准备输入数据：形状为 (1, sequence_length, 1)
            input_seq = np.array(current_seq).reshape(1, -1, 1)
            
            # 预测下一步
            pred = model.predict(input_seq)[0]

            pred = np.clip(pred, 0, 1)  # 确保在[0,1]范围内

            predictions.append(pred)
            
            # 更新序列：移除第一个值，添加新预测值
            current_seq = current_seq[1:] + [pred]
        
        # 反归一化所有预测值
        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)).flatten().tolist()
        return predictions
    
    def _forecast_arima(self, model, last_values, steps, scaler, forecast_dates):
        """ARIMA模型的预测方法"""
        predictions = []
        p = model.p  # 从模型中获取p值
        q = model.q if hasattr(model, 'q') else 1
        
        # 确保last_values长度正确
        if len(last_values) < p:
            last_values = [0.0] * (p - len(last_values)) + last_values
        current_ar = last_values[-p:].copy()  # 取最后p个值
        
        # 初始化MA序列（归一化误差）
        current_ma = []
        if hasattr(model, 'recent_errors') and model.recent_errors is not None:
            current_ma = model.recent_errors.cpu().numpy().tolist()
        else:
            current_ma = [0.0] * q
        
        # 确保MA序列长度正确
        if len(current_ma) < q:
            current_ma = [0.0] * (q - len(current_ma)) + current_ma
        elif len(current_ma) > q:
            current_ma = current_ma[-q:]
        
        for i in range(steps):
            # 1. 构建月份特征 (one-hot)
            date_str = forecast_dates[i]
            month = pd.to_datetime(date_str).month
            month_feature = [0] * 12
            month_feature[month-1] = 1  # 月份索引从0开始
            
            # 2. 组合特征: [AR特征] + [MA特征] + [月份特征]
            input_features = current_ar.copy()
            input_features.extend(current_ma)
            input_features.extend(month_feature)
            
            # 3. 转换为2D数组并预测下一个点
            input_array = np.array(input_features).reshape(1, -1)  # 关键修复：确保是2D数组
            pred = model.predict(input_array)[0]
            predictions.append(pred)
            
            # 4. 更新AR序列：移除首位，添加新预测
            current_ar = current_ar[1:] + [pred]
            
            # 5. 更新MA序列：用0填充未来误差
            if len(current_ma) > 0:
                current_ma = current_ma[1:] + [0.0]  # 未知误差用0代替
        
        # 反归一化预测结果
        predictions = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1))
        return predictions.flatten().tolist()

    def _restore_state_dict(self, serialized_state):
        """将序列化状态字典还原为PyTorch张量"""
        state_dict = {}
        for key, value in serialized_state.items():
            if isinstance(value, list):  # 检测张量数据
                # 转换为numpy数组再转为PyTorch张量
                tensor = torch.from_numpy(np.array(value)).float()
                state_dict[key] = tensor
            elif isinstance(value, (int, float)):  # 标量张量
                # 创建0维张量
                state_dict[key] = torch.tensor(value)
            else:
                state_dict[key] = value
        return state_dict