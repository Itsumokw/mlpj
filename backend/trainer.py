import time
import json
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
from models.arima import NeuralARIMA
from models.tcn import TCN
from models.gru import TimeGRU
from models.linear import TimeLinear
from typing import Dict, Any, Tuple
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.jobs = {}
    
    def create_model(self, config: Dict) -> Any:
        """根据配置创建模型"""
        model_type = config['model_type']
        
        if model_type == "Neural ARIMA":
            p = config.get('p', 12)
            q = config.get('q', 1)
            input_size = p + q + 12  # AR特征 + MA特征 + 月份特征
            return NeuralARIMA(
                input_size=input_size,
                hidden_size=config['hidden_size'],
                output_size=1,
                lr=config['lr'],
                epochs=config['epochs'],
                alpha=config['alpha'],
                q=q,
                p=p
            )
        
        elif "TCN" in model_type:
            return TCN(
                input_size=1,
                output_size=1,
                num_channels=[config['hidden_size']] * config['num_layers'],
                kernel_size=config['kernel_size'],
                dropout=config['dropout'],
                lr=config['lr'],
                epochs=config['epochs'],
                batch_size=config.get('batch_size', 16)
            )
        
        elif "GRU" in model_type:
            return TimeGRU(
                input_size=1,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=1,
                dropout=config['dropout'],
                lr=config['lr'],
                epochs=config['epochs'],
                batch_size=config.get('batch_size', 16)
            )
        
        
        else:  # Time Linear
            p = config.get('p', 12)
            q = config.get('q', 1)
            input_size = p + q + 12
            return TimeLinear(
                input_size=input_size,  # 使用实际特征数量
                output_size=1,
                lr=config['lr'],
                epochs=config['epochs']
            )
    
    def train_model(self, job_id: str, config: Dict, train_data: Dict) -> Dict:
        """训练模型并返回结果"""
        # 初始化训练状态
        self.jobs[job_id] = {
            "status": "training",
            "progress": 0,
            "results": None
        }
        
        try:
            # 创建模型
            model = self.create_model(config)
            
            # 准备数据
            X_train = np.array(train_data['X_train'])
            y_train = np.array(train_data['y_train'])
            X_test = np.array(train_data['X_test'])
            y_test = np.array(train_data['y_test'])
            
            # 检查数据是否为空
            if len(X_train) == 0 or len(y_train) == 0:
                logger.error("Empty training data received")
                raise ValueError("Training data is empty")
            
            # 训练模型
            start_time = time.time()
            loss_history = model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # 评估模型
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            logger.info(f"scaler: {train_data['scaler']}")
            
            # 计算RMSE - 添加空数据检查
            if len(y_train) > 0:
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            else:
                train_rmse = 0.0
                logger.warning("Skipping train RMSE calculation due to empty data")
            
            if len(y_test) > 0:
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            else:
                test_rmse = 0.0
                logger.warning("Skipping test RMSE calculation due to empty data")

            # 反归一化预测值和实际值
            scaler = self._create_scaler(train_data['scaler'])
            def inverse_transform(values):
                values_2d = np.array(values).reshape(-1, 1)
                return scaler.inverse_transform(values_2d).flatten().tolist()
            
            train_pred_orig = inverse_transform(train_pred)
            test_pred_orig = inverse_transform(test_pred)
            actuals_orig = inverse_transform(y_train.tolist() + y_test.tolist())
            logger.info(f"train_pred: {train_pred[:5]}, test_pred: {test_pred[:5]}, train_pred_orig: {train_pred_orig[:5]}, test_pred_orig: {test_pred_orig[:5]}")
            
            # 准备模型状态
            model_state = {
                "type": config['model_type'],
                "state_dict": self._convert_state_dict(model.state_dict()),
                "params": model.get_params(),
                "last_values": train_data['last_values'],
                "dates": train_data['dates'],
                "scaler": train_data['scaler']
            }
            
            # 保存结果
            results = {
                "model_state": model_state,
                "loss_history": loss_history,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "training_time": training_time,
                "predictions": train_pred_orig + test_pred_orig,  # 预测值
                "final_loss": loss_history[-1] if loss_history else 0.0,  # 取最后一个损失值
                "test_rmse_percentage": (test_rmse / np.mean(y_test)) * 100 if len(y_test) > 0 else 0.0,
                "num_params": self.count_model_parameters(model),  # 需要实现此方法
                "actuals": actuals_orig,  # 实际值
                "train_size": len(y_train),  # 训练集大小
                "model_type": config['model_type']  # 模型类型
            }
            
            # 更新任务状态
            self.jobs[job_id] = {
                "status": "completed",
                "progress": 100,
                "results": results
            }
            
            return results
        
        except Exception as e:
            self.jobs[job_id] = {
                "status": "failed",
                "progress": 100,
                "message": str(e)
            }
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def _convert_state_dict(self, state_dict):
        """将模型状态字典转换为可序列化格式"""
        serializable_state = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                # 保持张量结构信息
                serializable_state[key] = tensor.cpu().numpy().tolist()
            else:
                serializable_state[key] = tensor
        return serializable_state
    
    def count_model_parameters(self, model):
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters())
    
    def _create_scaler(self, scaler_info):
        """创建并配置归一化器"""
        scaler = MinMaxScaler()
        scaler.scale_ = np.array(scaler_info["scale"])
        scaler.min_ = np.array(scaler_info["min"])
        scaler.data_min_ = np.array(scaler_info["data_min"])
        scaler.data_max_ = np.array(scaler_info["data_max"])
        scaler.feature_range = tuple(scaler_info["feature_range"])
        return scaler
    
    def get_job_status(self, job_id: str) -> Dict:
        """获取任务状态"""
        return self.jobs.get(job_id, {"status": "not_found"})