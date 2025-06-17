from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class PreprocessRequest(BaseModel):
    dataset_name: str
    time_col: str
    value_col: str
    custom_data: Optional[List[Dict]] = None
    p: int = 12
    q: int = 1

class PreprocessResponse(BaseModel):
    X_train: List[List[float]]
    y_train: List[float]
    X_test: List[List[float]]
    y_test: List[float]
    scaler: Dict[str, Any]
    last_values: List[float]
    dates: List[str]

class TrainConfig(BaseModel):
    model_type: str
    p: int = 12
    q: int = 1
    hidden_size: int = 64
    kernel_size: int = 3
    num_layers: int = 3
    dropout: float = 0.1
    output_steps: int = 12
    lr: float = 0.005
    epochs: int = 500
    alpha: float = 0.1

class TrainRequest(BaseModel):
    config: TrainConfig
    train_data: Dict[str, Any]

class TrainResponse(BaseModel):
    job_id: str
    status: str

class TrainingStatus(BaseModel):
    job_id: str
    status: str  # queued, training, completed, failed
    progress: Optional[int] = None  # 百分比
    results: Optional[Dict] = None  # 训练结果

class ForecastRequest(BaseModel):
    model_state: Dict[str, Any]
    forecast_months: int
    last_values: List[float]
    output_steps: int
    dates: List[str]
    scaler: Dict[str, Any]

class ForecastResponse(BaseModel):
    forecast_values: List[float]
    forecast_dates: List[str]
    history_values: List[float]
    history_dates: List[str]
    forecast_stats: Dict[str, float]  # mean, min, max