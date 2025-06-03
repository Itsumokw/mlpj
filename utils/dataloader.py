import numpy as np

class DataPipeline:
    """轻量数据预处理管道"""
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.scaler = None
        
    def normalize(self, X: np.ndarray) -> np.ndarray:
        """在线归一化"""
        if self.scaler is None:
            self.min = X.min()
            self.max = X.max()
        return (X - self.min) / (self.max - self.min + 1e-8)

    def process(self, raw_data: np.ndarray) -> tuple:
        """返回 (features, targets)"""
        scaled = self.normalize(raw_data)
        features = FeatureExtractor(self.window_size).transform(scaled)
        targets = scaled[self.window_size:]
        return features, targets