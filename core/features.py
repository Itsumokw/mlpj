import numpy as np

class FeatureExtractor:
    """轻量级时序特征提取器"""
    def __init__(self, window_size=10):
        self.window_size = window_size
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        features = []
        for i in range(len(X) - self.window_size):
            window = X[i:i+self.window_size]
            features.append([
                # 统计特征
                np.mean(window),
                np.std(window),
                np.max(window) - np.min(window),
                # 频域特征
                np.abs(np.fft.fft(window)[1]),
                # 趋势特征
                np.polyfit(np.arange(self.window_size), window, 1)[0]
            ])
        return np.array(features)

    @staticmethod
    def sliding_window(X: np.ndarray, window: int) -> np.ndarray:
        """零拷贝滑动窗口生成"""
        shape = X.shape[:-1] + (X.shape[-1] - window + 1, window)
        strides = X.strides + (X.strides[-1],)
        return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)