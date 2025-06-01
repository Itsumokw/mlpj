from .tensor import Tensor
import numpy as np

class FeatureExtractor:
    """Feature extractor for time series data."""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        
    def transform(self, X: np.ndarray) -> np.ndarray:
        features = []
        for i in range(len(X) - self.window_size):
            window = X[i:i+self.window_size]
            features.append([
                np.mean(window),  # Mean
                np.std(window),   # Standard deviation
                np.max(window) - np.min(window),  # Range
                np.abs(np.fft.fft(window)[1]),  # Frequency domain feature
                np.polyfit(np.arange(self.window_size), window, 1)[0]  # Trend feature
            ])
        return np.array(features)

    @staticmethod
    def sliding_window(X: np.ndarray, window: int) -> np.ndarray:
        """Generate a sliding window view of the input data."""
        shape = X.shape[:-1] + (X.shape[-1] - window + 1, window)
        strides = X.strides + (X.strides[-1],)
        return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)