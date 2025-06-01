import numpy as np
import torch

class Dataset:
    """Dataset class for loading and preprocessing time series data."""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Retrieve a data sample and its corresponding label."""
        return self.data[index], self.labels[index]

    def get_samples(self, indices: list):
        """Retrieve multiple samples based on provided indices."""
        return self.data[indices], self.labels[indices]
    
class TimeSeriesDataset:
    """Dataset class for time series data with multi-step prediction support."""
    
    def __init__(self, data, window_size, target_size=1, stride=1):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size
        self.target_size = target_size
        self.stride = stride
        
    def __len__(self):
        return (len(self.data) - self.window_size - self.target_size) // self.stride + 1
        
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        target_idx = end_idx + self.target_size
        
        x = self.data[start_idx:end_idx]
        y = self.data[end_idx:target_idx]
        
        return x, y