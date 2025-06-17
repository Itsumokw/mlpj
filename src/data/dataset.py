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
    """Dataset class for time series data with features."""
    
    def __init__(self, features, targets):
        """Initialize TimeSeriesDataset.
        
        Args:
            features: Array of shape (n_samples, seq_length, n_features)
            targets: Array of shape (n_samples, prediction_steps, 1)
        """
        # Convert arrays to tensors if needed
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32)
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets, dtype=torch.float32)
            
        self.features = features
        self.targets = targets
        
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.features)
        
    def __getitem__(self, idx):
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            tuple: (features, target) where:
                features is a tensor of shape (seq_length, n_features)
                target is a tensor of shape (prediction_steps, 1)
        """
        return self.features[idx], self.targets[idx]