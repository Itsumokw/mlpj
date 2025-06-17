import numpy as np
import torch
from typing import Optional, Tuple, Union

class BaseTransform:
    """Base class for all transforms."""
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'BaseTransform':
        return self

    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        raise NotImplementedError

    def fit_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        return self.fit(data).transform(data)

class StandardScaler(BaseTransform):
    """Standardize features by removing mean and scaling to unit variance."""
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'StandardScaler':
        # Convert to torch tensor and ensure float type
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = data.float()  # Convert to float
        
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True) + 1e-8
        return self
        
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler not fitted")
            
        # Convert to torch tensor and ensure float type
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = data.float()  # Convert to float
        
        return (data - self.mean) / self.std
        
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = data.float()  # Convert to float
        return data * self.std + self.mean

class MinMaxScaler(BaseTransform):
    """Scale features to a given range."""
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.scale = None
        
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'MinMaxScaler':
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data_min = data.min(dim=0, keepdim=True)[0]
        data_max = data.max(dim=0, keepdim=True)[0]
        
        self.min = data_min
        self.scale = (self.feature_range[1] - self.feature_range[0]) / (data_max - data_min + 1e-8)
        return self
        
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return (data - self.min) * self.scale + self.feature_range[0]
        
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        return (data - self.feature_range[0]) / self.scale + self.min

class Augmentation:
    """Class for data augmentation techniques."""
    
    @staticmethod
    def add_noise(data: Union[np.ndarray, torch.Tensor], 
                 noise_level: float = 0.01) -> Union[np.ndarray, torch.Tensor]:
        """Add random noise to the data."""
        if isinstance(data, np.ndarray):
            noise = np.random.normal(0, noise_level, data.shape)
        else:
            noise = torch.randn_like(data) * noise_level
        return data + noise

    @staticmethod
    def time_warp(data: Union[np.ndarray, torch.Tensor], 
                  sigma: float = 0.2) -> Union[np.ndarray, torch.Tensor]:
        """Apply time warping to the data."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        
        length = data.shape[-1]
        grid = torch.linspace(0, length-1, length)
        warp = grid + torch.randn_like(grid) * sigma
        warp = torch.clamp(warp, 0, length-1)
        
        # Interpolate warped data
        warped_data = torch.nn.functional.interpolate(
            data.unsqueeze(0), 
            size=length,
            mode='linear',
            align_corners=True
        ).squeeze(0)
        
        return warped_data.numpy() if isinstance(data, np.ndarray) else warped_data

    @staticmethod
    def magnitude_warp(data: Union[np.ndarray, torch.Tensor], 
                      sigma: float = 0.2) -> Union[np.ndarray, torch.Tensor]:
        """Apply magnitude warping to the data."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            
        length = data.shape[-1]
        smooth_noise = torch.randn(length) * sigma
        # Apply Gaussian smoothing to the noise
        smooth_noise = torch.nn.functional.conv1d(
            smooth_noise.view(1, 1, -1),
            torch.ones(1, 1, 5)/5,
            padding=2
        ).view(-1)
        
        warped_data = data * (1 + smooth_noise)
        return warped_data.numpy() if isinstance(data, np.ndarray) else warped_data