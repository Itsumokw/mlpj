import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List, Dict  # Add this import
from .nn.modules import Module, Linear, Conv1d, Sequential, LayerNorm, ReLU, Dropout, GlobalAveragePool, Attention, ModuleList
from .nn import functional as F
from .tensor import Tensor  # Import Tensor class

class BaseModel(Module):
    """Base class for all models"""
    def __init__(self):
        super().__init__()
        self.is_fitted = False
        self._parameters: Dict[str, nn.Parameter] = {}  # Add type hint

    def register_parameter(self, name: str, param: Optional[nn.Parameter]) -> None:
        """Register a parameter with the model.
        
        Args:
            name: Name of the parameter
            param: Parameter to register
        """
        if param is not None:
            self._parameters[name] = param

    def forward(self, x, features=None):
        """Forward pass of the model with both raw data and extracted features.
        
        Args:
            x: Raw time series data of shape (batch_size, sequence_length, input_size)
            features: Optional extracted features of shape (batch_size, sequence_length, feature_size)
        """
        raise NotImplementedError("Forward method not implemented.")

    def fit(self, data, labels):
        """Train the model on the provided data and labels."""
        raise NotImplementedError("Fit method not implemented.")

    def predict(self, data):
        """Generate predictions for the input data."""
        raise NotImplementedError("Predict method not implemented.")

    def save(self, filepath):
        """Save the model to the specified filepath."""
        torch.save(self.state_dict(), filepath)

    def load(self, filepath):
        """Load the model from the specified filepath."""
        self.load_state_dict(torch.load(filepath))


class SimpleARIMA(BaseModel):
    """Simplified ARIMA model for time series prediction."""
    
    def __init__(self, p=3, feature_size=0, output_steps=10):
        super().__init__()
        self.p = p
        self.feature_size = feature_size
        self.output_steps = output_steps
        
        # Create parameters using our Tensor class
        self.coef_ = Tensor(torch.randn(p), requires_grad=True)
        self.bias = Tensor(torch.zeros(1), requires_grad=True)
        
        if feature_size > 0:
            self.feature_weights = Tensor(torch.randn(feature_size), requires_grad=True)
        else:
            self.feature_weights = None
            
        # Register parameters
        self._parameters = {
            'coef_': self.coef_,
            'bias': self.bias
        }
        if self.feature_weights is not None:
            self._parameters['feature_weights'] = self.feature_weights

    def forward(self, x, features=None):
        """Forward pass of ARIMA model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_dim)
            features: Optional extracted features
            
        Returns:
            Predictions tensor of shape (batch_size, output_steps, 1)
        """
        # Handle input shape
        if x.data.ndim == 3:
            batch_size, seq_length, input_dim = x.data.shape
            x = Tensor(x.data.squeeze(-1))
        else:
            batch_size, seq_length = x.data.shape
            
        # Only predict the requested number of steps
        start_idx = seq_length - self.output_steps
        predictions = []
        
        for i in start_idx, seq_length:
            # Get window of past values
            window = x[:, i-self.p:i]
            pred = (window @ self.coef_)
            if isinstance(pred, Tensor):
                pred = pred.reshape(-1, 1)
            else:
                pred = Tensor(pred.reshape(-1, 1))
            
            # Add feature contribution if available
            if features is not None and self.feature_weights is not None:
                if features.data.ndim == 3:
                    feat = features.data[:, min(i, features.data.shape[1]-1), :]
                else:
                    feat = features.data[:, min(i, features.data.shape[1]-1)]
                feat_contribution = Tensor(feat) @ self.feature_weights
                pred = pred + feat_contribution.reshape(-1, 1)
            
            # Add bias term
            pred = pred + self.bias
            predictions.append(pred)
            
        # Concatenate predictions along sequence dimension
        return Tensor(torch.cat([p.data for p in predictions], dim=1).unsqueeze(-1))


class ARIMA(BaseModel):
    def __init__(self, p=1, d=0, q=0, feature_size=0, output_steps=10):
        super().__init__()
        self.p = p  # AR order
        self.d = d  # Difference order
        self.q = q  # MA order
        self.feature_size = feature_size
        self.output_steps = output_steps
        
        # Neural network layers for AR, MA and seasonal components
        self.fc1 = Linear(p + q + feature_size, 64)  # Combined input features
        self.fc2 = Linear(64, 64)
        self.fc3 = Linear(64, output_steps)
        self.relu = ReLU()
        
    def difference(self, x: Tensor, interval: int = 1) -> Tensor:
        """Apply differencing of specified interval."""
        if isinstance(x, Tensor):
            data = x.data
        else:
            data = x
            
        return Tensor([data[i] - data[i - interval] for i in range(interval, len(data))])
    
    def inverse_difference(self, diff_values: Tensor, last_obs: Tensor) -> Tensor:
        """Inverse operation of differencing."""
        if isinstance(diff_values, Tensor):
            diff_data = diff_values.data
        else:
            diff_data = diff_values
            
        if isinstance(last_obs, Tensor):
            last_data = last_obs.data
        else:
            last_data = last_obs
            
        return Tensor(last_data + diff_data)
    
    def forward(self, x, features=None):
        """Forward pass of ARIMA model."""
        # Handle input shape
        if x.data.ndim == 3:
            batch_size, seq_length, input_dim = x.shape
            x = Tensor(x.data[:, :, 0])  # Only take first feature
        else:
            batch_size, seq_length = x.shape
            
        # Store original data for inverse differencing
        x_orig = x
        
        # Apply differencing if needed
        if self.d > 0:
            x = self.difference(x, self.d)
            
        # Initialize lists for predictions and errors
        predictions = []
        errors = []
        
        # Calculate start index
        start_idx = max(self.p, seq_length - self.output_steps)
        
        for i in range(start_idx, seq_length):
            # AR features: past p values
            ar_features = x[:, i-self.p:i]
            
            # MA features: past q errors
            if len(errors) >= self.q:
                ma_features = torch.stack(errors[-self.q:], dim=1)
            else:
                ma_features = torch.zeros(batch_size, self.q)
                
            # Combine AR and MA features
            combined_features = torch.cat([ar_features, ma_features], dim=1)
            
            # Add external features if available
            if features is not None:
                feat = features.data[:, i, :]
                combined_features = torch.cat([combined_features, feat], dim=1)
            
            # Neural network forward pass
            combined_features = Tensor(combined_features)
            h1 = self.relu(self.fc1(combined_features))
            h2 = self.relu(self.fc2(h1))
            pred = self.fc3(h2)
            
            predictions.append(pred)
            
            # Calculate error for MA component
            if i < seq_length - 1:
                error = x[:, i+1] - pred.data[:, 0]
                errors.append(error)
        
        # Stack predictions
        predictions = torch.stack([p.data for p in predictions], dim=1)
        predictions = Tensor(predictions)
        
        # Inverse differencing if needed
        if self.d > 0:
            predictions = self.inverse_difference(predictions, x_orig[:, -self.output_steps:])
            
        return predictions.unsqueeze(-1)


class LiteTCN(BaseModel):
    """Lightweight Temporal Convolutional Network."""
    
    def __init__(self, input_size=1, feature_size=0, hidden_size=32, kernel_size=3, dropout=0.1):
        super().__init__()
        total_input = input_size + feature_size
        self.conv1 = nn.Conv1d(total_input, hidden_size, kernel_size, padding='same')
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding='same')
        self.conv3 = nn.Conv1d(hidden_size, 1, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, features=None):
        # Combine raw data and features
        if features is not None:
            x = torch.cat([x, features], dim=-1)  # Concatenate along feature dimension
        
        x = x.transpose(1, 2)  # (batch_size, total_input, sequence_length)
        
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.conv3(x)
        
        return x.transpose(1, 2)  # (batch_size, sequence_length, 1)


class TimeCNN(BaseModel):
    """Convolutional Neural Network for time series."""
    
    def __init__(self, input_size=1, feature_size=0, hidden_size=32, num_layers=3):
        super().__init__()
        total_input = input_size + feature_size
        layers = []
        in_channels = total_input
        
        for _ in range(num_layers):
            layers.extend([
                nn.Conv1d(in_channels, hidden_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.MaxPool1d(2, 2)
            ])
            in_channels = hidden_size
            
        self.features = nn.Sequential(*layers)
        self.regressor = nn.Linear(hidden_size, 1)
        
    def forward(self, x, features=None):
        if features is not None:
            x = torch.cat([x, features], dim=-1)
        x = x.transpose(1, 2)  # (batch_size, total_input, sequence_length)
        x = self.features(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        return self.regressor(x)


class TimeLinear(BaseModel):
    """Linear model with time-based features."""
    
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        return self.linear3(x)


class TimeGRU(BaseModel):
    """GRU-based model for time series prediction."""
    
    def __init__(self, input_size=1, feature_size=0, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        total_input = input_size + feature_size
        self.gru = nn.GRU(
            input_size=total_input,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x, features=None):
        if features is not None:
            x = torch.cat([x, features], dim=-1)
        out, _ = self.gru(x)
        return self.linear(out)


class CustomLinear(BaseModel):
    """Custom implementation of linear model"""
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.linear1 = Linear(input_size, hidden_size)
        self.linear2 = Linear(hidden_size, hidden_size)
        self.linear3 = Linear(hidden_size, 1)
        
    def forward(self, x, features=None):
        if features is not None:
            x = torch.cat([x, features], dim=-1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.linear3(x)


class CustomCNN(BaseModel):
    """Custom implementation of 1D CNN"""
    def __init__(self, input_size=1, feature_size=0, hidden_size=32, num_layers=3):
        super().__init__()
        total_input = input_size + feature_size
        self.convs = []
        
        in_channels = total_input
        for _ in range(num_layers):
            self.convs.append(Conv1d(in_channels, hidden_size, kernel_size=3, padding=1))
            in_channels = hidden_size
            
        self.final = Linear(hidden_size, 1)


class EnhancedTCN(BaseModel):
    """Enhanced Temporal Convolutional Network with residual connections and dilated convolutions."""
    
    class DilatedResidualBlock(Module):
        """Dilated Residual Block with skip connection."""
        
        def __init__(self, channels, kernel_size, dilation):
            super().__init__()  # Need to call Module's __init__
            self.conv_block1 = Sequential(
                Conv1d(channels, channels, kernel_size, dilation=dilation, padding='same'),
                LayerNorm(channels),
                ReLU(),
                Dropout(0.1)
            )
            self.conv_block2 = Sequential(
                Conv1d(channels, channels, kernel_size, dilation=dilation, padding='same'),
                LayerNorm(channels),
                ReLU(),
                Dropout(0.1)
            )
            
        def forward(self, x):
            identity = x
            out = self.conv_block1(x)
            out = self.conv_block2(out)
            return F.relu(out + identity)
    
    def __init__(self, input_size=1, feature_size=0, hidden_size=32, 
                 num_layers=3, kernel_size=3, output_steps=1):
        super().__init__()
        self.output_steps = output_steps
        total_input = input_size + feature_size
        
        # Initial projection
        self.input_proj = Conv1d(total_input, hidden_size, 1)
        
        # Dilated convolution blocks
        self.blocks = ModuleList([
            self.DilatedResidualBlock(
                hidden_size, kernel_size, dilation=2**i
            ) for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = Conv1d(hidden_size, output_steps, 1)
        
    def forward(self, x, features=None):
        if features is not None:
            x = torch.cat([x, features], dim=-1)
        
        # Transform to (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # Initial projection
        x = self.input_proj(x)
        
        # Apply TCN blocks
        for block in self.blocks:
            x = block(x)
            
        # Output projection
        x = self.output_proj(x)
        
        # Return to (batch, seq_len, output_steps)
        return x.transpose(1, 2)


class AdvancedTCN(BaseModel):
    """Advanced Temporal Convolutional Network with skip connections and attention."""
    
    class DilatedResidualBlock(Module):
        def __init__(self, channels, kernel_size, dilation, dropout=0.1):
            super().__init__()
            self.conv_block1 = Sequential(
                Conv1d(channels, channels, kernel_size, dilation=dilation, padding='same'),
                LayerNorm(channels),
                ReLU(),
                Dropout(dropout)
            )
            self.conv_block2 = Sequential(
                Conv1d(channels, channels, kernel_size, dilation=dilation, padding='same'),
                LayerNorm(channels),
                ReLU(),
                Dropout(dropout)
            )
            self.attention = Attention(channels, num_heads=4)
            
        def forward(self, x):
            identity = x
            
            # Convolution path
            out = self.conv_block1(x)
            out = self.conv_block2(out)
            
            # Attention path
            att = self.attention(out, out, out)
            
            # Combine paths
            return F.relu(out + att + identity)
    
    def __init__(self, input_size=1, feature_size=0, hidden_size=32, 
                 num_layers=3, kernel_size=3):
        super().__init__()
        total_input = input_size + feature_size
        
        self.input_proj = Conv1d(total_input, hidden_size, 1)
        
        # Dilated blocks with increasing receptive field
        self.blocks = ModuleList([
            self.DilatedResidualBlock(
                hidden_size, kernel_size, dilation=2**i
            ) for i in range(num_layers)
        ])
        
        # Global context
        self.global_context = Sequential(
            GlobalAveragePool(),
            Linear(hidden_size, hidden_size),
            ReLU()
        )
        
        self.output_proj = Sequential(
            Conv1d(hidden_size * 2, hidden_size, 1),
            ReLU(),
            Conv1d(hidden_size, 1, 1)
        )