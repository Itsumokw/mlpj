import numpy as np
import torch
from src.core.models import (
    SimpleARIMA, ARIMA, LiteTCN, TimeCNN, 
    TimeGRU, EnhancedTCN, AdvancedTCN
)
from src.core.features import FeatureExtractor
from src.data.dataset import TimeSeriesDataset
from src.data.dataloader import DataLoader
from src.utils.transforms import StandardScaler
from src.utils.metrics import rmse, mae
from src.visualization.plotter import plot_predictions, plot_training_curves
from src.core.optim import Adam  # Add this import
from src.core.nn.loss import MSELoss  # Add this import
from src.core.tensor import Tensor  # Add this import
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

# Set the path to the file you'd like to load

# path = kagglehub.dataset_download("rakannimer/AirPassengers")
# print("Path to dataset files:", path)
# Load the latest version
# file_path = "C:\\Users\\Administrator\\.cache\\kagglehub\\datasets\\rakannimer\\air-passengers\\versions\\1\\AirPassengers.csv"
# print("File path:", file_path)

# hf_dataset = pd.read_csv(file_path)

# print("Hugging Face Dataset:", hf_dataset)

def train_and_evaluate(model_name: str, model, train_loader, val_loader, test_loader,
                      feature_extractor, scaler, device='cpu'):
    """Train and evaluate a single model."""
    print(f"\nTraining {model_name}...")
    
    # Move model to device
    # model = model.to(device)
    
    # Use our own optimizer and loss implementations
    optimizer = Adam(model.parameters(), lr=0.01)  # Changed from torch.optim
    criterion = MSELoss()  # Changed from torch.nn
    
    # Training metrics
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Training loop
    for epoch in range(100):
        # Training phase
        model.train()
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            # Extract features only from passenger data (first column)
            if feature_extractor is not None:
                x_numpy = x.numpy()
                # Extract passenger count column for feature extraction
                passenger_data = x_numpy[:, :, 0:1]  # Keep the last dimension
                features = feature_extractor.transform(passenger_data)
                features = Tensor(features)
            else:
                features = None
            
            # Forward pass
            optimizer.zero_grad()
            # Convert PyTorch tensors to our Tensor class only once
            x_tensor = Tensor(x)
            y_tensor = Tensor(y)
            
            # Print shapes for debugging
            if batch_idx == 0 and epoch == 0:
                print(f"Input shape: {x_tensor.data.shape}")
                print(f"Target shape: {y_tensor.data.shape}")
                if features is not None:
                    print(f"Features shape: {features.data.shape}")
            
            outputs = model(x_tensor, features)
            
            # Ensure shapes match
            if outputs.data.shape != y_tensor.data.shape:
                raise ValueError(f"Shape mismatch in {model_name}: outputs {outputs.data.shape} "
                              f"!= targets {y_tensor.data.shape}")
            
            loss = criterion(outputs, y_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data.item()
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                # Extract features if needed
                if feature_extractor is not None:
                    x_numpy = x.numpy() if hasattr(x, 'numpy') else x.cpu().numpy()
                    passenger_data = x_numpy[:, :, 0:1]
                    features = feature_extractor.transform(passenger_data)
                    features = Tensor(features)
                else:
                    features = None
                
                # Convert inputs to our Tensor class
                x_tensor = Tensor(x)
                y_tensor = Tensor(y)
                
                outputs = model(x_tensor, features)
                loss = criterion(outputs, y_tensor)
                val_loss += loss.data.item()  # Access underlying PyTorch tensor
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Test phase
    model.eval()
    test_rmse = []
    test_mae = []
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for x, y in test_loader:
            if feature_extractor is not None:
                x_numpy = x.numpy() if hasattr(x, 'numpy') else x.numpy()
                passenger_data = x_numpy[:, :, 0:1]
                features = feature_extractor.transform(passenger_data)
                features = Tensor(features)
            else:
                features = None
            
            # Convert inputs to our Tensor class
            x_tensor = Tensor(x)
            y_tensor = Tensor(y)
            
            outputs = model(x_tensor, features)
            
            # Transform back to original scale
            if scaler is not None:
                outputs.data = scaler.inverse_transform(outputs.data)
                y_tensor.data = scaler.inverse_transform(y_tensor.data)
            
            # Calculate metrics
            test_rmse.append(rmse(y_tensor, outputs).item())
            test_mae.append(mae(y_tensor, outputs).item())
            
            # Store predictions and actuals
            predictions.extend(outputs.data.cpu().numpy())
            actuals.extend(y_tensor.data.cpu().numpy())
    
    return {
        'name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_rmse': np.mean(test_rmse),
        'test_mae': np.mean(test_mae),
        'predictions': np.array(predictions),
        'actuals': np.array(actuals)
    }

def main():
    # Load and preprocess data
    file_path = "C:\\Users\\Administrator\\.cache\\kagglehub\\datasets\\rakannimer\\air-passengers\\versions\\1\\AirPassengers.csv"
    df = pd.read_csv(file_path)
    
    # Convert Month column to datetime
    df['Month'] = pd.to_datetime(df['Month'])
    
    # Extract time features
    df['year'] = df['Month'].dt.year
    df['month'] = df['Month'].dt.month
    df['year_progress'] = (df['month'] - 1) / 12  # Normalized month position in year
    
    # Create seasonal features
    df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
    df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Convert passengers data to float and reshape
    passengers_data = df['#Passengers'].values.astype(np.float32).reshape(-1, 1)
    
    # Normalize passenger numbers
    scaler = StandardScaler()
    passengers_scaled = scaler.fit_transform(passengers_data)
    
    # Prepare sequences
    def create_sequences(data, time_features, window_size, prediction_steps):
        X, y = [], []
        for i in range(len(data) - window_size - prediction_steps + 1):
            # Input sequence includes passenger numbers and time features
            features = np.concatenate([
                data[i:i+window_size],
                time_features[i:i+window_size]
            ], axis=1)
            
            # Target sequence is next prediction_steps passenger numbers
            target = data[i+window_size:i+window_size+prediction_steps]
            
            X.append(features)
            y.append(target)
        return np.array(X), np.array(y)
    
    # Create time features array
    time_features = np.column_stack([
        df['year_progress'].values,
        df['sin_month'].values,
        df['cos_month'].values
    ])
    
    # Adjust window size and prediction steps
    window_size = 24  # Increase window size to have enough historical data
    prediction_steps = 6  # Decrease prediction steps to ensure we have enough data
    
    # Create sequences with adjusted parameters
    X, y = create_sequences(passengers_scaled, time_features, window_size, prediction_steps)
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    
    # Create dataloaders
    batch_size = 16  # Smaller batch size for this dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(window_size)
    feature_dim = feature_extractor.feature_size()
    
    # Define models with adjusted parameters
    models = {
        'ARIMA': ARIMA(
            p=12,  # Keep seasonal AR order
            d=1,   # Keep first-order differencing
            q=1,   # Keep first-order MA
            feature_size=feature_dim,
            output_steps=prediction_steps  # Use adjusted prediction steps
        ),
        'TimeGRU': TimeGRU(
            input_size=1,
            feature_size=feature_dim,
            hidden_size=64,
            num_layers=2
        ),
        'AdvancedTCN': AdvancedTCN(
            input_size=1,
            feature_size=feature_dim,
            hidden_size=32,
            num_layers=3
        )
    }
    
    # Train and evaluate models
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for name, model in models.items():
        results[name] = train_and_evaluate(
            name, model, train_loader, val_loader, test_loader,
            feature_extractor, scaler, device
        )
    
    # Plot results
    plot_training_curves(results)
    plot_predictions(results)

if __name__ == '__main__':
    main()