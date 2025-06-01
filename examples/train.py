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
            # Extract features if needed
            if feature_extractor is not None:
                x_numpy = x.numpy() if hasattr(x, 'numpy') else x.cpu().numpy()
                features = feature_extractor.transform(x_numpy)
                features = Tensor(features)  # Convert to our Tensor class
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
                    features = feature_extractor.transform(x_numpy)
                    features = Tensor(features)  # Use our Tensor class
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
                features = feature_extractor.transform(x_numpy)
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
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate synthetic data
    t = np.linspace(0, 100, 1000)
    data = np.sin(0.1 * t) + 0.1 * np.sin(0.5 * t) + np.random.normal(0, 0.1, 1000)
    
    # Data preprocessing
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    # Create datasets
    window_size = 50
    prediction_steps = 10
    train_size = int(0.7 * len(data_scaled))
    val_size = int(0.15 * len(data_scaled))
    
    # Split data
    train_data = data_scaled[:train_size]
    val_data = data_scaled[train_size:train_size+val_size]
    test_data = data_scaled[train_size+val_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, window_size, prediction_steps)
    val_dataset = TimeSeriesDataset(val_data, window_size, prediction_steps)
    test_dataset = TimeSeriesDataset(test_data, window_size, prediction_steps)
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(window_size)
    feature_dim = feature_extractor.feature_size()  # Get feature dimension
    
    # Define models to test
    models = {
        'SimpleARIMA': SimpleARIMA(p=5, feature_size=feature_dim, output_steps=prediction_steps),
        'ARIMA': ARIMA(p=2, d=1, q=1, feature_size=feature_dim),
        'LiteTCN': LiteTCN(input_size=1, feature_size=feature_dim),
        'EnhancedTCN': EnhancedTCN(input_size=1, feature_size=feature_dim),
        'TimeCNN': TimeCNN(input_size=1, feature_size=feature_dim),
        'TimeGRU': TimeGRU(input_size=1, feature_size=feature_dim),
        'AdvancedTCN': AdvancedTCN(input_size=1, feature_size=feature_dim)
    }
    
    # Train and evaluate all models
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