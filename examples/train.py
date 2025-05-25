import numpy as np
import torch
from src.core.models import SimpleARIMA, LiteTCN
from src.core.features import FeatureExtractor
from src.data.dataset import Dataset
from src.data.dataloader import DataLoader
from src.utils.metrics import mean_squared_error
from src.visualization.plotter import plot_results

# Hyperparameters
window_size = 30
batch_size = 16
epochs = 100
learning_rate = 0.001

# Load dataset
data = np.sin(np.linspace(0, 20, 1000)) + np.random.normal(0, 0.1, 1000)
dataset = Dataset(data, window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize feature extractor
feature_extractor = FeatureExtractor(window_size)

# Initialize model
model = SimpleARIMA(p=5)  # Example with SimpleARIMA

# Training loop
for epoch in range(epochs):
    for features, targets in dataloader:
        # Extract features
        extracted_features = feature_extractor.transform(features.numpy())
        
        # Convert to tensor
        inputs = torch.tensor(extracted_features, dtype=torch.float32)
        targets = torch.tensor(targets.numpy(), dtype=torch.float32)

        # Forward pass
        preds = model.predict(inputs)

        # Calculate loss
        loss = mean_squared_error(targets, preds)

        # Backward pass and optimization
        loss.backward()
        # optimizer.step()  # Uncomment and define optimizer as needed

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Visualization
plot_results(targets.numpy(), preds.detach().numpy())