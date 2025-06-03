import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datetime import datetime
from src.core.models import (
    SimpleARIMA, ARIMA, LiteTCN, TimeCNN,
    TimeGRU, EnhancedTCN, AdvancedTCN
)
from src.core.features import FeatureExtractor
from src.data.dataset import TimeSeriesDataset
from src.data.dataloader import DataLoader
from src.utils.transforms import StandardScaler
from src.utils.metrics import rmse, mae
from src.visualization.plotter import plot_predictions
from src.visualization.param_visualizer import plot_parameter_evolution, plot_weight_distribution
from src.core.optim import Adam, SGD, RMSprop
from src.core.nn.loss import MSELoss
from src.core.tensor import Tensor

# Initialize session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}


def generate_sine_data():
    """Generate synthetic sine wave data"""
    t = np.linspace(0, 100, 1000)
    return np.sin(0.1 * t) + 0.1 * np.sin(0.5 * t) + np.random.normal(0, 0.1, 1000)


def generate_random_walk():
    """Generate synthetic random walk data"""
    steps = np.random.normal(0, 1, 1000)
    return np.cumsum(steps)


def preprocess_data(data, window_size, prediction_steps):
    """Preprocess data and create datasets"""
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    train_size = int(0.7 * len(data_scaled))
    val_size = int(0.15 * len(data_scaled))

    train_data = data_scaled[:train_size]
    val_data = data_scaled[train_size:train_size + val_size]
    test_data = data_scaled[train_size + val_size:]

    train_dataset = TimeSeriesDataset(train_data, window_size, prediction_steps)
    val_dataset = TimeSeriesDataset(val_data, window_size, prediction_steps)
    test_dataset = TimeSeriesDataset(test_data, window_size, prediction_steps)

    return scaler, train_dataset, val_dataset, test_dataset


def train_model(model, train_loader, val_loader, optimizer, criterion, epochs, feature_extractor, use_features):
    """Train a single model"""
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            # Feature extraction
            if use_features and feature_extractor:
                # 确保正确处理各种类型
                if isinstance(x, np.ndarray):
                    x_numpy = x
                elif hasattr(x, 'numpy'):
                    x_numpy = x.numpy()
                elif hasattr(x, 'cpu'):
                    x_numpy = x.cpu().numpy()
                else:
                    x_numpy = np.array(x)

                features = feature_extractor.transform(x_numpy)
                features = Tensor(features)
            else:
                features = None

            # Convert to custom Tensor
            x_tensor = Tensor(x)
            y_tensor = Tensor(y)

            # 其余代码保持不变...

            # Forward pass
            optimizer.zero_grad()
            outputs = model(x_tensor, features)
            loss = criterion(outputs, y_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.data.item()
            model.record_batch_loss(loss.data.item())

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                if use_features and feature_extractor:
                    x_numpy = x.numpy()
                    features = feature_extractor.transform(x_numpy)
                    features = Tensor(features)
                else:
                    features = None

                x_tensor = Tensor(x)
                y_tensor = Tensor(y)

                outputs = model(x_tensor, features)
                loss = criterion(outputs, y_tensor)
                val_loss += loss.data.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Record parameters every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.record_parameters(epoch)

        st.session_state.progress_text.text(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        st.session_state.progress_bar.progress((epoch + 1) / epochs)

    return train_losses, val_losses


def evaluate_model(model, test_loader, feature_extractor, scaler, use_features):
    """Evaluate a trained model"""
    model.eval()
    test_rmse = []
    test_mae = []
    predictions = []
    actuals = []

    with torch.no_grad():
        for x, y in test_loader:
            if use_features and feature_extractor:
                x_numpy = x.numpy()
                features = feature_extractor.transform(x_numpy)
                features = Tensor(features)
            else:
                features = None

            x_tensor = Tensor(x)
            y_tensor = Tensor(y)

            outputs = model(x_tensor, features)

            # Inverse scaling
            if scaler:
                outputs.data = scaler.inverse_transform(outputs.data)
                y_tensor.data = scaler.inverse_transform(y_tensor.data)

            # Calculate metrics
            test_rmse.append(rmse(y_tensor, outputs).item())
            test_mae.append(mae(y_tensor, outputs).item())

            # Store results
            predictions.extend(outputs.data.cpu().numpy().flatten())
            actuals.extend(y_tensor.data.cpu().numpy().flatten())

    return {
        'test_rmse': np.mean(test_rmse),
        'test_mae': np.mean(test_mae),
        'predictions': np.array(predictions),
        'actuals': np.array(actuals)
    }


def main():
    st.title("📈 Time Series Forecasting Lab")
    st.markdown("Interactive platform for comparing time series forecasting models")

    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.header("⚙️ Experiment Configuration")

        # Model selection
        model_options = ['SimpleARIMA', 'ARIMA', 'LiteTCN', 'TimeCNN',
                         'TimeGRU', 'EnhancedTCN', 'AdvancedTCN']
        selected_models = st.multiselect(
            "Select Models",
            model_options,
            default=['LiteTCN', 'TimeGRU']
        )

        # Dataset selection
        dataset_options = ['Sine Wave', 'Random Walk', 'Upload Custom']
        dataset_choice = st.selectbox("Dataset", dataset_options)

        # Handle custom data upload
        custom_data = None
        if dataset_choice == 'Upload Custom':
            uploaded_file = st.file_uploader("Upload CSV", type="csv")
            if uploaded_file:
                df = pd.read_csv(uploaded_file)
                if len(df.columns) != 1:
                    st.warning("Please upload a single-column CSV file")
                else:
                    custom_data = df.values.flatten()

        # Hyperparameters
        st.subheader("Hyperparameters")
        lr = st.slider("Learning Rate", 0.0001, 0.1, 0.01, 0.001)
        epochs = st.slider("Epochs", 10, 500, 100, 10)
        batch_size = st.slider("Batch Size", 8, 128, 32, 8)
        window_size = st.slider("Window Size", 10, 100, 50, 5)
        prediction_steps = st.slider("Prediction Steps", 1, 20, 10, 1)

        # Optimizer selection
        optimizer_options = ['Adam', 'SGD', 'RMSprop']
        selected_optimizer = st.selectbox("Optimizer", optimizer_options)

        # Feature extractor
        use_features = st.checkbox("Use Feature Extractor", True)

        # Start training
        start_training = st.button("🚀 Start Training")

    # ==================== MAIN CONTENT ====================
    # Initialize data
    if dataset_choice == 'Sine Wave':
        data = generate_sine_data()
    elif dataset_choice == 'Random Walk':
        data = generate_random_walk()
    elif custom_data is not None:
        data = custom_data
    else:
        data = generate_sine_data()  # Default

    # Preprocess data
    scaler, train_dataset, val_dataset, test_dataset = preprocess_data(
        data, window_size, prediction_steps
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize feature extractor
    feature_extractor = FeatureExtractor(window_size) if use_features else None

    # Initialize models
    models = {}
    for name in selected_models:
        if name == 'SimpleARIMA':
            models[name] = SimpleARIMA(p=5, feature_size=feature_extractor.feature_size() if use_features else 0,
                                       output_steps=prediction_steps)
        elif name == 'ARIMA':
            models[name] = ARIMA(p=2, d=1, q=1, feature_size=feature_extractor.feature_size() if use_features else 0)
        elif name == 'LiteTCN':
            models[name] = LiteTCN(input_size=1, feature_size=feature_extractor.feature_size() if use_features else 0)
        elif name == 'TimeCNN':
            models[name] = TimeCNN(input_size=1, feature_size=feature_extractor.feature_size() if use_features else 0)
        elif name == 'TimeGRU':
            models[name] = TimeGRU(input_size=1, feature_size=feature_extractor.feature_size() if use_features else 0)
        elif name == 'EnhancedTCN':
            models[name] = EnhancedTCN(input_size=1,
                                       feature_size=feature_extractor.feature_size() if use_features else 0)
        elif name == 'AdvancedTCN':
            models[name] = AdvancedTCN(input_size=1,
                                       feature_size=feature_extractor.feature_size() if use_features else 0)

    # Training section
    if start_training:
        st.session_state.progress_bar = st.progress(0)
        st.session_state.progress_text = st.empty()

        results = {}

        for i, (name, model) in enumerate(models.items()):
            st.session_state.progress_text.text(f"Training {name} ({i + 1}/{len(models)})")

            # Initialize optimizer
            if selected_optimizer == 'Adam':
                optimizer = Adam(model.parameters(), lr=lr)
            elif selected_optimizer == 'SGD':
                optimizer = SGD(model.parameters(), lr=lr)
            else:  # RMSprop
                optimizer = RMSprop(model.parameters(), lr=lr)

            criterion = MSELoss()

            # Train model
            train_losses, val_losses = train_model(
                model, train_loader, val_loader, optimizer, criterion,
                epochs, feature_extractor, use_features
            )

            # Evaluate model
            eval_results = evaluate_model(
                model, test_loader, feature_extractor, scaler, use_features
            )

            # Store results
            results[name] = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'batch_losses': model.batch_losses,
                'test_rmse': eval_results['test_rmse'],
                'test_mae': eval_results['test_mae'],
                'predictions': eval_results['predictions'],
                'actuals': eval_results['actuals'],
                'param_history': model.param_history
            }

        st.session_state.trained_models = results
        st.success("✅ Training completed!")

    # Display results if available
    if st.session_state.trained_models:
        st.header("📊 Results Analysis")

        # Model comparison table
        st.subheader("Model Performance")
        metrics_data = []
        for name, res in st.session_state.trained_models.items():
            metrics_data.append({
                'Model': name,
                'Train Loss': f"{res['train_losses'][-1]:.4f}",
                'Val Loss': f"{res['val_losses'][-1]:.4f}",
                'Test RMSE': f"{res['test_rmse']:.4f}",
                'Test MAE': f"{res['test_mae']:.4f}"
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df.style.highlight_min(axis=0, subset=['Test RMSE', 'Test MAE']))

        # Loss curves
        st.subheader("Training Curves")
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, res in st.session_state.trained_models.items():
            ax.plot(res['train_losses'], label=f"{name} Train")
            ax.plot(res['val_losses'], '--', label=f"{name} Val")
        ax.set_title("Training & Validation Loss")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Parameter visualization
        st.subheader("Parameter Evolution")
        selected_model = st.selectbox("Select Model", list(st.session_state.trained_models.keys()))

        if selected_model:
            model_results = st.session_state.trained_models[selected_model]
            if model_results['param_history']:
                layer_names = list(model_results['param_history'][0].keys())
                selected_layer = st.selectbox("Select Layer", layer_names)

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Parameter Evolution**")
                    fig1 = plot_parameter_evolution(model_results['param_history'], selected_layer)
                    if fig1:
                        st.pyplot(fig1)

                with col2:
                    epoch_idx = st.slider("Select Epoch", 0, len(model_results['param_history']) - 1, 0)
                    st.write(f"**Weight Distribution (Epoch {epoch_idx * 5 + 5})**")
                    fig2 = plot_weight_distribution(model_results['param_history'], epoch_idx, selected_layer)
                    if fig2:
                        st.pyplot(fig2)

        # Predictions visualization
        st.subheader("Predictions vs Actuals")
        selected_model = st.selectbox("Select Model for Prediction", list(st.session_state.trained_models.keys()))

        if selected_model:
            model_results = st.session_state.trained_models[selected_model]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(model_results['actuals'], label='Actual', alpha=0.7)
            ax.plot(model_results['predictions'], label='Predicted', alpha=0.7)
            ax.set_title(f"{selected_model} Predictions vs Actuals")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Value")
            ax.legend()
            st.pyplot(fig)

        # Batch loss visualization
        st.subheader("Batch Loss Progression")
        selected_model = st.selectbox("Select Model for Batch Loss", list(st.session_state.trained_models.keys()))

        if selected_model:
            batch_losses = st.session_state.trained_models[selected_model]['batch_losses']
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(batch_losses, alpha=0.7)
            ax.set_title(f"{selected_model} Batch Loss Progression")
            ax.set_xlabel("Batch")
            ax.set_ylabel("Loss")
            ax.grid(True)
            st.pyplot(fig)


if __name__ == "__main__":
    main()