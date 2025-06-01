import matplotlib.pyplot as plt
import numpy as np

def plot_predictions(true_values, predicted_values, title='Model Predictions', xlabel='Time', ylabel='Value'):
    plt.figure(figsize=(10, 5))
    plt.plot(true_values, label='True Values', color='blue')
    plt.plot(predicted_values, label='Predicted Values', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def plot_training_progress(training_losses, validation_losses, title='Training Progress', xlabel='Epoch', ylabel='Loss'):
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss', color='blue')
    plt.plot(validation_losses, label='Validation Loss', color='orange')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    plt.show()

def plot_training_curves(results):
    """Plot training and validation curves for all models."""
    plt.figure(figsize=(12, 6))
    
    for name, result in results.items():
        plt.plot(result['train_losses'], label=f'{name} (train)')
        plt.plot(result['val_losses'], '--', label=f'{name} (val)')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/training_curves.png')
    plt.close()

def plot_predictions(results):
    """Plot predictions vs actual values for all models."""
    n_models = len(results)
    fig, axs = plt.subplots(n_models, 1, figsize=(12, 4*n_models))
    
    for i, (name, result) in enumerate(results.items()):
        ax = axs[i] if n_models > 1 else axs
        ax.plot(result['actuals'], label='Actual', alpha=0.7)
        ax.plot(result['predictions'], label='Predicted', alpha=0.7)
        ax.set_title(f'{name} - RMSE: {result["test_rmse"]:.4f}, MAE: {result["test_mae"]:.4f}')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/predictions.png')
    plt.close()