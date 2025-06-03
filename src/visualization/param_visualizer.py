import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict


def plot_parameter_evolution(param_history: List[Dict], layer_name: str):
    """Plot the evolution of model parameters over epochs"""
    if not param_history or layer_name not in param_history[0]:
        return None

    epochs = list(range(len(param_history)))
    means = [p[layer_name]['mean'] for p in param_history]
    stds = [p[layer_name]['std'] for p in param_history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, means, 'b-', label='Mean')
    plt.fill_between(epochs,
                     [m - s for m, s in zip(means, stds)],
                     [m + s for m, s in zip(means, stds)],
                     alpha=0.2, color='b')
    plt.title(f'{layer_name} Weight Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Value')
    plt.legend()
    plt.grid(True)
    return plt


def plot_weight_distribution(param_history: List[Dict], epoch_idx: int, layer_name: str):
    """Plot the distribution of weights at a specific epoch"""
    if epoch_idx >= len(param_history) or layer_name not in param_history[epoch_idx]:
        return None

    weights = param_history[epoch_idx][layer_name]['values']

    plt.figure(figsize=(10, 6))
    sns.histplot(weights, kde=True)
    plt.title(f'{layer_name} Weight Distribution at Epoch {epoch_idx + 1}')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    return plt