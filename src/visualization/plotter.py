import matplotlib.pyplot as plt

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