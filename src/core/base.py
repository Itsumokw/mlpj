class BaseModel:
    """Base class for all models in the time series prediction framework."""
    
    def __init__(self):
        self.is_trained = False

    def train(self, train_data, train_labels):
        """Train the model using the provided training data and labels."""
        raise NotImplementedError("Train method not implemented.")

    def evaluate(self, test_data, test_labels):
        """Evaluate the model on the test data and return the performance metrics."""
        raise NotImplementedError("Evaluate method not implemented.")

    def save(self, filepath):
        """Save the model to the specified filepath."""
        raise NotImplementedError("Save method not implemented.")

    def load(self, filepath):
        """Load the model from the specified filepath."""
        raise NotImplementedError("Load method not implemented.")