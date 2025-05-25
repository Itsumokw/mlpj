class Dataset:
    """Dataset class for loading and preprocessing time series data."""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = data
        self.labels = labels

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int):
        """Retrieve a data sample and its corresponding label."""
        return self.data[index], self.labels[index]

    def get_samples(self, indices: list):
        """Retrieve multiple samples based on provided indices."""
        return self.data[indices], self.labels[indices]