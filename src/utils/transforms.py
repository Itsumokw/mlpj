class Normalization:
    """Class for normalizing data."""
    
    @staticmethod
    def min_max_scaling(data):
        """Apply Min-Max scaling to the data."""
        return (data - data.min()) / (data.max() - data.min() + 1e-8)

    @staticmethod
    def z_score_normalization(data):
        """Apply Z-score normalization to the data."""
        return (data - data.mean()) / (data.std() + 1e-8)


class Augmentation:
    """Class for data augmentation techniques."""
    
    @staticmethod
    def add_noise(data, noise_level=0.01):
        """Add random noise to the data."""
        noise = np.random.normal(0, noise_level, data.shape)
        return data + noise

    @staticmethod
    def time_warp(data):
        """Apply time warping to the data."""
        # Implementation of time warping can be added here
        pass