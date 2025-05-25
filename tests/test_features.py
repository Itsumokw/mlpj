import pytest
import numpy as np
from src.core.features import FeatureExtractor

def test_feature_extractor_transform():
    extractor = FeatureExtractor(window_size=5)
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    features = extractor.transform(data)
    
    assert features.shape == (6, 5), "Feature shape should be (6, 5) for input of length 10"
    assert np.allclose(features[0], [3.0, 1.41421356, 4.0, 1.0, 1.0]), "First feature extraction result is incorrect"

def test_sliding_window():
    data = np.array([1, 2, 3, 4, 5])
    window = FeatureExtractor.sliding_window(data, window=3)
    
    expected_shape = (3, 3)
    assert window.shape == expected_shape, f"Sliding window shape should be {expected_shape}"
    assert np.array_equal(window, np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])), "Sliding window output is incorrect"