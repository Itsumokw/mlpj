import pytest
import numpy as np
from src.core.models import SimpleARIMA, LiteTCN

def test_arima_fit():
    model = SimpleARIMA(p=2)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    model.fit(X)
    assert len(model.coef_) == 2, "Coefficient count should match the p parameter"

def test_arima_predict_shape():
    model = SimpleARIMA(p=3)
    X = np.random.randn(100)
    model.fit(X)
    preds = model.predict(X[-3:], steps=5)
    assert len(preds) == 5, "Predicted steps should match the steps parameter"

def test_lite_tcn_forward_shape():
    model = LiteTCN(input_size=1, hidden_size=8, kernel_size=3)
    X = torch.randn(10, 30)  # (batch_size, seq_len)
    output = model(X)
    assert output.shape == (10, 30), "Output shape should match input shape"