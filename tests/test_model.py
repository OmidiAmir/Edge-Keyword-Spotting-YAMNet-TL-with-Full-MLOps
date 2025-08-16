# tests/test_model.py
import torch
from kws.models.cnn import TinyCNN

def test_forward_shape():
    model = TinyCNN(n_mfcc=40, n_classes=10)
    X = torch.randn(2, 40, 100)  # [B, n_mfcc, T]
    out = model(X)
    assert out.shape == (2, 10)
