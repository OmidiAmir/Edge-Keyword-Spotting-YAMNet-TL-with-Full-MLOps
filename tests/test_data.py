import pytest
from torch.utils.data import DataLoader
from kws.data import KSDataset, collate_pad

def test_dataset_loads_and_shapes():
    ds = KSDataset(root="data", split="train", n_mfcc=40)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.ndim == 2 and x.shape[0] == 40  # [40, T]
    assert isinstance(y, int) and 0 <= y < 10

def test_collate_pad_batch_shapes():
    ds = KSDataset(root="data", split="train", n_mfcc=40)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=collate_pad)
    X, y = next(iter(loader))
    assert X.ndim == 3 and X.shape[0] == 4      # [B, 40, T_max]
    assert X.shape[1] == 40 and y.shape[0] == 4
