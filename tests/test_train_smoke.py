import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from kws.data import KSDataset, collate_pad
from kws.models.cnn import TinyCNN

def test_smoke_train_one_batch():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tiny subsets to keep test fast
    train_ds_full = KSDataset(root="data", split="train", n_mfcc=40)
    val_ds_full   = KSDataset(root="data", split="val",   n_mfcc=40)

    train_ds = Subset(train_ds_full, range(64))
    val_ds   = Subset(val_ds_full,   range(64))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_pad)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, collate_fn=collate_pad)

    model = TinyCNN(n_mfcc=40, n_classes=10).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    # one training step (single batch)
    model.train()
    X, y = next(iter(train_loader))
    X, y = X.to(device), y.to(device)
    opt.zero_grad()
    logits = model(X)
    loss = crit(logits, y)
    loss.backward()
    opt.step()

    assert torch.isfinite(loss).item() is True

    # quick eval on tiny val subset
    model.eval()
    with torch.no_grad():
        correct = total = 0
        for Xv, yv in val_loader:
            Xv, yv = Xv.to(device), yv.to(device)
            pred = model(Xv).argmax(1)
            correct += (pred == yv).sum().item()
            total += yv.numel()
        acc = correct / total if total else 0.0

    assert 0.0 <= acc <= 1.0
