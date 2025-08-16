# src/kws/training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from kws.data import KSDataset, collate_pad
from kws.models.cnn import TinyCNN


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        pred = model(X).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


def train_model(
    data_root: str = "data",
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    n_mfcc: int = 40,
    n_classes: int = 10,
    num_workers: int = 2,
    device: str | None = None,
) -> nn.Module:
    """
    Minimal training loop for TinyCNN on Speech Commands (10 classes).

    Returns:
        Trained nn.Module.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets & loaders
    train_ds = KSDataset(root=data_root, split="train", n_mfcc=n_mfcc)
    val_ds   = KSDataset(root=data_root, split="val",   n_mfcc=n_mfcc)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_pad, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_pad, num_workers=num_workers, pin_memory=True
    )

    # Model/opt/loss
    model = TinyCNN(n_mfcc=n_mfcc, n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        running_loss = 0.0

        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss/ (pbar.n or 1):.4f}")

        val_acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch}] val_acc = {val_acc:.4f}")

    return model
