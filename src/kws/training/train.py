# src/kws/training/train.py
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from kws.data import KSDataset, collate_pad
from kws.models.cnn import TinyCNN

# MLflow
import mlflow
import mlflow.pytorch


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
    # MLflow toggles
    use_mlflow: bool = True,
    experiment: str = "kws-baseline",
    run_name: str | None = None,
) -> nn.Module:
    """
    Minimal training loop for TinyCNN on Speech Commands with optional MLflow logging.
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

    # MLflow setup
    if use_mlflow:
        mlflow.set_experiment(experiment)
        if run_name is None:
            run_name = f"run-{time.strftime('%Y%m%d-%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "n_mfcc": n_mfcc,
            "n_classes": n_classes,
            "num_workers": num_workers,
            "device": device,
        })

    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        epoch_loss_sum, epoch_count = 0.0, 0

        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            epoch_count += 1
            pbar.set_postfix(loss=f"{epoch_loss_sum / epoch_count:.4f}")

        val_acc = evaluate(model, val_loader, device)
        avg_loss = epoch_loss_sum / max(epoch_count, 1)
        print(f"[Epoch {epoch}] loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        if use_mlflow:
            mlflow.log_metrics({"train_loss": avg_loss, "val_acc": val_acc}, step=epoch)

    if use_mlflow:
        # log final model artifact
        mlflow.pytorch.log_model(model, artifact_path="model")
        mlflow.end_run()

    return model
