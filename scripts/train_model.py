# scripts/train_model.py
import argparse, yaml
from kws.training.train import train_model

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_model.yaml")
    # Optional CLI overrides:
    p.add_argument("--data_root")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--n_mfcc", type=int)
    p.add_argument("--n_classes", type=int)
    p.add_argument("--num_workers", type=int)
    args = p.parse_args()

    cfg = load_yaml(args.config)

    # Apply CLI overrides if provided
    for k in ["data_root","epochs","batch_size","lr","n_mfcc","n_classes","num_workers"]:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    train_model(**cfg)

if __name__ == "__main__":
    main()
