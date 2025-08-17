# scripts/evaluate_model.py
import argparse, yaml, torch, mlflow, mlflow.pytorch
from torch.utils.data import DataLoader
from kws.data import KSDataset, collate_pad
from kws.training.train import evaluate

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_model.yaml")
    p.add_argument("--run_id", required=True, help="MLflow Run ID to evaluate")
    p.add_argument("--experiment", default="kws-baseline")
    p.add_argument("--run_name", default=None)
    args = p.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model from the specific MLflow run
    model_uri = f"runs:/{args.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri).to(device).eval()

    # Test data
    test_ds = KSDataset(root=cfg["data_root"], split="test", n_mfcc=cfg["n_mfcc"])
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate_pad)

    # Evaluate
    acc = evaluate(model, test_loader, device)
    print(f"test_acc={acc:.4f} (run_id={args.run_id})")

    # Log evaluation as a separate run
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name or f"eval-{args.run_id}"):
        mlflow.set_tag("evaluated_run_id", args.run_id)
        mlflow.log_metric("test_acc", acc)

if __name__ == "__main__":
    main()
