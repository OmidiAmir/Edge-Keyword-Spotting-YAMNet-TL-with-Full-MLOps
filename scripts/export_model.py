
import argparse, yaml, torch, mlflow, mlflow.pytorch
from kws.models.cnn import TinyCNN

def load_yaml(p):
    import yaml
    with open(p, "r") as f:
        return yaml.safe_load(f) or {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_model.yaml")
    ap.add_argument("--run_id", required=True, help="MLflow run id to load model from (the training run)")
    ap.add_argument("--out", default="models/ts_model.pt", help="TorchScript output path")
    ap.add_argument("--artifact_path", default="torchscript", help="MLflow artifact subfolder")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load trained PyTorch model from MLflow run
    model_uri = f"runs:/{args.run_id}/model"
    model = mlflow.pytorch.load_model(model_uri).to(device).eval()

    # 2) Export TorchScript
    example = torch.randn(1, cfg["n_mfcc"], 100, device=device)  # example time length
    scripted = torch.jit.trace(model, example)
    scripted.save(args.out)
    print(f"Saved TorchScript model to: {args.out}")

    # 3) Log TorchScript into the SAME MLflow run as an artifact
    with mlflow.start_run(run_id=args.run_id):
        mlflow.log_artifact(args.out, artifact_path=args.artifact_path)
        # Optional: also log as an MLflow PyTorch model (scripted)
        mlflow.pytorch.log_model(scripted, artifact_path=f"{args.artifact_path}_mlflow_model")

    print(f"Logged TorchScript to MLflow run {args.run_id} under '{args.artifact_path}/'")

if __name__ == "__main__":
    main()
