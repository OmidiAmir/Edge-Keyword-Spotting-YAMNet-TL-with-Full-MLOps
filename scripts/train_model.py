import argparse
from kws.training.train import train_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_mfcc", type=int, default=40)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=2)
    args = p.parse_args()

    train_model(**vars(args))

if __name__ == "__main__":
    main()
