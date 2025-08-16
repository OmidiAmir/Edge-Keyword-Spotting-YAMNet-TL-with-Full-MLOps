import argparse, os, shutil
from torchaudio.datasets import SPEECHCOMMANDS

CLASSES_10 = ["yes","no","up","down","left","right","on","off","stop","go"]

def main(root, classes):
    os.makedirs(root, exist_ok=True)
    _ = SPEECHCOMMANDS(root=root, download=True, subset="training")
    # Nothing else: torchaudio handles splits; we filter in Dataset.
    print("Downloaded SpeechCommands to", root)
    print("Using classes:", classes or CLASSES_10)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--classes", nargs="*", default=CLASSES_10)
    args = p.parse_args()
    main(args.root, args.classes)