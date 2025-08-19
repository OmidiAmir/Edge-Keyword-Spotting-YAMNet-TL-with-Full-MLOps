import argparse, torch, torchaudio

# Must match training
CLASSES = ["yes","no","up","down","left","right","on","off","stop","go"]

def mfcc_transform(sr=16000, n_mfcc=40, n_fft=512, win_length=400, hop_length=160, f_min=0.0, f_max=None):
    if f_max is None: f_max = sr/2
    melkwargs = dict(n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                     f_min=f_min, f_max=f_max, n_mels=n_mfcc, center=True,
                     pad_mode="reflect", power=2.0, norm=None, mel_scale="htk")
    return torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc, melkwargs=melkwargs)

def load_wav(path, target_sr=16000):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1: wav = wav[:1]
    if sr != target_sr: wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav, target_sr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ts_model.pt", help="TorchScript model path")
    ap.add_argument("--wav", required=True, help="Path to input WAV (mono or stereo)")
    ap.add_argument("--n_mfcc", type=int, default=40)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(args.model, map_location=device).eval()

    wav, sr = load_wav(args.wav, 16000)
    mfcc = mfcc_transform(sr=sr, n_mfcc=args.n_mfcc)(wav).squeeze(0)      # [40, T]
    x = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
    x = x.unsqueeze(0).to(device)  # [1, 40, T]

    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(1).item())
    print(f"pred: {CLASSES[pred]} (class_id={pred})")

if __name__ == "__main__":
    main()
