from fastapi import FastAPI, UploadFile, File, Request
import os, io, tempfile, subprocess, shutil
import json
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

# =========================
# Config (env-overridable)
# =========================
APP_VER = "infer-v1"
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/ts_model.pt")
LABELS_PATH = os.getenv("LABELS_PATH", "/app/models/labels.txt")  # one label per line
INPUT_FORMAT = os.getenv("INPUT_FORMAT", "2d").lower()  # "2d" (Conv2D) or "1d" (Conv1D)
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
N_MFCC = int(os.getenv("N_MFCC", "40"))
N_MELS = int(os.getenv("N_MELS", "64"))
WIN_LENGTH = int(os.getenv("WIN_LENGTH", "400"))   # 25 ms @16k
HOP_LENGTH = int(os.getenv("HOP_LENGTH", "160"))   # 10 ms @16k
FMIN = float(os.getenv("FMIN", "20"))
FMAX = float(os.getenv("FMAX", "8000"))
TOPK = int(os.getenv("TOPK", "3"))

# =========================
# App
# =========================
app = FastAPI(title="KWS Inference API", version=APP_VER)

def ffmpeg_to_wav(in_path: str, out_path: str, ar: int = 16000):
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-i", in_path,
        "-f","wav","-acodec","pcm_s16le","-ac","1","-ar", str(ar),
        out_path
    ]
    subprocess.run(cmd, check=True)

def load_any_audio_bytes(data: bytes, filename_hint: str = "upload.bin", ar: int = 16000):
    """
    Save bytes -> ffmpeg->wav -> read with soundfile -> np.float32 mono
    """
    suffix = os.path.splitext(filename_hint)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
        src.write(data); src_path = src.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as dst:
        dst_path = dst.name
    try:
        ffmpeg_to_wav(src_path, dst_path, ar=ar)
        audio, sr = sf.read(dst_path, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
        return audio, sr
    finally:
        for p in (src_path, dst_path):
            try: os.unlink(p)
            except: pass

def load_labels(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            labels = [ln.strip() for ln in f if ln.strip()]
        return labels
    # default 10-class set (adjust if yours differs)
    return ["yes","no","up","down","left","right","on","off","stop","go"]

# Global objects (loaded at startup)
labels = load_labels(LABELS_PATH)
model = None
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=N_MFCC,
    melkwargs=dict(
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
        power=2.0,
    ),
)

def prepare_input(audio_np: np.ndarray) -> torch.Tensor:
    """
    Return tensor for model:
      - Conv2D model that does unsqueeze(1) internally -> [1, n_mfcc, time]
      - If you set INPUT_FORMAT=1d, returns [1, time, n_mfcc]
    """
    x = torch.from_numpy(audio_np)  # (T,)
    with torch.no_grad():
        feats = mfcc_transform(x)      # [n_mfcc, time]
        # normalize per utterance
        m = feats.mean(dim=(-1, -2), keepdim=True)
        s = feats.std(dim=(-1, -2), keepdim=True).clamp_min(1e-6)
        feats = (feats - m) / s

    # batch dim -> [1, n_mfcc, time]
    feats = feats.unsqueeze(0)

    if INPUT_FORMAT == "1d":
        # [1, n_mfcc, time] -> [1, time, n_mfcc]
        feats = feats.transpose(1, 2)

    return feats


def softmax_topk(logits: torch.Tensor, k: int):
    probs = F.softmax(logits, dim=-1)
    top_p, top_i = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
    top_p = top_p.squeeze(0).cpu().numpy().tolist()
    top_i = top_i.squeeze(0).cpu().numpy().tolist()
    return [(labels[i] if i < len(labels) else str(i), float(p)) for i, p in zip(top_i, top_p)]

@app.on_event("startup")
def _load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        app.extra = {"model_loaded": False, "error": f"MODEL_PATH not found: {MODEL_PATH}"}
        return
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    app.extra = {"model_loaded": True, "num_labels": len(labels), "input_format": INPUT_FORMAT}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": APP_VER,
        "model_path": MODEL_PATH,
        "labels": len(labels),
        "model_loaded": app.extra.get("model_loaded", False),
        "input_format": app.extra.get("input_format", INPUT_FORMAT),
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or not app.extra.get("model_loaded", False):
        return {"ok": False, "error": f"Model not loaded at {MODEL_PATH}"}
    data = await file.read()
    if not data:
        return {"ok": False, "error": "Empty file"}
    try:
        audio, sr = load_any_audio_bytes(data, filename_hint=file.filename or "upload.bin", ar=SAMPLE_RATE)
        if sr != SAMPLE_RATE:
            # Shouldn’t happen; we force ffmpeg resample, but assert anyway
            return {"ok": False, "error": f"Unexpected sample rate: {sr}"}
        x = prepare_input(audio)  # shape depends on INPUT_FORMAT
        with torch.no_grad():
            logits = model(x)  # expect [1, num_classes]
        top = softmax_topk(logits, TOPK)
        return {"ok": True, "top1": top[0], "topk": top, "filename": file.filename}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

# Raw-bytes fallback for PS5 -InFile
@app.post("/predict-bytes")
async def predict_bytes(request: Request):
    if model is None or not app.extra.get("model_loaded", False):
        return {"ok": False, "error": f"Model not loaded at {MODEL_PATH}"}
    data = await request.body()
    if not data:
        return {"ok": False, "error": "Empty body"}
    try:
        audio, sr = load_any_audio_bytes(data, filename_hint="upload.bin", ar=SAMPLE_RATE)
        x = prepare_input(audio)
        with torch.no_grad():
            logits = model(x)
        top = softmax_topk(logits, TOPK)
        return {"ok": True, "mode": "bytes", "top1": top[0], "topk": top}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
