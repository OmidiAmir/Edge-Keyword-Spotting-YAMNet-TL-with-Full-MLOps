from fastapi import FastAPI, UploadFile, File, Request
import tempfile, os, subprocess, shutil
import torch
import soundfile as sf  # <-- read decoded WAV via libsndfile

APP_VER = "v4-soundfile"

app = FastAPI()

def ffmpeg_to_wav(in_path: str, out_path: str, ar: int = 16000):
    cmd = [
        "ffmpeg","-hide_banner","-loglevel","error","-y",
        "-i", in_path, "-f","wav","-acodec","pcm_s16le","-ac","1","-ar", str(ar),
        out_path
    ]
    subprocess.run(cmd, check=True)

def load_any_audio_bytes(data: bytes, filename_hint: str = "upload.bin", ar: int = 16000):
    suffix = os.path.splitext(filename_hint)[1] or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src:
        src.write(data); src_path = src.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as dst:
        dst_path = dst.name
    try:
        ffmpeg_to_wav(src_path, dst_path, ar=ar)
        audio, sr = sf.read(dst_path, always_2d=False)  # numpy array
        return audio, sr
    finally:
        for p in (src_path, dst_path):
            try: os.unlink(p)
            except: pass

@app.get("/health")
def health():
    return {"status":"ok","version":APP_VER,"ffmpeg": shutil.which("ffmpeg") is not None}

@app.get("/selftest")
def selftest():
    # simple in-memory tone (no file IO)
    sr = 16000
    t = torch.arange(int(0.5*sr)) / sr
    tone = torch.sin(2*torch.pi*440*t)
    return {"ok": True, "sr": sr, "samples": int(tone.numel())}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return {"ok": False, "error": "Empty file"}
    try:
        audio, sr = load_any_audio_bytes(data, filename_hint=file.filename or "upload.bin")
    except Exception as e:
        return {"ok": False, "error": f"Audio read failed: {e}"}
    if audio.ndim > 1:
        import numpy as np
        audio = audio.mean(axis=1)
    dur = float(len(audio)) / float(sr)
    return {"ok": True, "filename": file.filename, "sr": int(sr), "duration_sec": round(dur, 3), "label": "yes"}

@app.post("/predict-bytes")
async def predict_bytes(request: Request):
    data = await request.body()
    if not data:
        return {"ok": False, "error": "Empty body"}
    try:
        audio, sr = load_any_audio_bytes(data, filename_hint="upload.bin")
    except Exception as e:
        return {"ok": False, "error": f"Audio read failed: {e}"}
    if audio.ndim > 1:
        import numpy as np
        audio = audio.mean(axis=1)
    dur = float(len(audio)) / float(sr)
    return {"ok": True, "mode": "bytes", "sr": int(sr), "duration_sec": round(dur, 3), "label": "yes"}
