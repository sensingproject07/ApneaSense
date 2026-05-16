"""
Audio extraction, mel-spectrogram preprocessing, and mel_cnn inference.
"""
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as AT

try:
    import streamlit as st
except ImportError:
    class _StreamlitFallback:
        @staticmethod
        def cache_resource(func=None, **_kwargs):
            if func is None:
                return lambda wrapped: wrapped
            return func

    st = _StreamlitFallback()

from .config import (
    SAMPLE_RATE, WINDOW_DURATION, SAMPLES_PER_WINDOW,
    N_MELS, N_FFT, HOP_LENGTH, F_MIN, F_MAX,
    DB_FLOOR, DB_CEIL, POWER_FLOOR, MEL_TIME_FRAMES,
    AUDIO_MODEL_PATH, AUDIO_CLASS_NAMES,
)
from .models import MelCNN


# ── Audio extraction ───────────────────────────────────────────────────────────

def _find_ffmpeg() -> str:
    import shutil
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found. Install it with:\n"
            "  .venv\\Scripts\\pip install imageio-ffmpeg"
        )


def extract_audio(video_path: str | Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Extract mono audio from a video file via ffmpeg.
    Returns a (N_samples,) float32 tensor at target_sr.
    """
    video_path = Path(video_path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [
                _find_ffmpeg(), "-y",
                "-i", str(video_path),
                "-ar", str(target_sr),
                "-ac", "1",           # mono
                "-vn",                # drop video stream
                str(tmp_path),
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed (exit {result.returncode}):\n"
                + result.stderr.decode(errors="replace")
            )
        import scipy.io.wavfile as wavfile
        sr, data = wavfile.read(str(tmp_path))
        waveform = torch.tensor(data, dtype=torch.float32)
        if waveform.ndim == 2:          # stereo → mono
            waveform = waveform.mean(dim=1)
        waveform = waveform / (32768.0 if data.dtype == np.int16 else 1.0)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        return waveform                 # (N_samples,)
    finally:
        tmp_path.unlink(missing_ok=True)


def segment_waveform(
    waveform: torch.Tensor,
    sr: int = SAMPLE_RATE,
    duration: int = WINDOW_DURATION,
) -> List[Dict]:
    """
    Split waveform into non-overlapping fixed-length windows.
    The last partial window is dropped.
    Returns a list of dicts: {window_idx, start_s, end_s, waveform}.
    """
    n_samples = sr * duration
    n_windows = len(waveform) // n_samples
    return [
        {
            "window_idx": i,
            "start_s":    float(i * duration),
            "end_s":      float((i + 1) * duration),
            "waveform":   waveform[i * n_samples : (i + 1) * n_samples],
        }
        for i in range(n_windows)
    ]


# ── Mel preprocessing ──────────────────────────────────────────────────────────

_mel_transform = AT.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    f_min=F_MIN,
    f_max=F_MAX,
)
_amplitude_to_db = AT.AmplitudeToDB(stype="power", top_db=None)


def waveform_to_mel_tensor(waveform: torch.Tensor) -> torch.Tensor:
    """
    Apply mel spectrogram + dB normalisation to match training preprocessing.
    Returns (1, 1, 128, MEL_TIME_FRAMES) float32, padded/cropped as needed.
    """
    mel   = _mel_transform(waveform)                          # (128, T)
    mel   = mel.clamp(min=POWER_FLOOR)
    db    = _amplitude_to_db(mel)                             # (128, T) in dB
    db    = db.clamp(min=DB_FLOOR, max=DB_CEIL)
    norm  = (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)           # → [0, 1]

    # Pad or crop time axis to the fixed width the model expects
    T = norm.shape[-1]
    if T < MEL_TIME_FRAMES:
        norm = torch.nn.functional.pad(norm, (0, MEL_TIME_FRAMES - T))
    elif T > MEL_TIME_FRAMES:
        norm = norm[..., :MEL_TIME_FRAMES]

    return norm.unsqueeze(0).unsqueeze(0)                     # (1, 1, 128, 313)


# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_audio_model(checkpoint_path=None, device=None):
    checkpoint_path = Path(checkpoint_path or AUDIO_MODEL_PATH)
    device = device or torch.device("cpu")
    model = MelCNN().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_audio_inference(model, windows: List[Dict], device) -> List[Dict]:
    """
    Run mel_cnn on each 10-second window.
    Returns a list of dicts: {window_idx, start_s, end_s, apnea_prob, apnea_pred}.
    apnea_pred uses the raw 0.5 threshold — fusion applies the posture-conditioned
    threshold on top of apnea_prob later.
    """
    results = []
    for w in windows:
        tensor = waveform_to_mel_tensor(w["waveform"]).to(device)
        logits = model(tensor)                                # (1, 2)
        probs  = torch.softmax(logits, dim=1)[0]
        apnea_prob = float(probs[1])
        results.append({
            "window_idx": w["window_idx"],
            "start_s":    w["start_s"],
            "end_s":      w["end_s"],
            "apnea_prob": apnea_prob,
            "apnea_pred": int(apnea_prob >= 0.5),
        })
    return results
