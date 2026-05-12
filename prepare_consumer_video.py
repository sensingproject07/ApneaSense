"""
Strip audio from a consumer-mode RGB video and replace it with PSG audio clips.

Input:  demo/consumer/consumer_input.mp4
Output: demo/consumer/consumer_demo.mp4  (ready for consumer pipeline)

APNEA_REGIONS defines exact second ranges where apnea audio is spliced in.
Everything outside those ranges uses non-apnea audio.
Audio clips are sourced from the PSG dataset and sliced to the required duration.
A short 50ms crossfade is applied at each splice boundary to avoid clicks.

If the video is shorter than the last apnea region end time, that region is
clipped to fit — no slowdown or speedup needed.
"""

from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import scipy.io.wavfile as wavfile

# ── Paths ──────────────────────────────────────────────────────────────────────

PSG_BASE = Path(
    r"c:\Users\arshi\Desktop\AIS\Sem-2\Intelligent Sensing Systems"
    r"\Practise Module\Project\Dataset\Audio Dataset\PSG-AUDIO\APNEA_EDF"
    r"\00000995-100507"
)
DEMO_DIR   = Path(__file__).parent / "demo" / "consumer"
INPUT_VID  = DEMO_DIR / "consumer_input.mp4"
OUTPUT_VID = DEMO_DIR / "consumer_demo.mp4"

SAMPLE_RATE = 16_000

# ── Apnea regions (start_s, end_s) ────────────────────────────────────────────
# Edit these to match where the subject is supine in the consumer video.

APNEA_REGIONS = [
    (4,  8),    # 4s – 8s   : supine, apnea audio
    (17, 20),   # 17s – 20s : supine again, apnea audio
]

# PSG clip indices (top by RMS energy)
AP_CLIPS  = [68, 104, 54, 76, 90, 132]
NAP_CLIPS = [74,  96, 140, 158, 134, 128]

CROSSFADE_SAMPLES = int(0.05 * SAMPLE_RATE)   # 50 ms crossfade at splice points

# ── Find ffmpeg ────────────────────────────────────────────────────────────────

def _find_ffmpeg() -> str:
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "ffmpeg not found.\n"
            "  Option A: winget install ffmpeg  (then restart terminal)\n"
            "  Option B: .venv\\Scripts\\pip install imageio-ffmpeg"
        )

FFMPEG = _find_ffmpeg()

# ── Get video duration ─────────────────────────────────────────────────────────

if not INPUT_VID.exists():
    raise FileNotFoundError(
        f"Input video not found: {INPUT_VID}\n"
        "Place your consumer RGB video at  demo/consumer/consumer_input.mp4"
    )

probe  = subprocess.run([FFMPEG, "-i", str(INPUT_VID)], capture_output=True)
stderr = probe.stderr.decode(errors="replace")
m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", stderr)
if not m:
    raise RuntimeError("Could not parse video duration from ffmpeg output.")
h, mn, s   = int(m.group(1)), int(m.group(2)), float(m.group(3))
duration_s = h * 3600 + mn * 60 + s
n_samples  = int(duration_s * SAMPLE_RATE)
print(f"Video duration: {duration_s:.2f}s  ({n_samples} samples at {SAMPLE_RATE} Hz)")

# ── Load PSG pools ─────────────────────────────────────────────────────────────

print("Loading PSG audio pools…")
ap_pool  = np.load(PSG_BASE / f"{PSG_BASE.name}_ap.npy").astype(np.float32)
nap_pool = np.load(PSG_BASE / f"{PSG_BASE.name}_nap.npy").astype(np.float32)

def _get_samples(pool, clip_indices, n_needed: int) -> np.ndarray:
    """Concatenate clips from pool until we have at least n_needed samples."""
    out = []
    total = 0
    for idx in clip_indices * 10:          # cycle through indices
        out.append(pool[idx])
        total += len(pool[idx])
        if total >= n_needed:
            break
    return np.concatenate(out)[:n_needed]

# ── Build audio track ──────────────────────────────────────────────────────────

# Start with a full-length non-apnea base track
track = _get_samples(nap_pool, NAP_CLIPS, n_samples)

# Splice apnea audio into each APNEA_REGION
ap_used = 0
for region_idx, (start_s, end_s) in enumerate(APNEA_REGIONS):
    start_samp = int(start_s * SAMPLE_RATE)
    end_samp   = min(int(end_s * SAMPLE_RATE), n_samples)
    if start_samp >= n_samples:
        print(f"  Region {region_idx} ({start_s}s–{end_s}s) is beyond video length — skipping.")
        continue

    region_len = end_samp - start_samp
    ap_chunk   = _get_samples(ap_pool, AP_CLIPS[ap_used:] + AP_CLIPS, region_len)
    ap_used    = (ap_used + 1) % len(AP_CLIPS)

    # Crossfade at start boundary
    cf = min(CROSSFADE_SAMPLES, region_len // 4)
    fade_in  = np.linspace(0, 1, cf)
    fade_out = np.linspace(1, 0, cf)

    blended_start = track[start_samp:start_samp + cf] * fade_out + ap_chunk[:cf] * fade_in
    blended_end   = ap_chunk[region_len - cf:] * fade_out + track[end_samp - cf:end_samp] * fade_in

    track[start_samp:end_samp]               = ap_chunk
    track[start_samp:start_samp + cf]        = blended_start
    track[end_samp - cf:end_samp]            = blended_end

    print(f"  Region {region_idx}: spliced apnea audio {start_s}s – {end_s}s "
          f"({region_len / SAMPLE_RATE:.2f}s)")

# Normalise
max_val = np.abs(track).max()
if max_val > 0:
    track = track / max_val * 0.9
audio_int16 = (track * 32767).astype(np.int16)

# ── Save WAV and mux ───────────────────────────────────────────────────────────

DEMO_DIR.mkdir(parents=True, exist_ok=True)
tmp_wav = str(DEMO_DIR / "_consumer_audio.wav")
wavfile.write(tmp_wav, SAMPLE_RATE, audio_int16)
print(f"Audio track written ({duration_s:.2f}s).")

print("Muxing into output video…")
result = subprocess.run(
    [
        FFMPEG, "-y",
        "-i", str(INPUT_VID),
        "-i", tmp_wav,
        "-map", "0:v",
        "-map", "1:a",
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(OUTPUT_VID),
    ],
    capture_output=True,
)
if result.returncode != 0:
    raise RuntimeError(result.stderr.decode(errors="replace"))

Path(tmp_wav).unlink(missing_ok=True)
print(f"\nDone → {OUTPUT_VID}")
