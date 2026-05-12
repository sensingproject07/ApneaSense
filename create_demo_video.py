"""
Create ApneaSense demo videos from SimLab frames + PSG audio.

Outputs (in demo/clinical/):
    demo_rgb.mp4   — RGB frames with audio
    demo_depth.mp4 — depth frames with audio

Frame ordering rule (from danaLab posture_labels_all_modalities.csv):
    image_000001 – image_000015  → supine
    image_000016 – image_000030  → left
    image_000031 – image_000045  → right

Scene plan  (6 windows × 10 s = 60 s total):
    Win 0  frame 20  left    nap[74]   → LOW       (posture gate: left)
    Win 1  frame  8  supine  nap[140]  → LOW       (supine, normal breathing)
    Win 2  frame 10  supine  ap[68]    → HIGH      (wins 2+3 consecutive → confirmed)
    Win 3  frame 12  supine  ap[104]   → HIGH      (wins 2+3 consecutive → confirmed)
    Win 4  frame 35  right   nap[158]  → LOW       (posture gate: right)
    Win 5  frame  9  supine  ap[90]    → suspected (single positive → not confirmed)

Audio clips chosen by highest RMS energy from PSG subject 00000995-100507:
    ap  pool: clips [68, 104, 90] — loud, clearly abnormal breathing
    nap pool: clips [74, 140, 158] — quiet, regular breathing
"""

from pathlib import Path
import numpy as np
import cv2
import scipy.io.wavfile as wavfile

# ── Paths ──────────────────────────────────────────────────────────────────────

SIMLAB_BASE = Path(
    r"c:\Users\arshi\Desktop\AIS\Sem-2\Intelligent Sensing Systems"
    r"\Practise Module\Project\Dataset\SLP2022\SLP\simLab"
)
PSG_BASE = Path(
    r"c:\Users\arshi\Desktop\AIS\Sem-2\Intelligent Sensing Systems"
    r"\Practise Module\Project\Dataset\Audio Dataset\PSG-AUDIO\APNEA_EDF"
    r"\00000995-100507"
)
OUT_DIR = Path(__file__).parent / "demo" / "clinical"

# ── Constants ──────────────────────────────────────────────────────────────────

SUBJECT    = "00001"
CONDITION  = "cover1"
SAMPLE_RATE = 16_000        # PSG audio: 160 000 samples / 16 000 Hz = 10 s
FPS        = 25
WIN_SECS   = 10
FRAMES_PER_WIN = FPS * WIN_SECS   # 250 frames per video window

# (frame_1based, audio_type, clip_idx, note)
SCENES = [
    (20, "nap",  74, "left   + non-apnea -> LOW      (posture gate)"),
    ( 8, "nap", 140, "supine + non-apnea -> LOW      (normal breathing)"),
    (10, "ap",   68, "supine + apnea     -> HIGH     (confirmed: wins 2+3)"),
    (12, "ap",  104, "supine + apnea     -> HIGH     (confirmed: wins 2+3)"),
    (35, "nap", 158, "right  + non-apnea -> LOW      (posture gate)"),
    ( 9, "ap",   90, "supine + apnea     -> suspected (single positive)"),
]

# ── Load audio clips ───────────────────────────────────────────────────────────

print("Loading PSG audio clips...")
ap_clips  = np.load(PSG_BASE / f"{PSG_BASE.name}_ap.npy").astype(np.float32)
nap_clips = np.load(PSG_BASE / f"{PSG_BASE.name}_nap.npy").astype(np.float32)

def _get_clip(audio_type: str, idx: int) -> np.ndarray:
    pool = ap_clips if audio_type == "ap" else nap_clips
    return pool[idx]   # (160 000,) float32

# ── Build audio track ──────────────────────────────────────────────────────────

audio_segments = [_get_clip(atype, cidx) for (_, atype, cidx, _) in SCENES]
audio_track    = np.concatenate(audio_segments)   # (N * 160000,)

# Normalise to int16 for WAV
max_val = np.abs(audio_track).max()
if max_val > 0:
    audio_track = audio_track / max_val * 0.9
audio_int16 = (audio_track * 32767).astype(np.int16)

# ── Helper: load SimLab frame ──────────────────────────────────────────────────

# Portrait dimensions — SLP images are portrait (896×1600 RGB, 424×512 depth).
# Landscape letterboxing shrinks the person to ~32% of frame width, which destroys accuracy.
# 432×768 (portrait) fills ~99% of frame for RGB and ~68% for depth — matches training distribution.
TARGET_W, TARGET_H = 432, 768


def _letterbox(img: np.ndarray) -> np.ndarray:
    """Scale img to fit inside TARGET_W×TARGET_H, pad remainder with black."""
    h, w = img.shape[:2]
    scale   = min(TARGET_W / w, TARGET_H / h)
    new_w   = int(w * scale)
    new_h   = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas  = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
    x = (TARGET_W - new_w) // 2
    y = (TARGET_H - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


def load_rgb(frame_1based: int) -> np.ndarray:
    path = SIMLAB_BASE / SUBJECT / "RGB" / CONDITION / f"image_{frame_1based:06d}.png"
    img  = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return _letterbox(img)


def load_depth(frame_1based: int) -> np.ndarray:
    path = SIMLAB_BASE / SUBJECT / "depth" / CONDITION / f"image_{frame_1based:06d}.png"
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(path)
    # Store as BGR grayscale so pipeline can recover values with COLOR_BGR2GRAY
    return _letterbox(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))


# ── Write videos ───────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

rgb_tmp   = str(OUT_DIR / "_rgb_silent.mp4")
depth_tmp = str(OUT_DIR / "_depth_silent.mp4")
audio_wav = str(OUT_DIR / "_audio.wav")
rgb_out   = str(OUT_DIR / "demo_rgb.mp4")
depth_out = str(OUT_DIR / "demo_depth.mp4")

# ── Find ffmpeg (system PATH or bundled via imageio-ffmpeg) ────────────────────

import shutil, subprocess

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
            "  Option A (system-wide): winget install ffmpeg  then restart terminal\n"
            "  Option B (Python-only): .venv\\Scripts\\pip install imageio-ffmpeg"
        )

FFMPEG = _find_ffmpeg()
print(f"Using ffmpeg: {FFMPEG}")

# ── Write silent videos (skip if temp files already exist from a previous run) ─

temps_exist = Path(rgb_tmp).exists() and Path(depth_tmp).exists() and Path(audio_wav).exists()

if temps_exist:
    print("Temp files found from previous run — skipping video/audio generation.")
else:
    # Probe frame sizes
    sample_rgb   = load_rgb(SCENES[0][0])
    sample_depth = load_depth(SCENES[0][0])
    rgb_h, rgb_w = sample_rgb.shape[:2]
    dep_h, dep_w = sample_depth.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_rgb   = cv2.VideoWriter(rgb_tmp,   fourcc, FPS, (rgb_w, rgb_h))
    vw_depth = cv2.VideoWriter(depth_tmp, fourcc, FPS, (dep_w, dep_h))

    # Load joints mat for joint CSV generation (IMG_W/H from SLP convention)
    import scipy.io as sio
    IMG_W, IMG_H = 576.0, 1024.0
    mat_data = sio.loadmat(
        str(SIMLAB_BASE / SUBJECT / "joints_gt_RGB.mat")
    )["joints_gt"]   # (3, 14, 45)

    joint_rows = []   # (video_frame_idx, j0_x, j0_y, j0_occ, ..., j13_x, j13_y, j13_occ)

    print(f"Writing {len(SCENES)} scenes x {WIN_SECS}s = {len(SCENES)*WIN_SECS}s of video...")
    video_frame_idx = 0
    for scene_idx, (frame_1b, atype, cidx, note) in enumerate(SCENES):
        note_ascii = note.replace("→", "->")
        print(f"  Scene {scene_idx}: {note_ascii}")
        rgb_frame   = load_rgb(frame_1b)
        depth_frame = load_depth(frame_1b)

        # Joint vector for this SimLab frame (0-indexed)
        slp_idx = frame_1b - 1   # mat is 0-indexed
        if slp_idx < mat_data.shape[2]:
            jf = mat_data[:, :, slp_idx]
            jx  = (jf[0] / IMG_W).tolist()
            jy  = (jf[1] / IMG_H).tolist()
            joc = jf[2].tolist()
            joint_vec = []
            for j in range(14):
                joint_vec += [jx[j], jy[j], joc[j]]
        else:
            joint_vec = [0.0] * 42

        for _ in range(FRAMES_PER_WIN):
            vw_rgb.write(rgb_frame)
            vw_depth.write(depth_frame)
            joint_rows.append([video_frame_idx] + joint_vec)
            video_frame_idx += 1

    vw_rgb.release()
    vw_depth.release()
    print("Silent video files written.")

    # Write joints CSV
    import csv
    joints_csv = str(OUT_DIR / "joints_demo.csv")
    header = ["frame_idx"] + [
        f"j{j}_{c}" for j in range(14) for c in ("x", "y", "occ")
    ]
    with open(joints_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(joint_rows)
    print(f"Joints CSV written -> {joints_csv}")

    wavfile.write(audio_wav, SAMPLE_RATE, audio_int16)
    print(f"Audio WAV written  ({len(audio_int16)/SAMPLE_RATE:.1f}s).")

# ── Mux audio into both videos ─────────────────────────────────────────────────

def _mux(video_in: str, audio_in: str, video_out: str):
    result = subprocess.run(
        [
            FFMPEG, "-y",
            "-i", video_in,
            "-i", audio_in,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-shortest",
            video_out,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace"))

print("Muxing audio into RGB video...")
_mux(rgb_tmp, audio_wav, rgb_out)

print("Muxing audio into depth video...")
_mux(depth_tmp, audio_wav, depth_out)

# Clean up temp files
for p in [rgb_tmp, depth_tmp, audio_wav]:
    Path(p).unlink(missing_ok=True)

print(f"\nDone.\n  RGB   -> {rgb_out}\n  Depth -> {depth_out}")
