"""
High-level inference pipelines.

consumer_pipeline  — RGB+audio video only; generates synthetic depth + MediaPipe joints.
clinical_pipeline  — RGB video + real depth folder/video + real joints (.mat or CSV).
"""
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from .config import (
    AUDIO_MODEL_PATH,
    VISION_MODEL_PATH,
    CONSUMER_VISION_MODEL_PATH,
    WINDOW_DURATION,
)
from .audio import extract_audio, segment_waveform, load_audio_model, run_audio_inference
from .vision import (
    load_vision_model,
    synthesize_depth, preprocess_synthetic_depth, estimate_joints_yolo,
    load_joints_from_mat, load_joints_from_csv,
    preprocess_rgb, preprocess_depth, preprocess_joints,
    run_frame_inference,
)
from .fusion import aggregate_posture_per_window, apply_fusion


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _extract_frames(video_path):
    """Extract all RGB frames from a video with OpenCV.
    Returns (frames, fps) where frames is a list of
    {frame_idx, timestamp_s, image (PIL.Image RGB)}.
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append({
            "frame_idx":   idx,
            "timestamp_s": idx / fps,
            "image":       Image.fromarray(rgb),
        })
        idx += 1
    cap.release()
    return frames, fps


def _load_depth_video(depth_video_path):
    """Pre-load all frames from a depth video into a dict keyed by frame_idx."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(depth_video_path))
    cache = {}
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cache[idx] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        idx += 1
    cap.release()
    return cache


# ── Consumer pipeline ──────────────────────────────────────────────────────────

def consumer_pipeline(
    video_path: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Consumer mode: RGB+audio video only.

    Processing steps:
      1. Extract audio → 10-second windows → mel_cnn apnea probabilities.
      2. Extract RGB frames → per-frame:
           a. Synthesize depth (Depth Anything v2 Small).
           b. Estimate joints (MediaPipe PoseLandmarker heavy).
           c. Run attention_fusion → posture label + confidence.
      3. Aggregate posture per audio window.
      4. Posture-conditioned threshold + hysteresis → verdict per window.

    Args:
        video_path  : path to the input video file (MP4, AVI, etc.)
        progress_cb : optional callback(str) for progress messages

    Returns:
        dict with keys:
            mode, fps, n_frames, n_windows,
            frame_results, audio_results, window_postures, verdicts
    """
    def _log(msg):
        if progress_cb:
            progress_cb(msg)

    device = _get_device()
    video_path = Path(video_path)

    _log("Loading models...")
    audio_model  = load_audio_model(AUDIO_MODEL_PATH, device)
    vision_model = load_vision_model(CONSUMER_VISION_MODEL_PATH, device)

    _log("Extracting audio from video...")
    waveform     = extract_audio(video_path)
    windows      = segment_waveform(waveform)
    _log(f"  {len(windows)} audio windows ({WINDOW_DURATION}s each)")

    _log("Running audio inference...")
    audio_results = run_audio_inference(audio_model, windows, device)

    _log("Extracting video frames...")
    frames, fps = _extract_frames(video_path)
    _log(f"  {len(frames)} frames at {fps:.1f} fps")

    yolo_device = 0 if device.type == "cuda" else "cpu"
    _log("Running vision inference (synthetic depth + YOLO joints)...")
    frame_results = []
    for frame in frames:
        i = frame["frame_idx"]
        if i % 30 == 0:
            _log(f"  Frame {i + 1}/{len(frames)}")
        joint_vec = estimate_joints_yolo(frame["image"], device=yolo_device)
        depth_pil = synthesize_depth(frame["image"], device)
        depth_pil = preprocess_synthetic_depth(depth_pil, joint_vec)
        result    = run_frame_inference(
            vision_model,
            preprocess_depth(depth_pil),
            preprocess_rgb(frame["image"]),
            preprocess_joints(joint_vec),
            device,
        )
        frame_results.append({
            "frame_idx":   i,
            "timestamp_s": frame["timestamp_s"],
            **result,
        })

    _log("Fusing audio and vision decisions...")
    window_postures = aggregate_posture_per_window(frame_results, fps)
    verdicts        = apply_fusion(audio_results, window_postures)

    return {
        "mode":             "consumer",
        "fps":              fps,
        "n_frames":         len(frames),
        "n_windows":        len(windows),
        "frame_results":    frame_results,
        "audio_results":    audio_results,
        "window_postures":  window_postures,
        "verdicts":         verdicts,
    }


# ── Clinical pipeline ──────────────────────────────────────────────────────────

def clinical_pipeline(
    video_path: str,
    depth_source: str,
    joints_source: str,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    """
    Clinical mode: RGB video + real depth + real joints.

    Args:
        video_path    : RGB video (MP4, AVI, etc.)
        depth_source  : folder of depth PNGs named image_XXXXXX.png (1-indexed)
                        OR path to a depth video file.
        joints_source : SLP .mat file (joints_gt_RGB.mat)
                        OR CSV with columns frame_idx, j0_x, j0_y, j0_occ, ...
        progress_cb   : optional callback(str) for progress messages

    Returns:
        Same dict structure as consumer_pipeline, with mode='clinical'.
    """
    from PIL import Image

    def _log(msg):
        if progress_cb:
            progress_cb(msg)

    device        = _get_device()
    video_path    = Path(video_path)
    depth_source  = Path(depth_source)
    joints_source = Path(joints_source)

    _log("Loading models...")
    audio_model  = load_audio_model(AUDIO_MODEL_PATH, device)
    vision_model = load_vision_model(VISION_MODEL_PATH, device)

    _log("Extracting audio from video...")
    waveform  = extract_audio(video_path)
    windows   = segment_waveform(waveform)
    _log(f"  {len(windows)} audio windows ({WINDOW_DURATION}s each)")

    _log("Running audio inference...")
    audio_results = run_audio_inference(audio_model, windows, device)

    _log("Extracting video frames...")
    frames, fps = _extract_frames(video_path)
    _log(f"  {len(frames)} frames at {fps:.1f} fps")

    # ── Depth loader ───────────────────────────────────────────────────────────
    if depth_source.is_dir():
        def _load_depth(frame_idx: int) -> Image.Image:
            # SLP convention: filenames are 1-indexed
            p = depth_source / f"image_{frame_idx + 1:06d}.png"
            if not p.exists():
                return Image.new("L", (424, 512))
            return Image.open(p).convert("L")
    else:
        # Depth video — pre-load all frames
        _log("Pre-loading depth video frames...")
        depth_cache = _load_depth_video(depth_source)
        def _load_depth(frame_idx: int) -> Image.Image:
            return depth_cache.get(frame_idx, Image.new("L", (424, 512)))

    # ── Joints loader ──────────────────────────────────────────────────────────
    if joints_source.suffix == ".mat":
        import scipy.io as sio
        from .config import IMG_W, IMG_H
        mat_data = sio.loadmat(joints_source)["joints_gt"]   # (3, 14, N)
        def _load_joints(frame_idx: int) -> np.ndarray:
            if frame_idx >= mat_data.shape[2]:
                return np.zeros(42, dtype=np.float32)
            frame = mat_data[:, :, frame_idx]
            x   = (frame[0] / IMG_W).astype(np.float32)
            y   = (frame[1] / IMG_H).astype(np.float32)
            occ = frame[2].astype(np.float32)
            return np.stack([x, y, occ], axis=1).reshape(-1)
    else:
        import pandas as pd
        _joints_df = pd.read_csv(joints_source).set_index("frame_idx")
        def _load_joints(frame_idx: int) -> np.ndarray:
            if frame_idx not in _joints_df.index:
                return np.zeros(42, dtype=np.float32)
            return _joints_df.loc[frame_idx].values[:42].astype(np.float32)

    _log("Running vision inference (real depth + real joints)...")
    frame_results = []
    for frame in frames:
        i = frame["frame_idx"]
        if i % 30 == 0:
            _log(f"  Frame {i + 1}/{len(frames)}")
        depth_pil = _load_depth(i)
        joint_vec = _load_joints(i)
        result    = run_frame_inference(
            vision_model,
            preprocess_depth(depth_pil),
            preprocess_rgb(frame["image"]),
            preprocess_joints(joint_vec),
            device,
        )
        frame_results.append({
            "frame_idx":   i,
            "timestamp_s": frame["timestamp_s"],
            **result,
        })

    _log("Fusing audio and vision decisions...")
    window_postures = aggregate_posture_per_window(frame_results, fps)
    verdicts        = apply_fusion(audio_results, window_postures)

    return {
        "mode":             "clinical",
        "fps":              fps,
        "n_frames":         len(frames),
        "n_windows":        len(windows),
        "frame_results":    frame_results,
        "audio_results":    audio_results,
        "window_postures":  window_postures,
        "verdicts":         verdicts,
    }
