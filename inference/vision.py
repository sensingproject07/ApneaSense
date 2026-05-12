"""
Vision inference: depth synthesis, joint estimation, preprocessing, and
attention_fusion forward pass.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .config import (
    IMG_W, IMG_H, VISION_INPUT_SIZE, VISION_CLASS_NAMES,
    VISION_MODEL_PATH, MEDIAPIPE_MODEL_PATH,
)
from .models import DepthEncoder, RGBEncoder, JointEncoder, AttentionFusionClassifier


# ── Transforms (no ImageNet normalisation — matches training) ──────────────────
_rgb_tf   = T.Compose([T.Resize(VISION_INPUT_SIZE), T.ToTensor()])
_depth_tf = T.Compose([T.Resize(VISION_INPUT_SIZE), T.ToTensor()])


# ── Vision model ───────────────────────────────────────────────────────────────

@st.cache_resource
def load_vision_model(checkpoint_path=None, device=None):
    checkpoint_path = Path(checkpoint_path or VISION_MODEL_PATH)
    device = device or torch.device("cpu")
    model = AttentionFusionClassifier(
        depth_encoder=DepthEncoder(),
        rgb_encoder=RGBEncoder(),
        joint_encoder=JointEncoder(input_dim=42, hidden_dim=128, feature_dim=128, dropout=0.3),
        depth_feature_dim=512, rgb_feature_dim=128, joint_feature_dim=128,
        common_feature_dim=256, num_heads=4, ff_dim=512, dropout=0.1, num_classes=3,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Synthetic depth (Depth Anything v2 Small) ──────────────────────────────────
_depth_anything = {}


def _get_depth_anything(device):
    if "model" not in _depth_anything:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        name = "depth-anything/Depth-Anything-V2-Small-hf"
        _depth_anything["processor"] = AutoImageProcessor.from_pretrained(name)
        _depth_anything["model"]     = (
            AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()
        )
    return _depth_anything["processor"], _depth_anything["model"]


@torch.no_grad()
def synthesize_depth(rgb_pil: Image.Image, device, target_size=(424, 512)) -> Image.Image:
    """
    Monocular depth from Depth Anything v2 Small.
    Returns a PIL grayscale image (mode L, uint8, closer=brighter) at target_size.
    """
    processor, da_model = _get_depth_anything(device)
    inputs = processor(images=rgb_pil, return_tensors="pt").to(device)
    pred   = da_model(**inputs).predicted_depth        # inverse depth, close=large
    pred   = F.interpolate(
        pred.unsqueeze(1),
        size=(target_size[1], target_size[0]),         # (H, W)
        mode="bicubic", align_corners=False,
    ).squeeze().cpu().numpy()
    p_min, p_max = pred.min(), pred.max()
    norm = (pred - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(pred)
    return Image.fromarray((norm * 255).astype(np.uint8), mode="L")


# ── MediaPipe joint estimation ─────────────────────────────────────────────────
_mp_cache = {}

# SLP LSP-14 index → MediaPipe landmark index (direct mappings)
_SLP_TO_MP = {0: 28, 1: 26, 2: 24, 3: 23, 4: 25, 5: 27,
              6: 16, 7: 14, 8: 12, 9: 11, 10: 13, 11: 15}
_VIS_THRESH = 0.5


def _get_mediapipe(model_path=None):
    if "detector" not in _mp_cache:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        path = str(model_path or MEDIAPIPE_MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        _mp_cache["detector"] = mp_vision.PoseLandmarker.create_from_options(options)
        _mp_cache["mp"]       = mp
    return _mp_cache["detector"], _mp_cache["mp"]


def estimate_joints_mediapipe(rgb_pil: Image.Image, model_path=None) -> np.ndarray:
    """
    Run MediaPipe PoseLandmarker on a PIL RGB image.
    Returns a (42,) float32 vector in SLP normalisation (x/IMG_W, y/IMG_H, occ).
    Returns zeros when no pose is detected.
    """
    detector, mp = _get_mediapipe(model_path)
    rgb_np   = np.array(rgb_pil)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)
    result   = detector.detect(mp_image)

    joints = np.zeros((3, 14), dtype=np.float32)
    if not result.pose_landmarks:
        return joints.T.reshape(-1)           # 42 zeros

    lm = result.pose_landmarks[0]
    for slp_idx, mp_idx in _SLP_TO_MP.items():
        l = lm[mp_idx]
        joints[0, slp_idx] = l.x
        joints[1, slp_idx] = l.y
        joints[2, slp_idx] = 0.0 if l.visibility > _VIS_THRESH else 1.0

    # Joint 12 — Neck: midpoint of shoulders
    ls, rs = lm[11], lm[12]
    joints[0, 12] = (ls.x + rs.x) / 2
    joints[1, 12] = (ls.y + rs.y) / 2
    joints[2, 12] = 0.0 if (ls.visibility + rs.visibility) / 2 > _VIS_THRESH else 1.0

    # Joint 13 — Head: nose
    nose = lm[0]
    joints[0, 13] = nose.x
    joints[1, 13] = nose.y
    joints[2, 13] = 0.0 if nose.visibility > _VIS_THRESH else 1.0

    # Stack as (14, 3) then flatten — matches preprocess_joint_frame_xyo convention
    return np.stack([joints[0], joints[1], joints[2]], axis=1).reshape(-1)


# ── GT joint loading (clinical mode) ──────────────────────────────────────────

def load_joints_from_mat(mat_path, frame_idx: int) -> np.ndarray:
    """
    Load one frame's joints from an SLP .mat file.
    Returns (42,) float32 in normalised (x/576, y/1024, occ) format.
    """
    import scipy.io as sio
    mat   = sio.loadmat(mat_path)["joints_gt"]    # (3, 14, N_frames)
    frame = mat[:, :, frame_idx]                   # (3, 14)
    x   = (frame[0] / IMG_W).astype(np.float32)
    y   = (frame[1] / IMG_H).astype(np.float32)
    occ = frame[2].astype(np.float32)
    return np.stack([x, y, occ], axis=1).reshape(-1)


def load_joints_from_csv(csv_path, frame_idx: int) -> np.ndarray:
    """
    Load one frame's joints from a CSV file.
    Expected columns: frame_idx, j0_x, j0_y, j0_occ, ..., j13_x, j13_y, j13_occ.
    Returns (42,) float32 or zeros if frame not found.
    """
    import pandas as pd
    df  = pd.read_csv(csv_path)
    row = df[df["frame_idx"] == frame_idx]
    if len(row) == 0:
        return np.zeros(42, dtype=np.float32)
    return row.iloc[0, 1:43].values.astype(np.float32)


# ── Tensor preprocessing ───────────────────────────────────────────────────────

def preprocess_rgb(rgb_pil: Image.Image) -> torch.Tensor:
    """PIL RGB → (1, 3, 224, 224) float32 tensor in [0, 1]."""
    return _rgb_tf(rgb_pil).unsqueeze(0)


def preprocess_depth(depth_pil: Image.Image) -> torch.Tensor:
    """PIL grayscale depth → (1, 1, 224, 224) float32 tensor in [0, 1]."""
    return _depth_tf(depth_pil.convert("L")).unsqueeze(0)


def preprocess_joints(joint_vec: np.ndarray) -> torch.Tensor:
    """(42,) float32 numpy → (1, 42) float32 tensor."""
    return torch.tensor(joint_vec, dtype=torch.float32).unsqueeze(0)


# ── Per-frame inference ────────────────────────────────────────────────────────

@torch.no_grad()
def run_frame_inference(model, depth_t, rgb_t, joint_t, device) -> dict:
    """
    Forward pass through AttentionFusionClassifier.
    Returns {posture_label, posture_conf, probs}.
    """
    logits = model(depth_t.to(device), rgb_t.to(device), joint_t.to(device))
    probs  = torch.softmax(logits, dim=1)[0].cpu()
    idx    = int(probs.argmax())
    return {
        "posture_label": VISION_CLASS_NAMES[idx],
        "posture_conf":  float(probs[idx]),
        "probs":         probs.tolist(),
    }
