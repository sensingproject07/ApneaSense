"""
Vision inference: depth synthesis, joint estimation, preprocessing, and
attention_fusion forward pass.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

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
    IMG_W, IMG_H, VISION_INPUT_SIZE, VISION_CLASS_NAMES,
    VISION_MODEL_PATH, MEDIAPIPE_MODEL_PATH,
    YOLO_POSE_MODEL_PATH, YOLO_CONF_THRESHOLD,
    CONSUMER_DEPTH_PREPROCESS, CONSUMER_BODY_PADDING,
    CONSUMER_DEPTH_NORM_PERCENTILES,
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


def joint_vector_to_bbox(
    joint_vec: np.ndarray,
    image_size: tuple[int, int],
    padding: float = 0.15,
) -> Optional[tuple[int, int, int, int]]:
    joints = np.asarray(joint_vec, dtype=np.float32).reshape(14, 3)
    xs = joints[:, 0]
    ys = joints[:, 1]
    occ = joints[:, 2]
    valid = (xs > 0) & (ys > 0) & (xs <= 1.5) & (ys <= 1.5) & (occ < 1.0)
    if valid.sum() < 3:
        valid = (xs > 0) & (ys > 0) & (xs <= 1.5) & (ys <= 1.5)
    if valid.sum() < 3:
        return None

    width, height = image_size
    x_px = xs[valid] * width
    y_px = ys[valid] * height
    x0, x1 = float(x_px.min()), float(x_px.max())
    y0, y1 = float(y_px.min()), float(y_px.max())
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    pad_x = box_w * padding
    pad_y = box_h * padding
    left = max(0, int(np.floor(x0 - pad_x)))
    top = max(0, int(np.floor(y0 - pad_y)))
    right = min(width, int(np.ceil(x1 + pad_x)))
    bottom = min(height, int(np.ceil(y1 + pad_y)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def preprocess_synthetic_depth(
    depth_pil: Image.Image,
    joint_vec: np.ndarray,
    mode: str = CONSUMER_DEPTH_PREPROCESS,
    body_padding: float = CONSUMER_BODY_PADDING,
    norm_percentiles: tuple[float, float] = CONSUMER_DEPTH_NORM_PERCENTILES,
) -> Image.Image:
    """
    Consumer synthetic-depth postprocess used by the fine-tuned checkpoint.
    """
    if mode == "none":
        return depth_pil

    arr = np.asarray(depth_pil.convert("L"), dtype=np.float32) / 255.0
    if mode == "invert":
        return Image.fromarray(((1.0 - arr) * 255).astype(np.uint8), mode="L")

    if mode not in {"body-norm", "body-norm-bg-zero", "body-norm-invert"}:
        raise ValueError(f"Unsupported synthetic-depth preprocess mode: {mode}")

    bbox = joint_vector_to_bbox(joint_vec, depth_pil.size, padding=body_padding)
    if bbox is None:
        ref = arr
        mask = np.ones_like(arr, dtype=bool)
    else:
        left, top, right, bottom = bbox
        mask = np.zeros_like(arr, dtype=bool)
        mask[top:bottom, left:right] = True
        ref = arr[mask]

    lo, hi = np.percentile(ref, norm_percentiles)
    if hi <= lo:
        norm = np.zeros_like(arr)
    else:
        norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

    if mode == "body-norm-bg-zero" and bbox is not None:
        norm = np.where(mask, norm, 0.0)
    elif mode == "body-norm-invert":
        norm = 1.0 - norm

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

# --- YOLO joint estimation (consumer mode) ---
_yolo_cache = {}

# SLP-14 order:
# 0 R ankle, 1 R knee, 2 R hip, 3 L hip, 4 L knee, 5 L ankle,
# 6 R wrist, 7 R elbow, 8 R shoulder, 9 L shoulder, 10 L elbow,
# 11 L wrist, 12 neck/thorax, 13 head.
_COCO17_TO_SLP14 = {
    0: 16,
    1: 14,
    2: 12,
    3: 11,
    4: 13,
    5: 15,
    6: 10,
    7: 8,
    8: 6,
    9: 5,
    10: 7,
    11: 9,
}


def _get_yolo_pose(model_path=None):
    path = str(model_path or YOLO_POSE_MODEL_PATH)
    if path not in _yolo_cache:
        from ultralytics import YOLO
        _yolo_cache[path] = YOLO(path)
    return _yolo_cache[path]


def _select_best_yolo_pose(result) -> Optional[int]:
    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return None
    if boxes is not None and boxes.conf is not None and len(boxes.conf) == len(keypoints.xy):
        return int(torch.argmax(boxes.conf).item())
    return 0


def _slp_joints_to_vector(joints_3x14: np.ndarray) -> np.ndarray:
    x = joints_3x14[0].astype(np.float32)
    y = joints_3x14[1].astype(np.float32)
    occ = joints_3x14[2].astype(np.float32)
    return np.stack([x, y, occ], axis=1).reshape(-1)


def yolo_result_to_slp_vector(
    result,
    image_size: tuple[int, int],
    conf_threshold: float = YOLO_CONF_THRESHOLD,
) -> tuple[np.ndarray, bool]:
    """
    Convert one Ultralytics pose result into the fusion model's 42-D joint vector.
    Coordinates are normalized to the input frame size.
    """
    pose_idx = _select_best_yolo_pose(result)
    if pose_idx is None:
        return np.zeros(42, dtype=np.float32), False

    width, height = image_size
    xy = result.keypoints.xy[pose_idx].detach().cpu().numpy()
    conf = None
    if getattr(result.keypoints, "conf", None) is not None:
        conf = result.keypoints.conf[pose_idx].detach().cpu().numpy()

    joints = np.zeros((3, 14), dtype=np.float32)
    if xy.shape[0] < 17:
        return np.zeros(42, dtype=np.float32), False

    for slp_idx, coco_idx in _COCO17_TO_SLP14.items():
        x_px, y_px = xy[coco_idx]
        score = float(conf[coco_idx]) if conf is not None else 1.0
        if x_px > 0 and y_px > 0:
            joints[0, slp_idx] = x_px / width
            joints[1, slp_idx] = y_px / height
        joints[2, slp_idx] = 0.0 if score >= conf_threshold else 1.0

    left_shoulder = xy[5]
    right_shoulder = xy[6]
    if np.all(left_shoulder > 0) and np.all(right_shoulder > 0):
        joints[0, 12] = (left_shoulder[0] + right_shoulder[0]) / 2 / width
        joints[1, 12] = (left_shoulder[1] + right_shoulder[1]) / 2 / height
    shoulder_conf = 1.0
    if conf is not None:
        shoulder_conf = min(float(conf[5]), float(conf[6]))
    joints[2, 12] = 0.0 if shoulder_conf >= conf_threshold else 1.0

    nose_conf = float(conf[0]) if conf is not None else 1.0
    if xy[0, 0] > 0 and xy[0, 1] > 0:
        joints[0, 13] = xy[0, 0] / width
        joints[1, 13] = xy[0, 1] / height
    joints[2, 13] = 0.0 if nose_conf >= conf_threshold else 1.0

    detected = bool(np.count_nonzero(joints[:2]) > 0)
    return _slp_joints_to_vector(joints), detected


def estimate_joints_yolo(
    rgb_pil: Image.Image,
    model_path=None,
    conf_threshold: float = YOLO_CONF_THRESHOLD,
    device=None,
) -> np.ndarray:
    """
    Run YOLO pose on a PIL RGB image.
    Returns a (42,) float32 vector in normalized SLP-14 order.
    """
    pose_model = _get_yolo_pose(model_path)
    result = pose_model.predict(
        source=np.array(rgb_pil),
        conf=conf_threshold,
        verbose=False,
        device=device,
    )[0]
    joint_vec, _ = yolo_result_to_slp_vector(result, rgb_pil.size, conf_threshold)
    return joint_vec


# --- GT joint loading (clinical mode) ---

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
