"""
Evaluate YOLO pose joints as a drop-in joint source for attention_fusion.

This experiment tests YOLO pose joints as a drop-in joint source for
attention_fusion. By default it isolates joint quality by keeping RGB and real
SLP depth fixed:

    Run A: real depth + ground-truth SLP joints
    Run B: real depth + YOLO-estimated joints

Use `--depth-source synthetic` to run the deployment-style variant:

    Run A: synthetic depth + ground-truth SLP joints
    Run B: synthetic depth + YOLO-estimated joints

Use `--depth-source zero` as a no-depth-signal ablation for the existing
fusion model:

    Run A: zero depth + ground-truth SLP joints
    Run B: zero depth + YOLO-estimated joints

Use `--depth-preprocess` with synthetic depth for cheap deployment-side
normalization ablations:

    none                 : raw cached monocular depth
    invert               : reverse depth polarity
    body-norm            : normalize using only the body/joint bounding box
    body-norm-bg-zero    : body-box normalization plus zero background
    body-norm-invert     : body-box normalization plus reversed polarity

If Run B stays close to Run A, YOLO is a viable replacement for MediaPipe/GT
joints. If it falls sharply, the joint estimator is not good enough for the
current fusion model without retraining or adaptation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = next(
    (p for p in [_THIS_FILE.parent, *_THIS_FILE.parents] if (p / "requirements.txt").exists()),
    Path.cwd().resolve(),
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MODELS_SPEC = importlib.util.spec_from_file_location(
    "apneasense_inference_models", _PROJECT_ROOT / "inference" / "models.py"
)
if _MODELS_SPEC is None or _MODELS_SPEC.loader is None:
    raise RuntimeError("Could not load inference/models.py")
_MODELS = importlib.util.module_from_spec(_MODELS_SPEC)
_MODELS_SPEC.loader.exec_module(_MODELS)

AttentionFusionClassifier = _MODELS.AttentionFusionClassifier
DepthEncoder = _MODELS.DepthEncoder
JointEncoder = _MODELS.JointEncoder
RGBEncoder = _MODELS.RGBEncoder


IMG_W = 576.0
IMG_H = 1024.0
CLASS_NAMES = ["supine", "left", "right"]
NUM_CLASSES = 3
SEED = 42

# SLP-14 order:
# 0 R ankle, 1 R knee, 2 R hip, 3 L hip, 4 L knee, 5 L ankle,
# 6 R wrist, 7 R elbow, 8 R shoulder, 9 L shoulder, 10 L elbow,
# 11 L wrist, 12 neck/thorax, 13 head.
COCO17_TO_SLP14 = {
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


class JointMLP(nn.Module):
    def __init__(self, input_dim: int = 42, hidden_dim: int = 128, feature_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    return next((p for p in [cwd, *cwd.parents] if (p / "requirements.txt").exists()), cwd)


def default_checkpoint(project_root: Path) -> Path:
    candidates = [
        project_root / "models" / "vision-spatial" / "attention_fusion.pth",
        project_root
        / "experiments"
        / "artifacts"
        / "vision-spatial"
        / "fusion"
        / "attention_fusion"
        / "checkpoints"
        / "best_attention_fusion.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_joint_checkpoint(project_root: Path) -> Path:
    candidates = [
        project_root
        / "experiments"
        / "artifacts"
        / "vision-spatial"
        / "encoders"
        / "joint_xyo"
        / "checkpoints"
        / "joint_best_joint_mlp_xyo_RGB.pth",
        project_root
        / "experiments"
        / "artifacts"
        / "vision-spatial"
        / "encoders"
        / "joint_xyo"
        / "checkpoints_joint_mlp_v2"
        / "best_joint_mlp_xyo_RGB.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def build_fusion_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["subject_id"] = df["subject_id"].astype(int).astype(str).str.zfill(5)
    rgb_df = df[df["modality"] == "RGB"].copy()
    rgb_df["rgb_path"] = (
        rgb_df["subject_id"]
        + "/RGB/"
        + rgb_df["condition"]
        + "/image_"
        + rgb_df["image_index"].astype(int).astype(str).str.zfill(6)
        + ".png"
    )
    rgb_df["depth_path"] = (
        rgb_df["subject_id"]
        + "/depth/"
        + rgb_df["condition"]
        + "/image_"
        + rgb_df["image_index"].astype(int).astype(str).str.zfill(6)
        + ".png"
    )
    rgb_df["joint_file"] = rgb_df["subject_id"] + "/joints_gt_RGB.mat"
    rgb_df["frame_idx_0based"] = rgb_df["image_index"] - 1
    return rgb_df[
        [
            "subject_id",
            "condition",
            "image_index",
            "label",
            "label_id",
            "rgb_path",
            "depth_path",
            "joint_file",
            "frame_idx_0based",
        ]
    ].reset_index(drop=True)


def subject_wise_split(
    subject_ids: list[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = SEED,
) -> tuple[list[str], list[str], list[str]]:
    subject_ids = sorted(subject_ids)
    rng = random.Random(seed)
    rng.shuffle(subject_ids)
    n = len(subject_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (
        subject_ids[:n_train],
        subject_ids[n_train : n_train + n_val],
        subject_ids[n_train + n_val :],
    )


def preprocess_gt_joints(frame_joints: np.ndarray) -> np.ndarray:
    x = frame_joints[0].astype(np.float32) / IMG_W
    y = frame_joints[1].astype(np.float32) / IMG_H
    occ = frame_joints[2].astype(np.float32)
    return np.stack([x, y, occ], axis=1).reshape(-1)


def slp_joints_to_vector(joints_3x14: np.ndarray) -> np.ndarray:
    x = joints_3x14[0].astype(np.float32)
    y = joints_3x14[1].astype(np.float32)
    occ = joints_3x14[2].astype(np.float32)
    return np.stack([x, y, occ], axis=1).reshape(-1)


def joint_vector_to_bbox(
    joint_vec: np.ndarray,
    image_size: tuple[int, int],
    padding: float = 0.15,
) -> tuple[int, int, int, int] | None:
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


def preprocess_depth_image(
    depth: Image.Image,
    mode: str,
    joint_vec: np.ndarray | None = None,
    body_padding: float = 0.15,
    norm_percentiles: tuple[float, float] = (2.0, 98.0),
) -> Image.Image:
    if mode == "none":
        return depth

    arr = np.asarray(depth.convert("L"), dtype=np.float32) / 255.0
    if mode == "invert":
        return Image.fromarray(((1.0 - arr) * 255).astype(np.uint8), mode="L")

    if mode not in {"body-norm", "body-norm-bg-zero", "body-norm-invert"}:
        raise ValueError(f"Unsupported depth preprocess mode: {mode}")

    bbox = (
        joint_vector_to_bbox(joint_vec, depth.size, padding=body_padding)
        if joint_vec is not None
        else None
    )
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


def _select_best_pose(result: Any) -> int | None:
    keypoints = getattr(result, "keypoints", None)
    boxes = getattr(result, "boxes", None)
    if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
        return None
    if boxes is not None and boxes.conf is not None and len(boxes.conf) == len(keypoints.xy):
        return int(torch.argmax(boxes.conf).item())
    return 0


def yolo_result_to_slp_vector(result: Any, conf_threshold: float = 0.2) -> tuple[np.ndarray, bool]:
    """Convert one Ultralytics pose result into the fusion model's 42-D joint vector."""
    pose_idx = _select_best_pose(result)
    if pose_idx is None:
        return np.zeros(42, dtype=np.float32), False

    xy = result.keypoints.xy[pose_idx].detach().cpu().numpy()
    conf = None
    if getattr(result.keypoints, "conf", None) is not None:
        conf = result.keypoints.conf[pose_idx].detach().cpu().numpy()

    joints = np.zeros((3, 14), dtype=np.float32)

    if xy.shape[0] >= 17:
        for slp_idx, coco_idx in COCO17_TO_SLP14.items():
            x_px, y_px = xy[coco_idx]
            score = float(conf[coco_idx]) if conf is not None else 1.0
            if x_px > 0 and y_px > 0:
                joints[0, slp_idx] = x_px / IMG_W
                joints[1, slp_idx] = y_px / IMG_H
            joints[2, slp_idx] = 0.0 if score >= conf_threshold else 1.0

        # Neck/thorax: midpoint of shoulders.
        left_shoulder = xy[5]
        right_shoulder = xy[6]
        shoulder_conf = (
            (float(conf[5]) + float(conf[6])) / 2 if conf is not None else 1.0
        )
        joints[0, 12] = (left_shoulder[0] + right_shoulder[0]) / 2 / IMG_W
        joints[1, 12] = (left_shoulder[1] + right_shoulder[1]) / 2 / IMG_H
        joints[2, 12] = 0.0 if shoulder_conf >= conf_threshold else 1.0

        # Head proxy: nose.
        nose_conf = float(conf[0]) if conf is not None else 1.0
        joints[0, 13] = xy[0, 0] / IMG_W
        joints[1, 13] = xy[0, 1] / IMG_H
        joints[2, 13] = 0.0 if nose_conf >= conf_threshold else 1.0
    elif xy.shape[0] == 14:
        for slp_idx in range(14):
            x_px, y_px = xy[slp_idx]
            score = float(conf[slp_idx]) if conf is not None else 1.0
            if x_px > 0 and y_px > 0:
                joints[0, slp_idx] = x_px / IMG_W
                joints[1, slp_idx] = y_px / IMG_H
            joints[2, slp_idx] = 0.0 if score >= conf_threshold else 1.0
    else:
        return np.zeros(42, dtype=np.float32), False

    detected = bool(np.count_nonzero(joints[:2]) > 0)
    return slp_joints_to_vector(joints), detected


def estimate_yolo_joints(
    model_path: Path,
    df: pd.DataFrame,
    slp_root: Path,
    cache_path: Path,
    conf_threshold: float,
    device: str | int | None,
    force: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, bool], dict[str, Any]]:
    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        joints = {k: np.array(v, dtype=np.float32) for k, v in raw["joints"].items()}
        flags = {k: bool(v) for k, v in raw["detected"].items()}
        return joints, flags, raw.get("metadata", {})

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: ultralytics. Install it with "
            "`.venv\\Scripts\\python.exe -m pip install ultralytics`."
        ) from exc

    pose_model = YOLO(str(model_path))
    joints_cache: dict[str, np.ndarray] = {}
    detection_flags: dict[str, bool] = {}
    keypoint_shape: list[int] | None = None

    for i, row in df.reset_index(drop=True).iterrows():
        key = f"{row['subject_id']}_{row['condition']}_{int(row['image_index']):06d}"
        image_path = slp_root / row["rgb_path"]
        result = pose_model.predict(
            source=str(image_path),
            conf=conf_threshold,
            verbose=False,
            device=device,
        )[0]
        joint_vec, detected = yolo_result_to_slp_vector(result, conf_threshold)
        joints_cache[key] = joint_vec
        detection_flags[key] = detected
        if keypoint_shape is None and getattr(result, "keypoints", None) is not None:
            keypoint_shape = list(result.keypoints.xy.shape)
        if (i + 1) % 100 == 0 or i == len(df) - 1:
            n_det = sum(detection_flags.values())
            print(f"YOLO joints {i + 1}/{len(df)} detected={n_det} ({n_det/(i+1):.1%})")

    metadata = {
        "model_path": str(model_path),
        "conf_threshold": conf_threshold,
        "keypoint_shape_example": keypoint_shape,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": metadata,
                "joints": {k: v.tolist() for k, v in joints_cache.items()},
                "detected": detection_flags,
            },
            f,
            indent=2,
        )

    return joints_cache, detection_flags, metadata


@torch.no_grad()
def synthesize_depth_pil(
    rgb_pil: Image.Image,
    processor: Any,
    depth_model: nn.Module,
    device: torch.device,
    target_size: tuple[int, int] = (424, 512),
) -> Image.Image:
    inputs = processor(images=rgb_pil, return_tensors="pt").to(device)
    pred = depth_model(**inputs).predicted_depth
    pred = F.interpolate(
        pred.unsqueeze(1),
        size=(target_size[1], target_size[0]),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()
    p_min, p_max = float(pred.min()), float(pred.max())
    norm = (pred - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(pred)
    return Image.fromarray((norm * 255).astype(np.uint8), mode="L")


def ensure_synthetic_depth_cache(
    df: pd.DataFrame,
    slp_root: Path,
    cache_root: Path,
    model_name: str,
    device: torch.device,
    force: bool = False,
) -> dict[str, Any]:
    missing_rows = []
    for _, row in df.iterrows():
        depth_path = (
            cache_root
            / row["subject_id"]
            / "depth"
            / row["condition"]
            / f"image_{int(row['image_index']):06d}.png"
        )
        if force or not depth_path.exists():
            missing_rows.append((row, depth_path))

    if not missing_rows:
        return {
            "depth_source": "synthetic",
            "model_name": model_name,
            "cache_root": str(cache_root),
            "generated": 0,
            "reused": len(df),
        }

    try:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install project requirements before "
            "generating synthetic depth."
        ) from exc

    print(f"Generating synthetic depth for {len(missing_rows)} missing frames...")
    processor = AutoImageProcessor.from_pretrained(model_name)
    depth_model = AutoModelForDepthEstimation.from_pretrained(model_name).to(device).eval()

    for i, (row, depth_path) in enumerate(missing_rows):
        rgb = Image.open(slp_root / row["rgb_path"]).convert("RGB")
        synth = synthesize_depth_pil(rgb, processor, depth_model, device)
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        synth.save(depth_path)
        if (i + 1) % 100 == 0 or i == len(missing_rows) - 1:
            print(f"Synthetic depth {i + 1}/{len(missing_rows)}")

    return {
        "depth_source": "synthetic",
        "model_name": model_name,
        "cache_root": str(cache_root),
        "generated": len(missing_rows),
        "reused": len(df) - len(missing_rows),
    }


class FusionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        slp_root: Path,
        joints_cache: dict[str, np.ndarray],
        joint_source: str,
        depth_source: str = "real",
        depth_synth_root: Path | None = None,
        depth_preprocess: str = "none",
        body_padding: float = 0.15,
        norm_percentiles: tuple[float, float] = (2.0, 98.0),
    ):
        if joint_source not in {"gt", "yolo"}:
            raise ValueError(f"Unsupported joint source: {joint_source}")
        if depth_source not in {"real", "synthetic", "zero"}:
            raise ValueError(f"Unsupported depth source: {depth_source}")
        self.df = df.reset_index(drop=True)
        self.slp_root = Path(slp_root)
        self.joints_cache = joints_cache
        self.joint_source = joint_source
        self.depth_source = depth_source
        self.depth_synth_root = Path(depth_synth_root) if depth_synth_root else None
        self.depth_preprocess = depth_preprocess
        self.body_padding = body_padding
        self.norm_percentiles = norm_percentiles
        self.rgb_tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        self.depth_tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        self._joint_mat_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_gt_joints(self, joint_rel_path: str) -> np.ndarray:
        if joint_rel_path not in self._joint_mat_cache:
            self._joint_mat_cache[joint_rel_path] = sio.loadmat(
                self.slp_root / joint_rel_path
            )["joints_gt"]
        return self._joint_mat_cache[joint_rel_path]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        rgb = Image.open(self.slp_root / row["rgb_path"]).convert("RGB")

        if self.joint_source == "gt":
            mat = self._load_gt_joints(row["joint_file"])
            frame = mat[:, :, int(row["frame_idx_0based"])]
            joint_vec = preprocess_gt_joints(frame)
        else:
            key = f"{row['subject_id']}_{row['condition']}_{int(row['image_index']):06d}"
            joint_vec = self.joints_cache[key]

        if self.depth_source == "zero":
            depth = Image.new("L", (224, 224), color=0)
        elif self.depth_source == "synthetic":
            if self.depth_synth_root is None:
                raise RuntimeError("depth_synth_root is required for synthetic depth")
            depth_path = (
                self.depth_synth_root
                / row["subject_id"]
                / "depth"
                / row["condition"]
                / f"image_{int(row['image_index']):06d}.png"
            )
            depth = Image.open(depth_path).convert("L")
            depth = preprocess_depth_image(
                depth,
                self.depth_preprocess,
                joint_vec,
                body_padding=self.body_padding,
                norm_percentiles=self.norm_percentiles,
            )
        else:
            depth_path = self.slp_root / row["depth_path"]
            depth = Image.open(depth_path).convert("L")

        return {
            "rgb": self.rgb_tf(rgb),
            "depth": self.depth_tf(depth),
            "joint": torch.tensor(joint_vec, dtype=torch.float32),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
        }


class JointOnlyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        slp_root: Path,
        joints_cache: dict[str, np.ndarray],
        joint_source: str,
    ):
        if joint_source not in {"gt", "yolo"}:
            raise ValueError(f"Unsupported joint source: {joint_source}")
        self.df = df.reset_index(drop=True)
        self.slp_root = Path(slp_root)
        self.joints_cache = joints_cache
        self.joint_source = joint_source
        self._joint_mat_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _load_gt_joints(self, joint_rel_path: str) -> np.ndarray:
        if joint_rel_path not in self._joint_mat_cache:
            self._joint_mat_cache[joint_rel_path] = sio.loadmat(
                self.slp_root / joint_rel_path
            )["joints_gt"]
        return self._joint_mat_cache[joint_rel_path]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        if self.joint_source == "gt":
            mat = self._load_gt_joints(row["joint_file"])
            frame = mat[:, :, int(row["frame_idx_0based"])]
            joint_vec = preprocess_gt_joints(frame)
        else:
            key = f"{row['subject_id']}_{row['condition']}_{int(row['image_index']):06d}"
            joint_vec = self.joints_cache[key]
        return {
            "joint": torch.tensor(joint_vec, dtype=torch.float32),
            "label": torch.tensor(int(row["label_id"]), dtype=torch.long),
        }


def load_fusion_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = AttentionFusionClassifier(
        depth_encoder=DepthEncoder(),
        rgb_encoder=RGBEncoder(),
        joint_encoder=JointEncoder(input_dim=42, hidden_dim=128, feature_dim=128, dropout=0.3),
        depth_feature_dim=512,
        rgb_feature_dim=128,
        joint_feature_dim=128,
        common_feature_dim=256,
        num_heads=4,
        ff_dim=512,
        dropout=0.1,
        num_classes=NUM_CLASSES,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_joint_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    model = JointMLP(
        input_dim=int(ckpt.get("input_dim", 42)),
        hidden_dim=int(config.get("hidden_dim1", 128)),
        feature_dim=int(ckpt.get("feature_dim", config.get("hidden_dim2", 128))),
        dropout=float(config.get("dropout", 0.3)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_torch_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, label: str) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for i, batch in enumerate(loader):
        logits = model(
            batch["depth"].to(device, non_blocking=True),
            batch["rgb"].to(device, non_blocking=True),
            batch["joint"].to(device, non_blocking=True),
        )
        preds = torch.argmax(logits, dim=1)
        y_true.extend(batch["label"].numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        if (i + 1) % 10 == 0 or i == len(loader) - 1:
            print(f"{label} batch {i + 1}/{len(loader)}")

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true_np,
            y_pred_np,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true_np, y_pred_np, labels=list(range(NUM_CLASSES))
        ).tolist(),
    }


@torch.no_grad()
def evaluate_joint_only(model: nn.Module, loader: DataLoader, device: torch.device, label: str) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    for i, batch in enumerate(loader):
        logits = model(batch["joint"].to(device, non_blocking=True))
        preds = torch.argmax(logits, dim=1)
        y_true.extend(batch["label"].numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())
        if (i + 1) % 10 == 0 or i == len(loader) - 1:
            print(f"{label} batch {i + 1}/{len(loader)}")

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true_np,
            y_pred_np,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true_np, y_pred_np, labels=list(range(NUM_CLASSES))
        ).tolist(),
    }


def save_per_class_comparison(results_gt: dict[str, Any], results_yolo: dict[str, Any], out_path: Path) -> None:
    rows = []
    for cls in CLASS_NAMES:
        gt = results_gt["classification_report"][cls]
        yolo = results_yolo["classification_report"][cls]
        rows.append(
            {
                "class": cls,
                "support": gt["support"],
                "f1_gt": round(gt["f1-score"], 4),
                "f1_yolo": round(yolo["f1-score"], 4),
                "delta_f1": round(yolo["f1-score"] - gt["f1-score"], 4),
                "recall_gt": round(gt["recall"], 4),
                "recall_yolo": round(yolo["recall"], 4),
                "delta_recall": round(yolo["recall"] - gt["recall"], 4),
                "precision_gt": round(gt["precision"], 4),
                "precision_yolo": round(yolo["precision"], 4),
                "delta_precision": round(yolo["precision"] - gt["precision"], 4),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def get_args() -> argparse.Namespace:
    project_root = find_project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--slp-root",
        type=Path,
        default=project_root.parent / "SLP2022" / "SLP" / "danaLab",
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint(project_root))
    parser.add_argument("--joint-checkpoint", type=Path, default=default_joint_checkpoint(project_root))
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=project_root.parent / "yolov8n-pose.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root
        / "experiments"
        / "artifacts"
        / "vision-spatial"
        / "inference_experiments"
        / "yolo_joints_eval",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-mode", choices=["fusion", "joint-only"], default="fusion")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test sample limit.")
    parser.add_argument("--conf-threshold", type=float, default=0.2)
    parser.add_argument("--force-yolo-cache", action="store_true")
    parser.add_argument("--depth-source", choices=["real", "synthetic", "zero"], default="real")
    parser.add_argument(
        "--depth-preprocess",
        choices=["none", "invert", "body-norm", "body-norm-bg-zero", "body-norm-invert"],
        default="none",
        help="Synthetic-depth preprocessing ablation.",
    )
    parser.add_argument(
        "--body-padding",
        type=float,
        default=0.15,
        help="Padding ratio for body-box synthetic-depth normalization.",
    )
    parser.add_argument(
        "--norm-low",
        type=float,
        default=2.0,
        help="Lower percentile for body-box synthetic-depth normalization.",
    )
    parser.add_argument(
        "--norm-high",
        type=float,
        default=98.0,
        help="Upper percentile for body-box synthetic-depth normalization.",
    )
    parser.add_argument(
        "--depth-model",
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face depth model used when --depth-source synthetic.",
    )
    parser.add_argument(
        "--depth-synth-root",
        type=Path,
        default=project_root
        / "experiments"
        / "artifacts"
        / "vision-spatial"
        / "inference_experiments"
        / "synthetic_depth_eval"
        / "depth_synth_cache",
    )
    parser.add_argument("--force-depth-cache", action="store_true")
    parser.add_argument(
        "--yolo-device",
        default=None,
        help="Ultralytics device override, e.g. '0' for CUDA GPU or 'cpu'.",
    )
    return parser.parse_args()


def main() -> None:
    args = get_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    csv_path = args.slp_root / "posture_labels_all_modalities.csv"
    results_dir = args.out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"SLP root      : {args.slp_root} exists={args.slp_root.exists()}")
    print(f"Labels CSV    : {csv_path} exists={csv_path.exists()}")
    print(f"Fusion ckpt   : {args.checkpoint} exists={args.checkpoint.exists()}")
    print(f"Joint ckpt    : {args.joint_checkpoint} exists={args.joint_checkpoint.exists()}")
    print(f"YOLO model    : {args.yolo_model} exists={args.yolo_model.exists()}")
    print(f"Depth source  : {args.depth_source}")
    print(f"Depth prep    : {args.depth_preprocess}")
    if args.depth_preprocess.startswith("body-norm"):
        print(f"Body padding  : {args.body_padding}")
        print(f"Norm pctile   : {args.norm_low:g}/{args.norm_high:g}")
    if args.depth_source == "synthetic":
        print(f"Synth cache   : {args.depth_synth_root}")
    print(f"Output dir    : {args.out_dir}")

    fusion_df = build_fusion_metadata(csv_path)
    subjects = sorted(fusion_df["subject_id"].unique().tolist())
    _, _, test_subjects = subject_wise_split(subjects)
    test_df = fusion_df[fusion_df["subject_id"].isin(test_subjects)].reset_index(drop=True)
    if args.limit is not None:
        test_df = test_df.iloc[: args.limit].reset_index(drop=True)

    print(f"Test subjects : {len(test_subjects)}")
    print(f"Test samples  : {len(test_df)}")
    print(test_df["label"].value_counts().rename("count").to_string())

    cache_suffix = f"_limit{args.limit}" if args.limit is not None else ""
    yolo_cache = args.out_dir / f"yolo_joints_cache{cache_suffix}.json"
    yolo_joints, detection_flags, yolo_metadata = estimate_yolo_joints(
        model_path=args.yolo_model,
        df=test_df,
        slp_root=args.slp_root,
        cache_path=yolo_cache,
        conf_threshold=args.conf_threshold,
        device=args.yolo_device,
        force=args.force_yolo_cache,
    )
    n_detected = sum(detection_flags.values())
    print(f"YOLO detection rate: {n_detected}/{len(test_df)} ({n_detected / len(test_df):.1%})")

    device = get_torch_device()
    print(f"Fusion device: {device}")

    if args.model_mode == "joint-only":
        model = load_joint_model(args.joint_checkpoint, device)
        loader_gt = DataLoader(
            JointOnlyDataset(test_df, args.slp_root, yolo_joints, joint_source="gt"),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        loader_yolo = DataLoader(
            JointOnlyDataset(test_df, args.slp_root, yolo_joints, joint_source="yolo"),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        results_gt = evaluate_joint_only(model, loader_gt, device, "joint-only + GT-joints")
        results_yolo = evaluate_joint_only(model, loader_yolo, device, "joint-only + YOLO-joints")
        output = {
            "experiment": "joint_only_yolo_joints_eval",
            "joint_checkpoint": str(args.joint_checkpoint),
            "yolo_model": str(args.yolo_model),
            "yolo_metadata": yolo_metadata,
            "test_samples": len(test_df),
            "test_subjects": test_subjects,
            "class_names": CLASS_NAMES,
            "detection_rate": {
                "n_detected": n_detected,
                "n_total": len(test_df),
                "fraction": n_detected / len(test_df),
            },
            "joint_only_gt_joints": results_gt,
            "joint_only_yolo_joints": results_yolo,
            "joint_estimation_gap": {
                "accuracy": results_yolo["accuracy"] - results_gt["accuracy"],
                "macro_f1": results_yolo["macro_f1"] - results_gt["macro_f1"],
            },
        }
        result_path = results_dir / "joint_only_yolo_joints_eval_results.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        save_per_class_comparison(
            results_gt,
            results_yolo,
            results_dir / "per_class_comparison_joint_only.csv",
        )
        print("\n=== Results ===")
        print(
            "joint-only + GT joints  : "
            f"acc {results_gt['accuracy']:.4f}, macro F1 {results_gt['macro_f1']:.4f}"
        )
        print(
            "joint-only + YOLO joints: "
            f"acc {results_yolo['accuracy']:.4f}, macro F1 {results_yolo['macro_f1']:.4f}"
        )
        print(
            "Gap                    : "
            f"acc {output['joint_estimation_gap']['accuracy']:+.4f}, "
            f"macro F1 {output['joint_estimation_gap']['macro_f1']:+.4f}"
        )
        print(f"Saved: {result_path}")
        return

    depth_metadata: dict[str, Any] = {"depth_source": args.depth_source}
    if args.depth_source == "synthetic":
        depth_metadata = ensure_synthetic_depth_cache(
            df=test_df,
            slp_root=args.slp_root,
            cache_root=args.depth_synth_root,
            model_name=args.depth_model,
            device=device,
            force=args.force_depth_cache,
        )
        depth_metadata["depth_preprocess"] = args.depth_preprocess
        depth_metadata["body_padding"] = args.body_padding
        depth_metadata["norm_percentiles"] = [args.norm_low, args.norm_high]

    model = load_fusion_model(args.checkpoint, device)

    run_prefix_by_source = {
        "real": "real-depth",
        "synthetic": "synthetic-depth",
        "zero": "zero-depth",
    }
    run_prefix = run_prefix_by_source[args.depth_source]

    loader_gt = DataLoader(
        FusionDataset(
            test_df,
            args.slp_root,
            yolo_joints,
            joint_source="gt",
            depth_source=args.depth_source,
            depth_synth_root=args.depth_synth_root,
            depth_preprocess=args.depth_preprocess,
            body_padding=args.body_padding,
            norm_percentiles=(args.norm_low, args.norm_high),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    loader_yolo = DataLoader(
        FusionDataset(
            test_df,
            args.slp_root,
            yolo_joints,
            joint_source="yolo",
            depth_source=args.depth_source,
            depth_synth_root=args.depth_synth_root,
            depth_preprocess=args.depth_preprocess,
            body_padding=args.body_padding,
            norm_percentiles=(args.norm_low, args.norm_high),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    results_gt = evaluate(model, loader_gt, device, f"{run_prefix} + GT-joints")
    results_yolo = evaluate(model, loader_yolo, device, f"{run_prefix} + YOLO-joints")

    output = {
        "experiment": "attention_fusion_yolo_joints_eval",
        "baseline_checkpoint": str(args.checkpoint),
        "depth_metadata": depth_metadata,
        "yolo_model": str(args.yolo_model),
        "yolo_metadata": yolo_metadata,
        "test_samples": len(test_df),
        "test_subjects": test_subjects,
        "class_names": CLASS_NAMES,
        "detection_rate": {
            "n_detected": n_detected,
            "n_total": len(test_df),
            "fraction": n_detected / len(test_df),
        },
        f"{args.depth_source}_depth_gt_joints": results_gt,
        f"{args.depth_source}_depth_yolo_joints": results_yolo,
        "joint_estimation_gap": {
            "accuracy": results_yolo["accuracy"] - results_gt["accuracy"],
            "macro_f1": results_yolo["macro_f1"] - results_gt["macro_f1"],
        },
    }

    prep_suffix = "" if args.depth_preprocess == "none" else f"_{args.depth_preprocess.replace('-', '_')}"
    if args.depth_preprocess.startswith("body-norm") and abs(args.body_padding - 0.15) > 1e-9:
        pad_tag = str(args.body_padding).replace(".", "p")
        prep_suffix = f"{prep_suffix}_pad_{pad_tag}"
    if args.depth_preprocess.startswith("body-norm") and (
        abs(args.norm_low - 2.0) > 1e-9 or abs(args.norm_high - 98.0) > 1e-9
    ):
        low_tag = str(args.norm_low).replace(".", "p")
        high_tag = str(args.norm_high).replace(".", "p")
        prep_suffix = f"{prep_suffix}_pct_{low_tag}_{high_tag}"
    result_path = results_dir / f"yolo_joints_eval_{args.depth_source}_depth{prep_suffix}_results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    save_per_class_comparison(
        results_gt,
        results_yolo,
        results_dir / f"per_class_comparison_{args.depth_source}_depth{prep_suffix}.csv",
    )

    print("\n=== Results ===")
    print(
        f"{run_prefix} + GT joints  : "
        f"acc {results_gt['accuracy']:.4f}, macro F1 {results_gt['macro_f1']:.4f}"
    )
    print(
        f"{run_prefix} + YOLO joints: "
        f"acc {results_yolo['accuracy']:.4f}, macro F1 {results_yolo['macro_f1']:.4f}"
    )
    print(
        "Gap                    : "
        f"acc {output['joint_estimation_gap']['accuracy']:+.4f}, "
        f"macro F1 {output['joint_estimation_gap']['macro_f1']:+.4f}"
    )
    print(f"Saved: {result_path}")


if __name__ == "__main__":
    main()
