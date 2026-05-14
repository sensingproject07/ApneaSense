"""
Consumer-only adaptation for attention_fusion.

This script leaves the clinical pipeline and clinical checkpoint untouched. It
loads the existing attention_fusion checkpoint, trains only the consumer-facing
fusion layers on deployment-style inputs, and writes a separate consumer
checkpoint under experiments/artifacts.

Inputs:
  RGB image + synthetic depth (Depth Anything v2 cache, body-normalized/inverted)
  + YOLO-estimated joints.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in [THIS_FILE.parent, *THIS_FILE.parents] if (p / "requirements.txt").exists()),
    Path.cwd().resolve(),
)
SCRIPT_DIR = THIS_FILE.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from yolo_joints_eval import (  # noqa: E402
    CLASS_NAMES,
    NUM_CLASSES,
    SEED,
    FusionDataset,
    build_fusion_metadata,
    default_checkpoint,
    ensure_synthetic_depth_cache,
    estimate_yolo_joints,
    get_torch_device,
    load_fusion_model,
    subject_wise_split,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    for param in model.parameters():
        param.requires_grad = False

    trainable_modules = [
        model.depth_proj,
        model.attention_block,
        model.classifier,
    ]
    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True
    model.modality_embed.requires_grad = True

    return [p for p in model.parameters() if p.requires_grad]


def set_frozen_encoders_eval(model: nn.Module) -> None:
    model.depth_encoder.eval()
    model.rgb_encoder.eval()
    model.joint_encoder.eval()


def make_loader(
    df: pd.DataFrame,
    slp_root: Path,
    joints_cache: dict[str, np.ndarray],
    depth_synth_root: Path,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    depth_preprocess: str,
    body_padding: float,
    norm_low: float,
    norm_high: float,
) -> DataLoader:
    dataset = FusionDataset(
        df=df,
        slp_root=slp_root,
        joints_cache=joints_cache,
        joint_source="yolo",
        depth_source="synthetic",
        depth_synth_root=depth_synth_root,
        depth_preprocess=depth_preprocess,
        body_padding=body_padding,
        norm_percentiles=(norm_low, norm_high),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        depth = batch["depth"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        joint = batch["joint"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(depth, rgb, joint)
        loss = criterion(logits, label)
        preds = torch.argmax(logits, dim=1)
        losses.append(float(loss.item()))
        y_true.extend(label.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
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
            y_true_np,
            y_pred_np,
            labels=list(range(NUM_CLASSES)),
        ).tolist(),
    }


@torch.no_grad()
def encode_features(model: nn.Module, loader: DataLoader, device: torch.device) -> TensorDataset:
    model.eval()
    set_frozen_encoders_eval(model)
    depth_features: list[torch.Tensor] = []
    rgb_features: list[torch.Tensor] = []
    joint_features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    for batch in loader:
        depth = batch["depth"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        joint = batch["joint"].to(device, non_blocking=True)
        depth_features.append(model.depth_encoder(depth).cpu())
        rgb_features.append(model.rgb_encoder(rgb).cpu())
        joint_features.append(model.joint_encoder(joint).cpu())
        labels.append(batch["label"].long().cpu())

    return TensorDataset(
        torch.cat(depth_features, dim=0),
        torch.cat(rgb_features, dim=0),
        torch.cat(joint_features, dim=0),
        torch.cat(labels, dim=0),
    )


def make_feature_loader(
    dataset: TensorDataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def logits_from_features(
    model: nn.Module,
    depth_features: torch.Tensor,
    rgb_features: torch.Tensor,
    joint_features: torch.Tensor,
) -> torch.Tensor:
    t_depth = model.depth_proj(depth_features)
    t_rgb = model.rgb_proj(rgb_features)
    t_joint = model.joint_proj(joint_features)
    tokens = torch.stack([t_rgb, t_depth, t_joint], dim=1)
    tokens = tokens + model.modality_embed.unsqueeze(0)
    tokens = model.attention_block(tokens)
    return model.classifier(tokens.mean(dim=1))


@torch.no_grad()
def evaluate_features(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    losses: list[float] = []
    criterion = nn.CrossEntropyLoss()

    for depth_f, rgb_f, joint_f, label in loader:
        depth_f = depth_f.to(device, non_blocking=True)
        rgb_f = rgb_f.to(device, non_blocking=True)
        joint_f = joint_f.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = logits_from_features(model, depth_f, rgb_f, joint_f)
        loss = criterion(logits, label)
        preds = torch.argmax(logits, dim=1)
        losses.append(float(loss.item()))
        y_true.extend(label.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
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
            y_true_np,
            y_pred_np,
            labels=list(range(NUM_CLASSES)),
        ).tolist(),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    set_frozen_encoders_eval(model)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []

    for batch in loader:
        depth = batch["depth"].to(device, non_blocking=True)
        rgb = batch["rgb"].to(device, non_blocking=True)
        joint = batch["joint"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(depth, rgb, joint)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def train_one_epoch_features(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []

    for depth_f, rgb_f, joint_f, label in loader:
        depth_f = depth_f.to(device, non_blocking=True)
        rgb_f = rgb_f.to(device, non_blocking=True)
        joint_f = joint_f.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = logits_from_features(model, depth_f, rgb_f, joint_f)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def save_per_class_comparison(
    baseline: dict[str, Any],
    finetuned: dict[str, Any],
    out_path: Path,
) -> None:
    rows = []
    for cls in CLASS_NAMES:
        base = baseline["classification_report"][cls]
        tuned = finetuned["classification_report"][cls]
        rows.append(
            {
                "class": cls,
                "support": base["support"],
                "f1_baseline": round(base["f1-score"], 4),
                "f1_finetuned": round(tuned["f1-score"], 4),
                "delta_f1": round(tuned["f1-score"] - base["f1-score"], 4),
                "recall_baseline": round(base["recall"], 4),
                "recall_finetuned": round(tuned["recall"], 4),
                "delta_recall": round(tuned["recall"] - base["recall"], 4),
                "precision_baseline": round(base["precision"], 4),
                "precision_finetuned": round(tuned["precision"], 4),
                "delta_precision": round(tuned["precision"] - base["precision"], 4),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--slp-root", type=Path, default=PROJECT_ROOT.parent / "SLP2022" / "SLP" / "danaLab")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint(PROJECT_ROOT))
    parser.add_argument("--yolo-model", type=Path, default=PROJECT_ROOT.parent / "yolov8n-pose.pt")
    parser.add_argument(
        "--depth-synth-root",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "artifacts"
        / "vision-spatial"
        / "inference_experiments"
        / "synthetic_depth_eval"
        / "depth_synth_cache",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "artifacts"
        / "vision-spatial"
        / "inference_experiments"
        / "consumer_finetune",
    )
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--depth-preprocess", default="body-norm-invert")
    parser.add_argument("--body-padding", type=float, default=0.0)
    parser.add_argument("--norm-low", type=float, default=2.0)
    parser.add_argument("--norm-high", type=float, default=98.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit-per-split", type=int, default=None, help="Smoke-test limit per split.")
    parser.add_argument("--force-yolo-cache", action="store_true")
    parser.add_argument("--force-depth-cache", action="store_true")
    parser.add_argument("--yolo-device", default=None)
    return parser.parse_args()


def main() -> None:
    args = get_args()
    set_seed(SEED)

    csv_path = args.slp_root / "posture_labels_all_modalities.csv"
    checkpoints_dir = args.out_dir / "checkpoints"
    results_dir = args.out_dir / "results"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"SLP root        : {args.slp_root} exists={args.slp_root.exists()}")
    print(f"Labels CSV      : {csv_path} exists={csv_path.exists()}")
    print(f"Clinical ckpt   : {args.checkpoint} exists={args.checkpoint.exists()}")
    print(f"YOLO model      : {args.yolo_model} exists={args.yolo_model.exists()}")
    print(f"Synth depth root: {args.depth_synth_root}")
    print(f"Output dir      : {args.out_dir}")
    print(
        "Consumer depth  : "
        f"{args.depth_preprocess}, padding={args.body_padding}, "
        f"pct={args.norm_low:g}/{args.norm_high:g}"
    )

    fusion_df = build_fusion_metadata(csv_path)
    train_subjects, val_subjects, test_subjects = subject_wise_split(
        sorted(fusion_df["subject_id"].unique().tolist())
    )
    train_df = fusion_df[fusion_df["subject_id"].isin(train_subjects)].reset_index(drop=True)
    val_df = fusion_df[fusion_df["subject_id"].isin(val_subjects)].reset_index(drop=True)
    test_df = fusion_df[fusion_df["subject_id"].isin(test_subjects)].reset_index(drop=True)
    if args.limit_per_split is not None:
        train_df = train_df.iloc[: args.limit_per_split].reset_index(drop=True)
        val_df = val_df.iloc[: args.limit_per_split].reset_index(drop=True)
        test_df = test_df.iloc[: args.limit_per_split].reset_index(drop=True)

    print(f"Train/val/test samples: {len(train_df)} / {len(val_df)} / {len(test_df)}")

    device = get_torch_device()
    print(f"Device: {device}")

    split_frames = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }
    yolo_caches: dict[str, dict[str, np.ndarray]] = {}
    yolo_metadata: dict[str, Any] = {}
    cache_suffix = f"_limit{args.limit_per_split}" if args.limit_per_split is not None else ""
    for split_name, split_df in split_frames.items():
        cache_path = args.out_dir / f"yolo_joints_cache_{split_name}{cache_suffix}.json"
        # Reuse the existing test cache from yolo_joints_eval when available.
        if split_name == "test" and args.limit_per_split is None:
            existing_test_cache = (
                PROJECT_ROOT
                / "experiments"
                / "artifacts"
                / "vision-spatial"
                / "inference_experiments"
                / "yolo_joints_eval"
                / "yolo_joints_cache.json"
            )
            if existing_test_cache.exists() and not cache_path.exists():
                cache_path = existing_test_cache
        joints, flags, metadata = estimate_yolo_joints(
            model_path=args.yolo_model,
            df=split_df,
            slp_root=args.slp_root,
            cache_path=cache_path,
            conf_threshold=0.2,
            device=args.yolo_device,
            force=args.force_yolo_cache,
        )
        yolo_caches[split_name] = joints
        yolo_metadata[split_name] = {
            **metadata,
            "n_detected": int(sum(flags.values())),
            "n_total": int(len(split_df)),
            "fraction": float(sum(flags.values()) / len(split_df)) if len(split_df) else 0.0,
        }
        print(
            f"{split_name} YOLO detections: "
            f"{yolo_metadata[split_name]['n_detected']}/{yolo_metadata[split_name]['n_total']} "
            f"({yolo_metadata[split_name]['fraction']:.1%})"
        )

    for split_name, split_df in split_frames.items():
        print(f"Ensuring synthetic depth cache for {split_name}...")
        ensure_synthetic_depth_cache(
            df=split_df,
            slp_root=args.slp_root,
            cache_root=args.depth_synth_root,
            model_name=args.depth_model,
            device=device,
            force=args.force_depth_cache,
        )

    train_loader = make_loader(
        train_df,
        args.slp_root,
        yolo_caches["train"],
        args.depth_synth_root,
        args.batch_size,
        shuffle=True,
        device=device,
        depth_preprocess=args.depth_preprocess,
        body_padding=args.body_padding,
        norm_low=args.norm_low,
        norm_high=args.norm_high,
    )
    val_loader = make_loader(
        val_df,
        args.slp_root,
        yolo_caches["val"],
        args.depth_synth_root,
        args.batch_size,
        shuffle=False,
        device=device,
        depth_preprocess=args.depth_preprocess,
        body_padding=args.body_padding,
        norm_low=args.norm_low,
        norm_high=args.norm_high,
    )
    test_loader = make_loader(
        test_df,
        args.slp_root,
        yolo_caches["test"],
        args.depth_synth_root,
        args.batch_size,
        shuffle=False,
        device=device,
        depth_preprocess=args.depth_preprocess,
        body_padding=args.body_padding,
        norm_low=args.norm_low,
        norm_high=args.norm_high,
    )

    feature_source_model = load_fusion_model(args.checkpoint, device)
    print("Encoding frozen train/val/test features...")
    train_feature_data = encode_features(feature_source_model, train_loader, device)
    val_feature_data = encode_features(feature_source_model, val_loader, device)
    test_feature_data = encode_features(feature_source_model, test_loader, device)
    train_feature_loader = make_feature_loader(
        train_feature_data,
        args.batch_size,
        shuffle=True,
        device=device,
    )
    val_feature_loader = make_feature_loader(
        val_feature_data,
        args.batch_size,
        shuffle=False,
        device=device,
    )
    test_feature_loader = make_feature_loader(
        test_feature_data,
        args.batch_size,
        shuffle=False,
        device=device,
    )

    baseline_test = evaluate_features(feature_source_model, test_feature_loader, device)
    print(
        "Baseline test: "
        f"acc {baseline_test['accuracy']:.4f}, macro F1 {baseline_test['macro_f1']:.4f}"
    )

    model = load_fusion_model(args.checkpoint, device)
    trainable_params = configure_trainable_parameters(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    best_val_f1 = -1.0
    best_state = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch_features(model, train_feature_loader, optimizer, device)
        val_metrics = evaluate_features(model, val_feature_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    finetuned_test = evaluate_features(model, test_feature_loader, device)
    print(
        "Fine-tuned test: "
        f"acc {finetuned_test['accuracy']:.4f}, macro F1 {finetuned_test['macro_f1']:.4f}"
    )

    ckpt_path = checkpoints_dir / "best_consumer_attention_fusion.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_checkpoint": str(args.checkpoint),
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "config": {
                "depth_preprocess": args.depth_preprocess,
                "body_padding": args.body_padding,
                "norm_low": args.norm_low,
                "norm_high": args.norm_high,
                "trainable": ["depth_proj", "modality_embed", "attention_block", "classifier"],
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
        },
        ckpt_path,
    )

    pd.DataFrame(history).to_csv(results_dir / "train_history.csv", index=False)
    save_per_class_comparison(
        baseline_test,
        finetuned_test,
        results_dir / "per_class_comparison.csv",
    )
    results = {
        "experiment": "consumer_attention_fusion_finetune",
        "source_checkpoint": str(args.checkpoint),
        "consumer_checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "test_subjects": test_subjects,
        "yolo_metadata": yolo_metadata,
        "config": {
            "depth_preprocess": args.depth_preprocess,
            "body_padding": args.body_padding,
            "norm_low": args.norm_low,
            "norm_high": args.norm_high,
            "trainable": ["depth_proj", "modality_embed", "attention_block", "classifier"],
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        "baseline_test": baseline_test,
        "finetuned_test": finetuned_test,
        "delta": {
            "accuracy": finetuned_test["accuracy"] - baseline_test["accuracy"],
            "macro_f1": finetuned_test["macro_f1"] - baseline_test["macro_f1"],
        },
    }
    with (results_dir / "consumer_finetune_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved results   : {results_dir / 'consumer_finetune_results.json'}")


if __name__ == "__main__":
    main()
