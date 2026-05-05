# Model Card — Vision-Spatial Posture Classifier (attention_fusion)

## Overview

Multi-modal classifier that predicts sleep posture (supine / left / right) from three synchronized inputs — RGB image, depth image, and 2D body joints. Self-attention fuses modality-specific features. Frozen-encoder design: RGB, depth, and joint encoders were pretrained separately and kept frozen during fusion-head training (selected over a Stage-2 finetune that did not improve performance — see "Provenance" below).

## Files

| File | Path | Purpose |
|---|---|---|
| Weights | `models/vision-spatial/attention_fusion.pth` | PyTorch state-dict checkpoint (full model, encoders included) |
| Preprocessing config | `artifacts/vision-spatial/vision_config.json` | Per-modality transforms; must match training |
| Label map | `artifacts/vision-spatial/class_names.json` | Logit-index → class-name |

The checkpoint contains `model_state_dict` (the entire `AttentionFusionClassifier` including the three encoders), plus `config`, `best_val_f1`, and `best_epoch` keys for traceability. **The standalone encoder checkpoints are not needed at inference** — they were used only at training time to initialize the encoder weights before freezing.

## Architecture

`AttentionFusionClassifier` — defined in [`experiments/notebooks/vision-spatial/fusion_experiments/attention_fusion_experiment.ipynb`](../../experiments/notebooks/vision-spatial/fusion_experiments/attention_fusion_experiment.ipynb). Inference code must be able to reconstruct this class to call `load_state_dict` — until it's extracted into a `.py` module, copy the class definitions from the notebook (or import the notebook's cell module).

### Components

- **DepthEncoder** — `torchvision.models.resnet18(weights=None)` with `conv1` reinitialised to 1-channel input; FC layer dropped; `Flatten()` after `avgpool`. Output: 512-d feature.
- **RGBEncoder** — `torchvision.models.resnet18(weights=None)` with `fc` replaced by `nn.Identity()`; followed by a 2-layer projection head: `Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 128)`. Output: 128-d feature.
- **JointEncoder** — MLP: `Linear(42, 128) → ReLU → Dropout(0.3) → Linear(128, 128) → ReLU → Dropout(0.3)`. Output: 128-d feature.
- **Projection layers** — three `Linear` layers projecting each modality feature into the common 256-d space.
- **Modality embedding** — learned `nn.Parameter(torch.randn(3, 256))` added to the three modality tokens before attention.
- **TransformerEncoder** — single layer; `d_model=256`, `nhead=4`, `dim_feedforward=512`, `dropout=0.1`, `activation="gelu"`, `batch_first=True`, `norm_first=True`.
- **Pooling** — mean over the 3 modality tokens.
- **Classifier head** — `Linear(256, 128) → ReLU → Dropout(0.3) → Linear(128, 3)`.

### Forward signature

```python
logits = model(depth_x, rgb_x, joint_x)
# depth_x : (B, 1, 224, 224) float32
# rgb_x   : (B, 3, 224, 224) float32
# joint_x : (B, 42)          float32
# logits  : (B, 3)           — argmax for class index, softmax for probabilities
```

Note the argument order: **depth, RGB, joint** (alphabetical-by-modality is RGB-first; the model is not). Token assembly inside `forward` is `[rgb, depth, joint]`, so the modality_embedding rows are also in that order, but the function signature itself takes depth first.

The encoders run inside `torch.no_grad()` in `forward`, matching the frozen-encoder training regime.

## Inference contract

1. Load the three modalities for one timestamp:
   - **RGB**: `PIL.Image.open(path).convert("RGB")`
   - **Depth**: `PIL.Image.open(path).convert("L")`
   - **Joints**: load the per-subject `.mat` file, extract `joints_gt[:, :, frame_idx]` → shape `(3, 14)` of `(x_pixel, y_pixel, occlusion)` per joint.
2. Apply the per-modality transforms in [`vision_config.json`](vision_config.json):
   - RGB / Depth: `Resize((224, 224))` → `ToTensor()` (yields `[0, 1]` float32). **No ImageNet mean/std** — the encoder was not trained with it.
   - Joints: per-frame, `x_norm = x_pixel / 576`, `y_norm = y_pixel / 1024`, occlusion kept as-is. Stack to `(14, 3)`, flatten to `(42,)`.
3. Add batch dim, move to device, forward → logits → argmax → label via [`class_names.json`](class_names.json).

## Training data

- **Dataset**: SLP-2022 (Simultaneously-collected multi-modal Lying Pose), `danaLab` split.
- **Modalities**: RGB images, depth images, and 2D joint annotations (`joints_gt`, 14 joints × {x, y, occlusion}).
- **Image dimensions**: 576 (W) × 1024 (H) at source — used for joint normalization.
- **Split**: subject-wise random split (seed=42), 70 / 15 / 15. Test subjects are held-out, never seen during training.
- **Train / Val / Test samples**: 9,585 / 2,025 / 2,160. Test set has 720 segments per posture class (perfectly balanced).

## Training recipe (for reproducibility)

- Loss: `nn.CrossEntropyLoss` (no class weighting — classes are balanced).
- Optimizer: Adam over trainable parameters only (encoders frozen via `requires_grad = False`); LR = 1e-3, weight decay = 1e-4.
- Batch size: 32. Epochs: 15. Best checkpoint selected by val macro-F1.
- No augmentation beyond resize + tensor conversion.
- Encoder checkpoints used at init (training-time only):
  - Depth: `experiments/artifacts/vision-spatial/encoders/depth_encoder_cnn/checkpoints/best_depth_encoder_only.pth`
  - RGB: `experiments/artifacts/vision-spatial/encoders/rgb_encoder_cnn/checkpoints/best_rgb_encoder.pt`
  - Joint: `experiments/artifacts/vision-spatial/encoders/joint_xyo/checkpoints/joint_encoder_xyo_RGB.pth`

## Test metrics (held-out subjects)

| Metric | Value |
|---|---|
| Accuracy | 0.9981 |
| Macro F1 | 0.9981 |
| Test loss (CE) | 0.0106 |
| Best epoch | 1 (val F1 plateaued at 1.0 from epoch 1) |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| supine | 0.9958 | 0.9986 | 0.9972 | 720 |
| left | 0.9986 | 0.9958 | 0.9972 | 720 |
| right | 1.0000 | 1.0000 | 1.0000 | 720 |

## Provenance and design notes

- **Source notebook**: [`experiments/notebooks/vision-spatial/fusion_experiments/attention_fusion_experiment.ipynb`](../../experiments/notebooks/vision-spatial/fusion_experiments/attention_fusion_experiment.ipynb)
- **Source artifacts (training-time)**: [`experiments/artifacts/vision-spatial/fusion/attention_fusion/`](../../experiments/artifacts/vision-spatial/fusion/attention_fusion/)
- **Fusion ablation**: attention fusion was selected over MLP fusion ([`MLP_experiment.ipynb`](../../experiments/notebooks/vision-spatial/fusion_experiments/MLP_experiment.ipynb)) and gated fusion ([`gated_fusion_experiment.ipynb`](../../experiments/notebooks/vision-spatial/fusion_experiments/gated_fusion_experiment.ipynb)).
- **Stage-2 finetune attempted and rejected.** A second-stage finetune (`best_attention_fusion_finetuned.pth`) unfroze RGB `layer4` and the last two depth-encoder modules. Test acc dropped from 0.9981 → 0.9963 — within noise but a methodological argument against finetuning given near-saturated performance with frozen encoders. The frozen version is what's promoted here.
- **Class imbalance**: none. SLP-2022 is balanced across postures (720 / 720 / 720 in test).
- **Why `norm_first=True` and a single transformer layer**: empirically chosen. With only 3 tokens (one per modality), one self-attention layer is sufficient to learn cross-modality interactions. Pre-norm helps stability with the small batch.

## Limitations

- **Domain**: trained on SLP-2022 lab data with controlled camera position, lighting, and bedding. Performance on consumer-grade depth sensors, oblique camera angles, or heavy bedding occlusion is untested.
- **Joint dependency at inference**: the model requires a 14-joint estimate at inference. If your runtime doesn't have a joint estimator (e.g., MediaPipe / OpenPose), this model is not usable as-is — you'd need to add a joint extractor to the pipeline.
- **2D joints only**: the joint encoder is trained on 2D `(x, y, occlusion)`. The `_xyo` variant was selected over a `_xyo_3d` alternative.
- **No ImageNet normalization** at inference — easy to get wrong if reusing other ResNet-based code. The wrong normalization will silently degrade accuracy.
- **Saturated benchmark**: 99.81% test accuracy on 2,160 segments suggests the dataset's subject-wise split is comparatively easy for this modality combination. Real-world posture distribution may differ.
- **Class layout fixed at 3**: supine / left / right. Prone, sitting, or lateral-decubitus variants are out of scope.
