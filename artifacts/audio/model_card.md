# Model Card — Audio Apnea Detector (mel_cnn)

## Overview

Binary classifier that predicts apnea / non-apnea from a 10-second audio segment. Used as the audio modality's contribution to the multi-modal pipeline (the encoder feeds downstream fusion; the classifier head can also be used standalone).

## Files

| File | Path | Purpose |
|---|---|---|
| Weights | `models/audio/mel_cnn.pth` | PyTorch state-dict checkpoint |
| Preprocessing config | `artifacts/audio/mel_config.json` | Mel-spectrogram + dB-window parameters; must match training |
| Label map | `artifacts/audio/class_names.json` | Logit-index → class-name |

## Architecture

- **Backbone**: ResNet18, ImageNet-pretrained. `conv1` reinitialised from 3-channel to 1-channel by averaging the pretrained kernel.
- **Input adapter**: `BatchNorm2d(1)` before the encoder.
- **Classifier head**: `Dropout(0.6)` → `Linear(512, 2)`.
- **Trainable parameters**: ~11.17 M.
- **Input**: `(batch, 1, 128, 313)` float32 — log-Mel spectrogram, dB-normalised to `[0, 1]`.
- **Output**: `(batch, 2)` raw logits. Apply softmax for probabilities, argmax for class.

The checkpoint contains `model_state_dict`, `encoder_state_dict`, and `classifier_state_dict` separately — fusion code that needs only the encoder can load `encoder_state_dict` directly.

## Inference contract

1. Load 10-second 16 kHz mono audio segment → `(160000,)` float32.
2. Compute Mel spectrogram with the parameters in `mel_config.json` (n_mels=128, n_fft=1024, hop=512, f_min=20, f_max=8000).
3. Convert to dB with a 1e-10 power floor; clamp to `[-100, 0]` dB; linearly map to `[0, 1]`.
4. Add channel and batch dims → `(1, 1, 128, 313)`.
5. Forward through the model → logits → argmax → label via `class_names.json`.

## Training data

- **Dataset**: PSG-Audio (192 subjects, 103,210 segments, 16 kHz, 10-second windows).
- **Split**: subject-wise stratified by per-subject apnea fraction (4-quartile stratification, seed=42), 70 / 15 / 15. Test set is held-out subjects, never seen during training.
- **Train / Val / Test segments**: 70,430 / 14,822 / 17,958.

## Training recipe (for reproducibility)

- Loss: focal loss, γ = 2.0, class-weighted α = `[1.3112, 0.8082]` (inverse frequency on the binary distribution).
- Optimizer: AdamW, LR = 1e-4, weight decay = 1e-3.
- Scheduler: `ReduceLROnPlateau` on val macro-F1, factor = 0.5, patience = 2.
- Augmentation: GPU SpecAugment (2 freq + 2 time masks per batch; freq-mask = 32 of 128 bins, time-mask = 40 frames).
- Schedule: up to 15 epochs, early-stop after 5 stagnant epochs on val macro-F1.
- bf16 autocast on Tensor Cores.

## Test metrics (held-out subjects)

| Metric | Value |
|---|---|
| Accuracy | 0.8038 |
| Macro F1 | 0.7766 |
| Loss (unweighted CE) | 0.4558 |
| Loss (focal, training objective) | 0.1695 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| non-apnea | 0.7361 | 0.6649 | 0.6987 | 6,145 |
| apnea | 0.8340 | 0.8760 | 0.8545 | 11,813 |

Best epoch: 7 of 12 (early stopped). Best val macro-F1: 0.7932, best val acc: 0.8058.

## Provenance and limitations

- **Source notebook**: [`experiments/notebooks/audio/feature_representation/Mel/mel_cnn.ipynb`](../../experiments/notebooks/audio/feature_representation/Mel/mel_cnn.ipynb)
- **Source artifacts (training-time)**: [`experiments/artifacts/audio/feature_representation/mel_cnn/`](../../experiments/artifacts/audio/feature_representation/mel_cnn/)
- **Feature ablation**: Mel + ResNet18 won the comparison against MFCC + ResNet18 (F1 0.7127), Mel + PANN Cnn14 (F1 0.7535), and raw waveform + 1D CNN (F1 0.6288). AST was abandoned on hardware grounds.
- **Sub-typing limit**: a separate experiment ([`classification_task.ipynb`](../../experiments/notebooks/audio/classification_task/classification_task.ipynb)) showed that audio reliably detects CSA but cannot separate OSA from Mixed/Hypopnea. Production audio task is therefore binary, by design.
- **Domain**: trained on clinical polysomnography audio. Performance on consumer-grade microphones (laptop / phone / smart-speaker) is untested — domain shift is likely.
- **Window**: fixed 10-second segments. Apnea events shorter than ~5 s or longer than 15 s may be mis-segmented by upstream code.
- **Single channel**: model accepts mono audio only. Stereo input must be downmixed first.
