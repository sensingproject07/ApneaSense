# ApneaSense

A multi-modal sleep apnea risk screening system that infers apnea events and sleep posture from a single overnight video.

**Audio branch** — ResNet18 on log-Mel spectrograms, binary apnea/non-apnea classification over 10-second windows.
Macro F1 **0.777** on PSG-Audio (192 subjects, 17,958 held-out segments).

**Vision-spatial branch** — self-attention fusion of RGB, depth, and 2D body-joint encoders, three-way sleep posture classification (supine / left / right).
Macro F1 **0.998** under clinical conditions (SLP-2022, 2,160 test segments).
Consumer mode (synthetic depth + YOLO joints + fine-tuned checkpoint) reaches macro F1 **0.985**.

**Late fusion** — deterministic posture-conditioned threshold with two-window hysteresis; a learned fusion head was not feasible as PSG-Audio and SLP-2022 share no overlapping subjects or recordings.

Delivered as a **Streamlit application** with clinical and consumer pipelines.

---

## Contributors

| Name | Affiliation |
|---|---|
| Arshi Saxena | NUS-ISS MTech AIS |
| Fia Thottan | NUS MSc SIDT |
| John Joseph Peter | NUS-ISS MTech AIS |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/sensingproject07/ApneaSense.git
cd ApneaSense

# 2. Create virtual environment and install dependencies
# Windows
.\setup_env.bat

# macOS / Linux
bash setup_env.sh

# 3. Install PyTorch (GPU recommended)
# Pick the command for your system at https://pytorch.org/get-started/locally/

# 4. Launch the app
.venv\Scripts\streamlit run app.py        # Windows
.venv/bin/streamlit run app.py            # macOS / Linux
```

> **VS Code users**: opening a terminal inside the project auto-activates `.venv` via `.vscode/settings.json`. You can run `streamlit run app.py` directly.

---

## Model Weights

Three checkpoints are required. Place them at the paths below (relative to the repo root):

| File | Purpose | Path |
|---|---|---|
| `mel_cnn.pth` | Audio apnea detector | `models/audio/mel_cnn.pth` |
| `attention_fusion.pth` | Clinical vision model | `models/vision-spatial/attention_fusion.pth` |
| `best_consumer_attention_fusion.pth` | Consumer vision model | `models/vision-spatial/best_consumer_attention_fusion.pth` |

---

## Inference Modes

### Clinical mode
Accepts an RGB video + real depth (folder of PNGs or depth video) + joint annotations (SLP `.mat` or CSV).
Uses `models/audio/mel_cnn.pth` for audio and `models/vision-spatial/attention_fusion.pth` for vision.

**Input requirements**
- RGB video: any format readable by OpenCV (MP4, AVI, ...)
- Depth: folder of `image_XXXXXX.png` files (1-indexed, SLP convention) **or** a depth video
- Joints: SLP `joints_gt_RGB.mat` **or** CSV with columns `frame_idx, j0_x, j0_y, j0_occ, ...` (42 values per frame)

### Consumer mode
Accepts an RGB video only. Depth and joints are synthesised automatically:
1. Monocular depth via **Depth Anything v2 Small**
2. 2D joints via **YOLOv8n-pose** with COCO-17 to SLP-14 keypoint remapping
3. Body-normalised inverted depth preprocessing (2nd-98th percentile, crop to joint bounding box)

Uses `models/audio/mel_cnn.pth` for audio and `models/vision-spatial/best_consumer_attention_fusion.pth` for vision.

### Output
Both modes produce:
- Annotated video with per-frame posture overlay and per-window verdict banner
- Per-window verdict timeline plot
- Per-window summary table (verdict, posture, audio probability, posture confidence)

---

## Project Structure

```
ApneaSense/
├── app.py                        # Streamlit application
├── inference/
│   ├── config.py                 # All paths, thresholds, preprocessing constants
│   ├── audio.py                  # Audio extraction, mel features, MelCNN inference
│   ├── vision.py                 # Depth synthesis, YOLO joints, frame inference
│   ├── fusion.py                 # Posture aggregation, posture-conditioned fusion
│   ├── pipeline.py               # consumer_pipeline() and clinical_pipeline()
│   └── models.py                 # DepthEncoder, RGBEncoder, JointEncoder, AttentionFusionClassifier
├── models/
│   ├── audio/
│   │   └── mel_cnn.pth
│   └── vision-spatial/
│       ├── attention_fusion.pth
│       └── best_consumer_attention_fusion.pth
├── experiments/
│   ├── notebooks/                # Training and evaluation notebooks
│   │   ├── audio/
│   │   └── vision-spatial/
│   └── artifacts/                # Saved checkpoints and result JSONs/CSVs
├── demo/                         # Demo video assets
├── requirements.txt
├── setup_env.bat / setup_env.sh
```

---

## Datasets

### PSG-Audio
Korompili et al., *Scientific Data* 2021.
192 subjects, full-night PSG recordings with synchronised over-the-bed microphone.
103,210 labelled 10-second segments (62.6% apnea / 37.4% non-apnea).
Download: [PhysioNet](https://physionet.org/content/psg-audio/1.0.0/)

### SLP-2022
Liu et al., WACV 2022.
107 subjects, three lying postures x three bedding-cover conditions.
Synchronised RGB (576x1024), Kinect depth, and 14-joint LSP annotations.
Download: [SLP Dataset](https://web.northeastern.edu/ostadabbas/2019/06/27/multimodal-in-bed-pose-estimation/)

Place datasets at the paths referenced in the experiment notebooks, or update the path variables at the top of each notebook.

---

## Key Results

### Audio (PSG-Audio test set, 17,958 segments)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| non-apnea | 0.736 | 0.665 | 0.699 |
| apnea | 0.834 | 0.876 | 0.855 |
| **Macro** | **0.785** | **0.770** | **0.777** |

### Vision — deployment trajectory (SLP-2022 test set, 2,160 segments)

| Depth source | Joint source | F1 | Acc |
|---|---|---|---|
| Real (Kinect) | GT (.mat) | 0.998 | 0.998 |
| Synthetic (DA-v2) | GT | 0.875 | 0.881 |
| Synthetic (DA-v2) | MediaPipe | 0.802 | 0.800 |
| Synthetic (DA-v2) | YOLOv8n-pose | 0.903 | 0.903 |
| Synthetic (DA-v2) | YOLOv8n-pose + fine-tune | **0.985** | **0.985** |

---

## Dependencies

Key packages (see `requirements.txt` for the full list):

- `torch` / `torchaudio` — deep learning backbone
- `ultralytics` — YOLOv8n-pose joint estimation (auto-downloads model weights on first run)
- `streamlit` — application UI
- `librosa` — audio loading
- `opencv-python` — video I/O
- `scipy` — `.mat` joint file loading (clinical mode)

PyTorch is **not** listed in `requirements.txt` because the correct version depends on your CUDA version. Install it separately before running `setup_env`.
