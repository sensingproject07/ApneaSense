from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Model weights ──────────────────────────────────────────────────────────────
AUDIO_MODEL_PATH  = PROJECT_ROOT / "models" / "audio" / "mel_cnn.pth"
VISION_MODEL_PATH = PROJECT_ROOT / "models" / "vision-spatial" / "attention_fusion.pth"
CONSUMER_VISION_MODEL_PATH = (
    PROJECT_ROOT / "models" / "vision-spatial" / "best_consumer_attention_fusion.pth"
)
YOLO_POSE_MODEL_PATH = "yolov8n-pose.pt"

# ── Class names ────────────────────────────────────────────────────────────────
AUDIO_CLASS_NAMES  = ["non-apnea", "apnea"]
VISION_CLASS_NAMES = ["supine", "left", "right"]

# ── Audio preprocessing (matches mel_config.json / training) ──────────────────
SAMPLE_RATE        = 16000
WINDOW_DURATION    = 10            # seconds
SAMPLES_PER_WINDOW = SAMPLE_RATE * WINDOW_DURATION
N_MELS             = 128
N_FFT              = 1024
HOP_LENGTH         = 512
F_MIN              = 20.0
F_MAX              = 8000.0
DB_FLOOR           = -100.0
DB_CEIL            = 0.0
POWER_FLOOR        = 1e-10
MEL_TIME_FRAMES    = 313           # expected time-axis length after mel transform

# ── Vision preprocessing (matches vision_config.json / training) ───────────────
IMG_W              = 576.0         # SLP image width  — for joint x-normalisation
IMG_H              = 1024.0        # SLP image height — for joint y-normalisation
VISION_INPUT_SIZE  = (224, 224)    # resize target for encoder

# ── Fusion thresholds ──────────────────────────────────────────────────────────
AUDIO_THRESHOLD_DEFAULT     = 0.5  # standard apnea-probability threshold
AUDIO_THRESHOLD_SUPINE      = 0.4  # lower threshold when vision confirms supine
SUPINE_CONF_GATE            = 0.6  # min posture confidence to apply supine threshold
LATERAL_CONF_GATE           = 0.6  # min confidence to suppress apnea for left/right postures
VISION_CONF_GATE            = 0.6  # min confidence to mark posture annotation as reliable
HYSTERESIS_MIN_CONSECUTIVE  = 2    # consecutive positive windows → confirmed apnea

# ── MediaPipe pose model (downloaded by synthetic_joints_eval.ipynb) ───────────
MEDIAPIPE_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments" / "artifacts" / "vision-spatial"
    / "inference_experiments" / "synthetic_joints_eval"
    / "pose_landmarker_heavy.task"
)

# Consumer posture preprocessing. These match the best synthetic-depth
# experiment and the consumer fine-tune checkpoint.
CONSUMER_DEPTH_PREPROCESS = "body-norm-invert"
CONSUMER_BODY_PADDING = 0.0
CONSUMER_DEPTH_NORM_PERCENTILES = (2.0, 98.0)
YOLO_CONF_THRESHOLD = 0.2
