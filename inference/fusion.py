"""
Deterministic late fusion of audio apnea decisions and vision posture annotations.

Strategy:
  1. Aggregate per-frame posture predictions into per-audio-window majority votes.
  2. Apply posture-conditioned audio threshold:
       - vision says supine (conf >= SUPINE_CONF_GATE) → threshold = 0.4  (more sensitive)
       - all other cases                                → threshold = 0.5
  3. Apply hysteresis: ≥ HYSTERESIS_MIN_CONSECUTIVE consecutive positive windows
     → confirmed apnea.  A single positive window is flagged as 'suspected'.
"""
from collections import Counter
from typing import List, Dict

from .config import (
    AUDIO_THRESHOLD_DEFAULT, AUDIO_THRESHOLD_SUPINE,
    SUPINE_CONF_GATE, LATERAL_CONF_GATE, VISION_CONF_GATE,
    HYSTERESIS_MIN_CONSECUTIVE, WINDOW_DURATION,
)


def aggregate_posture_per_window(
    frame_results: List[Dict],
    fps: float,
    window_duration: int = WINDOW_DURATION,
) -> List[Dict]:
    """
    For each audio window [start_s, end_s), collect all frames whose timestamp
    falls in that range and majority-vote on posture label.

    Args:
        frame_results : list of {frame_idx, timestamp_s, posture_label, posture_conf, ...}
        fps           : video frames-per-second (used only if needed; timestamps are used directly)
        window_duration: audio window length in seconds (default 10)

    Returns:
        list of {window_idx, posture_label, posture_conf, reliable, frame_count}
    """
    if not frame_results:
        return []

    max_ts    = max(r["timestamp_s"] for r in frame_results)
    n_windows = int(max_ts // window_duration) + 1

    window_postures = []
    for w_idx in range(n_windows):
        start_s = w_idx * window_duration
        end_s   = start_s + window_duration

        frames_in = [r for r in frame_results if start_s <= r["timestamp_s"] < end_s]

        if not frames_in:
            window_postures.append({
                "window_idx":    w_idx,
                "posture_label": "uncertain",
                "posture_conf":  0.0,
                "reliable":      False,
                "frame_count":   0,
            })
            continue

        label_counts   = Counter(f["posture_label"] for f in frames_in)
        majority_label = label_counts.most_common(1)[0][0]
        majority_confs = [f["posture_conf"] for f in frames_in
                          if f["posture_label"] == majority_label]
        mean_conf = sum(majority_confs) / len(majority_confs)

        window_postures.append({
            "window_idx":    w_idx,
            "posture_label": majority_label,
            "posture_conf":  mean_conf,
            "reliable":      mean_conf >= VISION_CONF_GATE,
            "frame_count":   len(frames_in),
        })

    return window_postures


def apply_fusion(
    audio_results: List[Dict],
    window_postures: List[Dict],
) -> List[Dict]:
    """
    Merge per-window audio and posture results into a fused verdict per window.

    Returns a list of dicts:
        window_idx, start_s, end_s,
        apnea_prob, threshold_used,
        posture_label, posture_conf, posture_reliable,
        confirmed_apnea,        # True  = ≥2 consecutive positives
        verdict                 # 'apnea' | 'suspected' | 'non-apnea'
    """
    posture_by_window = {p["window_idx"]: p for p in window_postures}

    # ── Step 1: posture-conditioned threshold ──────────────────────────────────
    thresholded = []
    for w in audio_results:
        posture  = posture_by_window.get(w["window_idx"], {})
        label    = posture.get("posture_label", "uncertain")
        conf     = posture.get("posture_conf",  0.0)
        reliable = posture.get("reliable",      False)

        if label in ("left", "right") and conf >= LATERAL_CONF_GATE:
            # Reliably non-supine: apnea events are clinically unlikely; suppress detection
            threshold = 1.1   # unreachable by any probability in [0, 1]
        elif label == "supine" and conf >= SUPINE_CONF_GATE:
            threshold = AUDIO_THRESHOLD_SUPINE
        else:
            threshold = AUDIO_THRESHOLD_DEFAULT
        thresholded.append({
            **w,
            "threshold_used":    threshold,
            "apnea_thresholded": int(w["apnea_prob"] >= threshold),
            "posture_label":     label,
            "posture_conf":      conf,
            "posture_reliable":  reliable,
        })

    # ── Step 2: hysteresis ─────────────────────────────────────────────────────
    n = len(thresholded)
    verdicts = []
    for i, w in enumerate(thresholded):
        confirmed = False
        if w["apnea_thresholded"] == 1:
            # Count consecutive positives starting at i
            run = sum(
                1 for j in range(i, min(i + HYSTERESIS_MIN_CONSECUTIVE, n))
                if thresholded[j]["apnea_thresholded"] == 1
            )
            confirmed = run >= HYSTERESIS_MIN_CONSECUTIVE

        verdict = (
            "apnea"     if confirmed
            else "suspected" if w["apnea_thresholded"] == 1
            else "non-apnea"
        )
        verdicts.append({**w, "confirmed_apnea": confirmed, "verdict": verdict})

    return verdicts
