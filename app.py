"""
ApneaSense — Streamlit inference UI.

Tabs:
  Consumer Mode  : upload RGB+audio video → synthetic depth & joints
  Clinical Mode  : upload RGB video + depth (folder ZIP or video) + joints (.mat or CSV)
"""
import io
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from inference import consumer_pipeline, clinical_pipeline

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ApneaSense",
    page_icon="🫁",
    layout="wide",
)

# ── Constants ──────────────────────────────────────────────────────────────────

_POSTURE_COLOR = {"supine": "#e74c3c", "left": "#2ecc71", "right": "#3498db", "uncertain": "#95a5a6"}
_VERDICT_COLOR = {"apnea": "#e74c3c", "suspected": "#e74c3c", "non-apnea": "#2ecc71"}
_VERDICT_LABEL = {"apnea": "HIGH", "suspected": "HIGH", "non-apnea": "LOW"}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_ffmpeg():
    import shutil
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _generate_annotated_video(
    video_path: str,
    frame_results: list,
    verdicts: list,
    out_path: str,
    progress_cb=None,
) -> str:
    """Burn posture + verdict overlays into every frame and write an H.264 MP4."""
    import subprocess

    frame_map   = {f["frame_idx"]: f for f in frame_results}
    verdict_map = {(v["start_s"], v["end_s"]): v for v in verdicts}

    def _window_for_ts(ts):
        for v in verdicts:
            if v["start_s"] <= ts < v["end_s"]:
                return v
        return None

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    tmp_path = out_path + "_raw.mp4"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    vw       = cv2.VideoWriter(tmp_path, fourcc, fps, (W, H))

    idx = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        fr = frame_map.get(idx)
        if fr:
            win     = _window_for_ts(fr["timestamp_s"])
            verdict = win["verdict"]       if win else "non-apnea"
            ap_prob = win["apnea_prob"]    if win else None
            thr     = win["threshold_used"] if win else None
            pil     = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            pil     = _annotate_frame(
                pil, fr["posture_label"], fr["posture_conf"],
                verdict, ap_prob, thr,
            )
            bgr     = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        vw.write(bgr)
        idx += 1
        if progress_cb and idx % 50 == 0:
            progress_cb(idx / total)

    cap.release()
    vw.release()

    # Re-encode to H.264 + yuv420p and carry audio from original video
    ff = _find_ffmpeg()
    if ff:
        subprocess.run(
            [
                ff, "-y",
                "-i", tmp_path,        # annotated video (no audio)
                "-i", video_path,      # original video (audio source)
                "-map", "0:v:0",
                "-map", "1:a:0?",      # optional: original audio if present
                "-c:v", "libx264",
                "-c:a", "aac",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                out_path,
            ],
            capture_output=True,
        )
        Path(tmp_path).unlink(missing_ok=True)
    else:
        Path(tmp_path).rename(out_path)

    return out_path


def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _annotate_frame(
    pil_img: Image.Image,
    posture_label: str,
    posture_conf: float,
    verdict: str,
    apnea_prob: float = None,
    threshold_used: float = None,
) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    W, H = img.size

    try:
        font = ImageFont.truetype("arial.ttf", max(14, H // 30))
    except Exception:
        font = ImageFont.load_default()

    # Top-left: posture (vision)
    p_text  = f"{posture_label.upper()}  {posture_conf:.0%}"
    p_color = _hex_to_rgb(_POSTURE_COLOR.get(posture_label, "#95a5a6"))
    _draw_label(draw, p_text, (8, 8), p_color, font)

    # Top-right: final verdict
    v_text  = f"RISK: {_VERDICT_LABEL.get(verdict, verdict.upper())}"
    v_color = _hex_to_rgb(_VERDICT_COLOR.get(verdict, "#95a5a6"))
    bbox    = draw.textbbox((0, 0), v_text, font=font)
    tw      = bbox[2] - bbox[0]
    _draw_label(draw, v_text, (W - tw - 16, 8), v_color, font)

    # Bottom-left: audio result
    if apnea_prob is not None:
        suppressed = threshold_used is not None and threshold_used >= 1.0
        if suppressed:
            a_text    = f"AUDIO: {apnea_prob:.2f}  [posture suppressed]"
            audio_color = _hex_to_rgb(_VERDICT_COLOR["non-apnea"])
        else:
            audio_verdict = "APNEA" if apnea_prob >= (threshold_used or 0.5) else "NON-APNEA"
            thr_str   = f"  thr:{threshold_used:.2f}" if threshold_used else ""
            a_text    = f"AUDIO: {apnea_prob:.2f} -> {audio_verdict}{thr_str}"
            audio_color = _hex_to_rgb(_VERDICT_COLOR.get(
                "apnea" if audio_verdict == "APNEA" else "non-apnea", "#95a5a6"
            ))
        bbox = draw.textbbox((0, 0), a_text, font=font)
        bh   = bbox[3] - bbox[1]
        _draw_label(draw, a_text, (8, H - bh - 28), audio_color, font)

    return img


def _draw_label(draw, text, xy, rgb_fill, font):
    x, y = xy
    bbox  = draw.textbbox((x, y), text, font=font)
    pad   = 4
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=(*rgb_fill, 180),
    )
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def _build_timeline_fig(verdicts: list, window_postures: list) -> plt.Figure:
    n = len(verdicts)
    posture_by_win = {p["window_idx"]: p["posture_label"] for p in window_postures}

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 2.2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    for v in verdicts:
        x     = v["start_s"]
        w     = v["end_s"] - v["start_s"]
        color = _VERDICT_COLOR.get(v["verdict"], "#95a5a6")
        ax.barh(0, w, left=x, height=0.6, color=color, edgecolor="#1a1a2e", linewidth=0.5)

        # Posture initial below bar
        posture = posture_by_win.get(v["window_idx"], "?")[0].upper()
        ax.text(
            x + w / 2, -0.45, posture,
            ha="center", va="top", fontsize=8,
            color=_POSTURE_COLOR.get(posture_by_win.get(v["window_idx"], "uncertain"), "#95a5a6"),
        )
        # Apnea prob above bar
        ax.text(
            x + w / 2, 0.35, f'{v["apnea_prob"]:.2f}',
            ha="center", va="bottom", fontsize=6.5, color="white",
        )

    ax.set_xlim(0, verdicts[-1]["end_s"] if verdicts else 10)
    ax.set_ylim(-0.8, 0.8)
    ax.set_xlabel("Time (s)", color="white", fontsize=9)
    ax.set_yticks([])
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333")

    legend_patches = [
        mpatches.Patch(color=_VERDICT_COLOR["apnea"],     label="High Risk"),
        mpatches.Patch(color=_VERDICT_COLOR["non-apnea"], label="Low Risk"),
    ]
    ax.legend(
        handles=legend_patches, loc="upper right",
        framealpha=0.3, labelcolor="white", fontsize=8,
    )
    fig.tight_layout(pad=0.5)
    return fig


def _show_results(state_key: str):
    """Render annotated video, then summary metrics and timeline."""
    res      = st.session_state[f"{state_key}_results"]
    vid_path = st.session_state[f"{state_key}_video_path"]

    verdicts        = res["verdicts"]
    frame_results   = res["frame_results"]
    window_postures = res["window_postures"]

    if res["n_frames"] == 0:
        st.info("No frames extracted.")
        return

    # ── Annotated video ────────────────────────────────────────────────────────
    ann_key = f"{state_key}_annotated_video"
    if ann_key not in st.session_state:
        ann_path = tempfile.mktemp(suffix="_annotated.mp4")
        pbar = st.progress(0.0, text="Generating annotated video…")
        _generate_annotated_video(
            vid_path, frame_results, verdicts, ann_path,
            progress_cb=lambda p: pbar.progress(min(p, 1.0)),
        )
        pbar.empty()
        st.session_state[ann_key] = ann_path

    ann_path = st.session_state[ann_key]
    st.markdown("### Analysis Video")
    st.caption("Posture label (top-left) and verdict (top-right) overlaid on every frame.")
    # Portrait clinical video (9:16) displayed in a narrow column so its height
    # matches a landscape consumer video (16:9) shown at half-page width.
    mode = res.get("mode", "consumer")
    col_ratio = [1, 4] if mode == "clinical" else [1, 1]
    vid_col, _ = st.columns(col_ratio)
    with vid_col:
        with open(ann_path, "rb") as f:
            st.video(f.read())

    # ── Overall risk banner ────────────────────────────────────────────────────
    n_apnea = sum(1 for v in verdicts if v["verdict"] in ("apnea", "suspected"))
    n_ok    = sum(1 for v in verdicts if v["verdict"] == "non-apnea")
    duration_s   = res["n_windows"] * 10
    n_total      = len(verdicts)
    risk_pct     = (n_apnea / n_total * 100) if n_total > 0 else 0

    if risk_pct == 0:
        severity, banner_color = "No Apnea Detected",      "#2ecc71"
    elif risk_pct < 25:
        severity, banner_color = "Low Apnea Activity",     "#f39c12"
    elif risk_pct < 50:
        severity, banner_color = "Moderate Apnea Activity","#e67e22"
    else:
        severity, banner_color = "High Apnea Activity",    "#e74c3c"

    st.markdown(
        f"""<div style="background:{banner_color}22; border-left:4px solid {banner_color};
        padding:12px 16px; border-radius:6px; margin-bottom:12px;">
        <span style="font-size:1.2em; font-weight:700; color:{banner_color};">{severity}</span>
        &nbsp;&nbsp;
        <span style="color:#ccc;">
        <b style="color:white;">{n_apnea}</b> of <b style="color:white;">{n_total}</b> segments
        flagged &nbsp;·&nbsp; {risk_pct:.0f}% of {duration_s/60:.1f} min recording
        </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Summary metrics ────────────────────────────────────────────────────────
    posture_counts = {}
    for p in window_postures:
        lbl = p["posture_label"]
        posture_counts[lbl] = posture_counts.get(lbl, 0) + 1

    st.markdown("### Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segments", len(verdicts))
    c2.metric("High Risk", n_apnea)
    c3.metric("Low Risk", n_ok)
    c4.metric("Duration", f"{duration_s}s")

    st.markdown("**Posture breakdown (segments)**")
    pcols = st.columns(len(posture_counts) or 1)
    for col, (lbl, cnt) in zip(pcols, posture_counts.items()):
        col.metric(lbl.capitalize(), cnt)

    # ── Timeline ───────────────────────────────────────────────────────────────
    st.markdown("### Segment Timeline")
    st.caption("Bar color = verdict | number = apnea probability | letter = posture (S/L/R)")
    if verdicts:
        fig = _build_timeline_fig(verdicts, window_postures)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ── Consumer tab ───────────────────────────────────────────────────────────────

def _consumer_tab():
    st.markdown("Upload a standard RGB video (MP4, AVI, MOV). Depth and body joints are "
                "synthesised automatically using Depth Anything v2 and MediaPipe.")

    video_file = st.file_uploader(
        "Video file (RGB + audio)", type=["mp4", "avi", "mov", "mkv"],
        key="consumer_upload",
    )

    run = st.button("Run Consumer Pipeline", key="consumer_run", type="primary",
                    disabled=video_file is None)

    if run and video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video_file.name).suffix) as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        st.session_state["consumer_video_path"] = tmp_path

        progress_bar = st.progress(0, text="Starting…")

        _steps       = [0]

        def _cb(msg: str):
            _steps[0] = min(_steps[0] + 1, 10)
            progress_bar.progress(_steps[0] / 10, text=msg)


        try:
            results = consumer_pipeline(tmp_path, progress_cb=_cb)
            st.session_state["consumer_results"] = results
            progress_bar.progress(1.0, text="Done!")

            st.success("Inference complete.")
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            return

    if "consumer_results" in st.session_state:
        _show_results("consumer")


# ── Clinical tab ───────────────────────────────────────────────────────────────

def _clinical_tab():
    st.markdown(
        "Upload the RGB video, a depth source (folder ZIP of PNGs **or** depth video), "
        "and joint annotations (SLP `.mat` file **or** CSV)."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        video_file = st.file_uploader(
            "RGB video", type=["mp4", "avi", "mov", "mkv"], key="clinical_vid"
        )

    with col2:
        depth_file = st.file_uploader(
            "Depth: ZIP of PNGs  or  depth video",
            type=["zip", "mp4", "avi"], key="clinical_depth",
        )

    with col3:
        joints_file = st.file_uploader(
            "Joints: .mat  or  .csv", type=["mat", "csv"], key="clinical_joints"
        )

    run = st.button(
        "Run Clinical Pipeline", key="clinical_run", type="primary",
        disabled=not (video_file and depth_file and joints_file),
    )

    if run and video_file and depth_file and joints_file:
        tmp_dir = Path(tempfile.mkdtemp())

        # Save RGB video
        vid_path = tmp_dir / Path(video_file.name).name
        vid_path.write_bytes(video_file.read())

        # Save / unzip depth
        depth_name = Path(depth_file.name)
        if depth_name.suffix == ".zip":
            zip_path = tmp_dir / depth_name.name
            zip_path.write_bytes(depth_file.read())
            depth_out = tmp_dir / "depth_frames"
            depth_out.mkdir()
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(depth_out)
            depth_source = str(depth_out)
        else:
            depth_vid_path = tmp_dir / depth_name.name
            depth_vid_path.write_bytes(depth_file.read())
            depth_source = str(depth_vid_path)

        # Save joints file
        joints_path = tmp_dir / Path(joints_file.name).name
        joints_path.write_bytes(joints_file.read())

        st.session_state["clinical_video_path"] = str(vid_path)

        progress_bar = st.progress(0, text="Starting…")

        _steps       = [0]

        def _cb(msg: str):
            _steps[0] = min(_steps[0] + 1, 10)
            progress_bar.progress(_steps[0] / 10, text=msg)


        try:
            results = clinical_pipeline(
                str(vid_path), depth_source, str(joints_path), progress_cb=_cb
            )
            st.session_state["clinical_results"] = results
            progress_bar.progress(1.0, text="Done!")

            st.success("Inference complete.")
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            return

    if "clinical_results" in st.session_state:
        _show_results("clinical")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.title("ApneaSense")
    st.caption("Sleep apnea detection via multimodal audio + vision-spatial inference")

    tab_consumer, tab_clinical = st.tabs(["Consumer Mode", "Clinical Mode"])

    with tab_consumer:
        _consumer_tab()

    with tab_clinical:
        _clinical_tab()


if __name__ == "__main__":
    main()
