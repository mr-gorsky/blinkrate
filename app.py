import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import math
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import plotly.graph_objects as go
import csv
import zipfile
from io import BytesIO

st.set_page_config(page_title="Pupil & Blink Analyzer (GC0308)", layout="wide", page_icon="👁️")


# ---------------------------------------------------------
# ----------- PUPIL DETECTOR (CLAHE + contour) ------------
# ---------------------------------------------------------
def detect_pupil(gray_roi):
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_roi)

    # blur
    blurred = cv2.medianBlur(enhanced, 5)

    # invert to make pupil bright
    inv = cv2.bitwise_not(blurred)

    # Adaptive threshold
    th = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

    # clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_score = 0
    best_diam = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < 8:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue

        circularity = 4 * np.pi * (area / (peri * peri))
        score = circularity * (area ** 0.60)   # emphasize roundish & reasonably large

        if score > best_score:
            best_score = score
            best_diam = np.sqrt((4 * area) / np.pi)

    return best_diam


# ---------------------------------------------------------
# ----------------- VIDEO ROTATION ------------------------
# ---------------------------------------------------------
def rotate_video_to_temp(input_path, rotation):
    """
    rotation: "none", "90cw", "90ccw", "180"
    Returns path to rotated temp video.
    """

    if rotation == "none":
        return input_path

    # create temporary output video file
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmp_out.name
    tmp_out.close()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return input_path

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # determine output dimensions
    if rotation in ("90cw", "90ccw"):
        out_w, out_h = h, w
    else:  # 180°
        out_w, out_h = w, h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h), True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if rotation == "90cw":
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == "90ccw":
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:  # 180
            rotated = cv2.rotate(frame, cv2.ROTATE_180)

        writer.write(rotated)

    cap.release()
    writer.release()

    return out_path


# ---------------------------------------------------------
# ----------- MINUTE-BY-MINUTE ANALYSIS -------------------
# ---------------------------------------------------------
def analyze_video(
    video_path,
    roi_mode,
    roi_offset_x,
    roi_offset_y,
    sample_fps,
    smoothing_sigma,
    min_blink_samples,
    threshold_method,
    threshold_param,
    auto_invert,
    max_minutes,
    progress_callback
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open rotated video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps
    total_minutes = math.ceil(duration / 60)

    if max_minutes and max_minutes > 0:
        total_minutes = min(total_minutes, max_minutes)

    # read first frame to set ROI
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read first frame")
    h, w = frame0.shape[:2]

    roi_w = int(w * 0.5)
    roi_h = int(h * 0.6)

    if roi_mode == "left":
        roi_x = int(w * roi_offset_x)
    elif roi_mode == "right":
        roi_x = max(0, w - roi_w - int(w * roi_offset_x))
    else:
        roi_x = int((w - roi_w) // 2 + w * roi_offset_x)

    roi_y = int((h - roi_h) // 2 + h * roi_offset_y)

    roi_x = max(0, min(roi_x, w - roi_w))
    roi_y = max(0, min(roi_y, h - roi_h))

    sample_step = max(1, int(round(fps / sample_fps)))

    # detect signal polarity using first ~5 seconds
    polarity = 1.0
    if auto_invert:
        quick_vals = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        idx = 0
        needed = max(10, int(5 * sample_fps))

        while len(quick_vals) < needed:
            ret, fr = cap.read()
            if not ret:
                break
            if idx % sample_step == 0:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                d = detect_pupil(roi)
                if d is not None:
                    quick_vals.append(d)
            idx += 1

        if len(quick_vals) > 5:
            q = np.array(quick_vals)
            # Heuristic: if mean>median -> dips represent blinks → invert
            if np.mean(q) > np.median(q):
                polarity = -1.0

    results = []

    # process minute-by-minute
    for minute in range(total_minutes):
        start_s = minute * 60
        start_f = int(start_s * fps)
        end_f = int(min((minute + 1) * 60 * fps, total_frames))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        metrics = []
        timestamps = []
        fi = start_f

        while fi < end_f:
            ret, fr = cap.read()
            if not ret:
                break

            if (fi - start_f) % sample_step == 0:
                gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

                d = detect_pupil(roi)
                metrics.append(d if d is not None else np.nan)
                timestamps.append(fi / fps)

            fi += 1

        vals = np.array(metrics, dtype=float)

        # interpolate NaNs
        if np.all(np.isnan(vals)):
            vals[:] = 0
        else:
            nans = np.isnan(vals)
            if np.any(nans):
                good = np.where(~nans)[0]
                vals[nans] = np.interp(np.where(nans)[0], good, vals[good])

        closure_raw = -polarity * vals

        if closure_raw.max() - closure_raw.min() > 1e-6:
            norm = (closure_raw - closure_raw.min()) / (closure_raw.max() - closure_raw.min())
        else:
            norm = np.zeros_like(closure_raw)

        smoothed = gaussian_filter1d(norm, sigma=smoothing_sigma) if len(norm) > 1 else norm

        # threshold
        if threshold_method == "percentile":
            thr = np.percentile(smoothed, threshold_param)
        elif threshold_method == "median_std":
            thr = np.median(smoothed) + threshold_param * np.std(smoothed)
        else:
            try:
                hist, bins = np.histogram(smoothed, bins=64)
                total = hist.sum()
                sum_total = (bins[:-1] * hist).sum()
                weight_b = 0
                sum_b = 0
                max_var = 0
                thresh = bins[0]
                for i in range(len(hist)):
                    weight_b += hist[i]
                    if weight_b == 0:
                        continue
                    weight_f = total - weight_b
                    if weight_f == 0:
                        break
                    sum_b += bins[i] * hist[i]
                    mean_b = sum_b / weight_b
                    mean_f = (sum_total - sum_b) / weight_f
                    var = weight_b * weight_f * (mean_b - mean_f)**2
                    if var > max_var:
                        max_var = var
                        thresh = bins[i]
                thr = thresh
            except:
                thr = np.percentile(smoothed, 90)

        min_dist = max(1, int(0.2 * sample_fps))
        peaks, _ = find_peaks(smoothed, height=thr, distance=min_dist)

        blink_times = [timestamps[p] for p in peaks]

        results.append({
            "minute": minute,
            "timestamps": timestamps,
            "signal": smoothed.tolist(),
            "threshold": float(thr),
            "blinks": blink_times,
            "blink_count": len(blink_times)
        })

        progress_callback(minute + 1, total_minutes)

    cap.release()

    return results, {"fps": fps, "duration": duration, "frames": total_frames}


# ---------------------------------------------------------
# ---------------------- STREAMLIT UI ----------------------
# ---------------------------------------------------------
def main():
    st.title("👁️ Pupil & Blink Analyzer — GC0308 (with rotation)")

    uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"])
    if not uploaded:
        st.info("Upload video to start.")
        return

    with st.expander("Video rotation", expanded=True):
        rotation = st.selectbox(
            "Select rotation",
            ["none", "90cw", "90ccw", "180"],
            format_func=lambda x: {
                "none": "No rotation",
                "90cw": "Rotate 90° clockwise",
                "90ccw": "Rotate 90° counter-clockwise",
                "180": "Rotate 180°"
            }[x]
        )

    with st.expander("Analysis settings", expanded=True):
        roi_mode = st.selectbox("ROI mode", ["left", "right", "auto"])
        roi_offset_x = st.slider("ROI offset X", -0.2, 0.2, 0.05, 0.01)
        roi_offset_y = st.slider("ROI offset Y", -0.2, 0.2, 0.0, 0.01)
        sample_fps = st.slider("Sample FPS", 4, 30, 10)
        smoothing_sigma = st.slider("Smoothing sigma", 0, 6, 2)
        min_blink_samples = st.slider("Min blink samples", 1, 6, 2)
        threshold_method = st.selectbox("Threshold method", ["percentile", "median_std", "otsu"])
        threshold_param = st.slider("Threshold parameter", 50, 99, 90) if threshold_method == "percentile" else \
                          st.slider("k", 0.1, 3.0, 1.2)
        auto_invert = st.checkbox("Auto-detect signal polarity", True)
        max_minutes = st.number_input("Max minutes (0 = full video)", min_value=0, value=0)

    # Save upload to temp
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        input_path = tmp.name

    # ROTATE FIRST
    st.write("Rotating video…")
    rotated_path = rotate_video_to_temp(input_path, rotation)
    st.success("Rotation done.")

    if st.button("Start analysis"):
        progress_text = st.empty()
        progress_bar = st.progress(0)

        def callback(done, total):
            progress_bar.progress(int(done / total * 100))
            progress_text.text(f"Processing minute {done}/{total}")

        results, meta = analyze_video(
            rotated_path,
            roi_mode, roi_offset_x, roi_offset_y,
            sample_fps,
            smoothing_sigma,
            min_blink_samples,
            threshold_method,
            threshold_param,
            auto_invert,
            max_minutes,
            callback
        )

        st.success("Analysis complete.")

        # SUMMARY TABLE
        st.subheader("Per-minute blink counts")
        mins = [r["minute"] for r in results]
        counts = [r["blink_count"] for r in results]
        st.table({"minute": mins, "blinks": counts})

        # GRAPH BLINK RATE
        fig_rate = go.Figure()
        fig_rate.add_trace(go.Bar(x=mins, y=counts))
        fig_rate.update_layout(title="Blink count per minute", xaxis_title="Minute", yaxis_title="Blinks/min")
        st.plotly_chart(fig_rate, use_container_width=True)

        # Detailed minute-by-minute plots
        st.subheader("Minute details")
        for r in results:
            with st.expander(f"Minute {r['minute']} (blinks: {r['blink_count']})", expanded=False):
                ts = r["timestamps"]
                sig = r["signal"]
                thr = r["threshold"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts, y=sig, mode="lines", name="signal"))
                fig.add_trace(go.Scatter(x=[ts[0], ts[-1]], y=[thr, thr], mode="lines",
                                         name="threshold", line=dict(color="red", dash="dash")))

                # Mark blinks
                if r["blinks"]:
                    blink_vals = [sig[np.argmin(np.abs(np.array(ts) - t))] for t in r["blinks"]]
                    fig.add_trace(go.Scatter(x=r["blinks"], y=blink_vals,
                                             mode="markers", name="blinks",
                                             marker=dict(color="red", symbol="x", size=10)))

                st.plotly_chart(fig, use_container_width=True)

                # CSV download
                csv_buf = "timestamp,signal\n"
                for t, s in zip(ts, sig):
                    csv_buf += f"{t:.3f},{s:.6f}\n"
                st.download_button(
                    f"Download minute {r['minute']} CSV",
                    csv_buf,
                    file_name=f"minute_{r['minute']:03d}.csv",
                    mime="text/csv"
                )

        # ZIP export
        zip_bytes = BytesIO()
        with zipfile.ZipFile(zip_bytes, "w", zipfile.ZIP_DEFLATED) as z:
            for r in results:
                csv_buf = "timestamp,signal\n"
                for t, s in zip(r["timestamps"], r["signal"]):
                    csv_buf += f"{t:.3f},{s:.6f}\n"
                z.writestr(f"minute_{r['minute']:03d}.csv", csv_buf)

            summary = "minute,blinks\n"
            for r in results:
                summary += f"{r['minute']},{r['blink_count']}\n"
            z.writestr("summary.csv", summary)

        st.download_button("Download all results (ZIP)",
                           zip_bytes.getvalue(),
                           file_name="analysis.zip",
                           mime="application/zip")


    # Cleanup
    if os.path.exists(input_path):
        os.unlink(input_path)
    if rotated_path != input_path and os.path.exists(rotated_path):
        os.unlink(rotated_path)


if __name__ == "__main__":
    main()
