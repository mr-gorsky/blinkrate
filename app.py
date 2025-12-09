import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import math
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import plotly.graph_objects as go
import zipfile
from io import BytesIO

st.set_page_config(page_title="Pupil & Blink Analyzer (robust timestamps)", layout="wide", page_icon="👁️")

# -------------------------
# pupil detection function
# -------------------------
def detect_pupil(gray_roi):
    """
    Return approximate pupil diameter in pixels or None.
    Uses CLAHE, median blur, invert, adaptive threshold, morphological open,
    then picks contour with highest circularity*area^0.6 score.
    """
    if gray_roi is None or gray_roi.size == 0:
        return None
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray_roi)
        blurred = cv2.medianBlur(enhanced, 5)
        inv = cv2.bitwise_not(blurred)
        th = cv2.adaptiveThreshold(inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = 0.0
        best_diam = None
        for c in contours:
            area = cv2.contourArea(c)
            if area < 8:
                continue
            peri = cv2.arcLength(c, True)
            if peri <= 0:
                continue
            circularity = 4.0 * math.pi * area / (peri * peri)
            score = circularity * (area ** 0.60)
            if score > best_score:
                best_score = score
                best_diam = math.sqrt(4.0 * area / math.pi)
        return float(best_diam) if best_diam is not None else None
    except Exception:
        return None

# -------------------------
# rotation helper
# -------------------------
def rotate_video_to_temp(input_path, rotation):
    """
    rotation: "none", "90cw", "90ccw", "180"
    Returns path to rotated temp video (or input_path if none).
    """
    if rotation == "none":
        return input_path

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return input_path

    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if rotation in ("90cw", "90ccw"):
        out_w, out_h = h, w
    else:
        out_w, out_h = w, h

    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = tmpf.name
    tmpf.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps_meta, (out_w, out_h), True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if rotation == "90cw":
            rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == "90ccw":
            rotated = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            rotated = cv2.rotate(frame, cv2.ROTATE_180)
        writer.write(rotated)

    cap.release()
    writer.release()
    return out_path

# -------------------------
# core analyzer using timestamps
# -------------------------
def analyze_video_by_timestamps(video_path,
                                roi_mode="left",
                                roi_offset_x=0.05,
                                roi_offset_y=0.0,
                                sample_fps=10,
                                smoothing_sigma=2,
                                threshold_method="percentile",
                                threshold_param=90,
                                auto_invert=True,
                                max_minutes=None,
                                progress_callback=None):
    """
    Read video sequentially, use CAP_PROP_POS_MSEC timestamps for timing.
    Sampling: we sample when elapsed_time - last_sample_time >= 1/sample_fps
    Grouping: minute_index = int(timestamp_s // 60)
    Returns: results (list per minute), meta dict with fps info
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    # metadata fps and framecount for display only
    fps_meta = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # We'll measure real fps from timestamp differences
    measured_ts = []
    # First pass: collect up to N timestamp samples to measure fps quickly
    # But we need to preserve stream position for full pass; easiest approach: do full pass but compute measured fps on the fly
    # We'll perform single pass and compute measured fps as mean diff of successive timestamps used for sampling.

    # Get first frame to determine ROI dimensions
    ret, frame0 = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Cannot read first frame")
    h, w = frame0.shape[:2]

    roi_w = int(w * 0.5)
    roi_h = int(h * 0.6)
    if roi_mode == "left":
        roi_x = int(max(0, w * roi_offset_x))
    elif roi_mode == "right":
        roi_x = int(max(0, w - roi_w - w * roi_offset_x))
    else:
        roi_x = int(max(0, (w - roi_w)//2 + w * roi_offset_x))
    roi_y = int((h - roi_h)//2 + int(h * roi_offset_y))
    roi_x = max(0, min(roi_x, w - roi_w))
    roi_y = max(0, min(roi_y, h - roi_h))

    # rewind to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # storage for minute buckets keyed by minute index
    buckets = {}  # minute -> {"timestamps":[], "diam": []}
    last_sample_time = None
    sample_interval = 1.0 / float(max(1, sample_fps))
    total_minutes = None
    measured_diffs = []

    # iterate through frames sequentially
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # get timestamp in seconds (use POS_MSEC)
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if t_ms is None or t_ms <= 0:
            # fallback: estimate from frame index -> not ideal but keep
            # attempt to get current frame index and approximate
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if fps_meta > 0:
                t_s = pos_frame / fps_meta
            else:
                t_s = 0.0
        else:
            t_s = float(t_ms) / 1000.0

        # initialize total_minutes on first valid timestamp
        if total_minutes is None and t_s is not None:
            # get approximate duration if possible using metadata; otherwise leave None
            if total_frames_meta > 0 and fps_meta > 0:
                approx_duration = total_frames_meta / fps_meta
                total_minutes = int(math.ceil(approx_duration / 60.0))
            else:
                total_minutes = None

        # sampling decision: use POS_MSEC differences
        do_sample = False
        if last_sample_time is None:
            do_sample = True
        else:
            if (t_s - last_sample_time) >= (sample_interval - 1e-6):
                do_sample = True

        if do_sample:
            # measure measured timestamps for FPS estimation
            if len(measured_ts) > 0:
                diff = t_s - measured_ts[-1]
                if diff > 0:
                    measured_diffs.append(diff)
            measured_ts.append(t_s)
            last_sample_time = t_s

            # process sample: detect pupil
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            diam = detect_pupil(roi)
            # store to bucket by minute index using timestamp
            minute_idx = int(t_s // 60)  # minute 0..n
            if max_minutes and max_minutes > 0:
                if minute_idx >= max_minutes:
                    # reached requested maximum minutes -> stop reading further frames
                    break
            if minute_idx not in buckets:
                buckets[minute_idx] = {"timestamps": [], "diam": []}
            buckets[minute_idx]["timestamps"].append(t_s)
            buckets[minute_idx]["diam"].append(diam if diam is not None else np.nan)

    cap.release()

    # compute measured FPS
    measured_fps = None
    if len(measured_diffs) >= 1:
        # remove zeros if any
        diffs = np.array(measured_diffs)
        diffs = diffs[diffs > 1e-6]
        if diffs.size > 0:
            measured_fps = float(1.0 / np.mean(diffs))

    # compute aggregated duration from measured timestamps if possible
    duration_measured = None
    if len(measured_ts) >= 2:
        duration_measured = measured_ts[-1] - measured_ts[0]

    # build per-minute results array sorted by minute index
    minute_indices = sorted(buckets.keys())
    results = []
    # decide polarity using first minute data if auto_invert True
    polarity = 1.0
    if auto_invert and len(minute_indices) > 0:
        # gather first N diam values (non-nan)
        first_vals = []
        for i in minute_indices[:1]:
            arr = np.array(buckets[i]["diam"], dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size > 0:
                first_vals.extend(arr.tolist())
        if len(first_vals) > 5:
            q = np.array(first_vals)
            # if mean>median -> blinks are dips -> set polarity negative
            if np.mean(q) > np.median(q):
                polarity = -1.0

    for mi in minute_indices:
        ts_list = buckets[mi]["timestamps"]
        diam_list = np.array(buckets[mi]["diam"], dtype=float)
        # interpolate nan values
        if np.all(np.isnan(diam_list)):
            diam_list = np.zeros_like(diam_list)
        else:
            nans = np.isnan(diam_list)
            if np.any(nans):
                good = np.where(~nans)[0]
                if good.size > 0:
                    diam_list[nans] = np.interp(np.where(nans)[0], good, diam_list[good])
                else:
                    diam_list[nans] = 0.0

        # closure raw: want high=closed, so invert/negate based on polarity
        closure_raw = -polarity * diam_list

        if closure_raw.max() - closure_raw.min() > 1e-6:
            norm = (closure_raw - closure_raw.min()) / (closure_raw.max() - closure_raw.min())
        else:
            norm = np.zeros_like(closure_raw)

        if len(norm) > 1:
            smoothed = gaussian_filter1d(norm, sigma=smoothing_sigma)
        else:
            smoothed = norm

        # thresholding
        if threshold_method == "percentile":
            thr = float(np.percentile(smoothed, threshold_param)) if len(smoothed) > 0 else 0.0
        elif threshold_method == "median_std":
            med = float(np.median(smoothed)) if len(smoothed) > 0 else 0.0
            std = float(np.std(smoothed)) if len(smoothed) > 0 else 0.0
            thr = med + threshold_param * std
        else:
            # Otsu-like using histogram
            try:
                hist, bins = np.histogram(smoothed, bins=64)
                total = hist.sum()
                sum_total = (bins[:-1] * hist).sum()
                weight_b = 0.0
                sum_b = 0.0
                max_var = 0.0
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
                    var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
                    if var_between > max_var:
                        max_var = var_between
                        thresh = bins[i]
                thr = float(thresh)
            except Exception:
                thr = float(np.percentile(smoothed, 90)) if len(smoothed) > 0 else 0.0

        # detect peaks (closure high)
        min_distance_samples = max(1, int(round(0.2 * sample_fps)))
        peaks, props = find_peaks(smoothed, height=thr, distance=min_distance_samples)
        blink_times = [ts_list[p] for p in peaks] if len(peaks) > 0 else []

        results.append({
            "minute": mi,
            "timestamps": ts_list,
            "signal": smoothed.tolist(),
            "threshold": thr,
            "blinks": blink_times,
            "blink_count": len(blink_times)
        })

        if progress_callback:
            progress_callback(mi + 1, (max_minutes if (max_minutes and max_minutes>0) else (max(minute_indices)+1 if minute_indices else 1)))

    meta = {
        "fps_meta": float(fps_meta),
        "frames_meta": int(total_frames_meta),
        "measured_fps": float(measured_fps) if measured_fps is not None else None,
        "duration_measured_s": float(duration_measured) if duration_measured is not None else None
    }
    return results, meta

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("👁️ Pupil & Blink Analyzer — Robust timestamps (GC0308)")

    uploaded = st.file_uploader("Upload video (mp4/mov/avi/mkv). For long videos we process by-minute.", type=["mp4","mov","avi","mkv"])
    if not uploaded:
        st.info("Upload a video sample (the one you already uploaded is fine). This tool measures timestamps and uses those for correct minute segmentation.")
        return

    with st.expander("Rotation", expanded=True):
        rotation = st.selectbox("Rotate video before analysis", ["none", "90cw", "90ccw", "180"],
                                format_func=lambda x: {"none":"No rotation", "90cw":"90° clockwise", "90ccw":"90° counter-clockwise", "180":"180°"}[x])

    with st.expander("Analysis settings", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            roi_mode = st.selectbox("ROI mode (camera side)", ["left","right","auto"], index=0)
            roi_offset_x = st.slider("ROI offset X (rel)", -0.2, 0.2, 0.05, 0.01)
            roi_offset_y = st.slider("ROI offset Y (rel)", -0.2, 0.2, 0.0, 0.01)
            sample_fps = st.slider("Sample FPS (used for sampling timestamps)", 3, 30, 10)
        with colB:
            smoothing_sigma = st.slider("Smoothing sigma", 0, 6, 2)
            threshold_method = st.selectbox("Threshold method", ["percentile","median_std","otsu"])
            if threshold_method == "percentile":
                threshold_param = st.slider("Percentile", 50, 99, 90)
            else:
                threshold_param = st.slider("k (for median_std)", 0.1, 3.0, 1.2)
            auto_invert = st.checkbox("Auto-detect signal polarity (recommended)", True)
            max_minutes = st.number_input("Max minutes to analyze (0 = all)", min_value=0, value=0, step=1)

    # save uploaded to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmpf:
        tmpf.write(uploaded.read())
        input_path = tmpf.name

    try:
        # rotate first (if asked)
        st.write("Preparing video (rotation if requested)...")
        rotated_path = rotate_video_to_temp(input_path, rotation)
        st.success("Video prepared.")

        # Show metadata and measured fps (quick pre-scan)
        cap_check = cv2.VideoCapture(rotated_path)
        fps_meta = cap_check.get(cv2.CAP_PROP_FPS) or 0.0
        frames_meta = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # quick measured fps sample: grab up to N frames and read POS_MSEC
        measured_ts = []
        grab_count = min(frames_meta, 300)
        read_idx = 0
        cap_check.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while read_idx < grab_count:
            ret, f = cap_check.read()
            if not ret:
                break
            if read_idx % max(1, int(max(1, round((frames_meta / grab_count))))) == 0:
                ms = cap_check.get(cv2.CAP_PROP_POS_MSEC)
                if ms > 0:
                    measured_ts.append(ms/1000.0)
            read_idx += 1
        cap_check.release()
        measured_fps_preview = None
        if len(measured_ts) >= 2:
            diffs = np.diff(np.array(measured_ts))
            diffs = diffs[diffs > 0]
            if diffs.size > 0:
                measured_fps_preview = float(1.0 / np.mean(diffs))

        st.write(f"Metadata FPS (OpenCV reported): **{fps_meta:.3f}**")
        st.write(f"Measured FPS (from timestamps preview): **{measured_fps_preview:.3f}**" if measured_fps_preview else "Measured FPS: not enough timestamps in preview")

        if st.button("Start analysis"):
            progress_text = st.empty()
            progress_bar = st.progress(0)

            def progress_cb(done, total):
                # total may be approximate; keep safe
                try:
                    pct = int(min(100, (done / total) * 100))
                except Exception:
                    pct = 0
                progress_bar.progress(pct)
                progress_text.text(f"Processing minute {done}/{total if total else '?'}...")

            results, meta = analyze_video_by_timestamps(
                rotated_path,
                roi_mode=roi_mode,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y,
                sample_fps=sample_fps,
                smoothing_sigma=smoothing_sigma,
                threshold_method=threshold_method,
                threshold_param=threshold_param,
                auto_invert=auto_invert,
                max_minutes=(None if max_minutes == 0 else int(max_minutes)),
                progress_callback=progress_cb
            )

            st.success("Analysis finished ✅")

            # show fps info
            st.subheader("FPS info")
            st.write(f"OpenCV reported FPS (metadata): **{meta.get('fps_meta',0):.3f}**")
            mfps = meta.get('measured_fps', None)
            st.write(f"Measured FPS (from timestamps): **{mfps:.3f}**" if mfps else "Measured FPS: not available (not enough timestamp samples)")
            if meta.get('duration_measured_s', None):
                st.write(f"Measured duration (s, from timestamps): **{meta['duration_measured_s']:.2f}**")

            # per-minute summary
            st.subheader("Per-minute summary")
            minutes = [r["minute"] for r in results]
            blink_counts = [r["blink_count"] for r in results]
            st.table({"minute": minutes, "blinks": blink_counts})

            # blink rate bar
            fig_rate = go.Figure()
            fig_rate.add_trace(go.Bar(x=minutes, y=blink_counts))
            fig_rate.update_layout(title="Blinks per minute", xaxis_title="Minute", yaxis_title="Blinks/min")
            st.plotly_chart(fig_rate, use_container_width=True)

            # minute details
            st.subheader("Minute details (expand each minute)")
            for r in results:
                with st.expander(f"Minute {r['minute']} — blinks: {r['blink_count']}", expanded=False):
                    ts = r["timestamps"]
                    sig = r["signal"]
                    thr = r["threshold"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts, y=sig, mode="lines", name="closure signal"))
                    if len(ts) > 0:
                        fig.add_trace(go.Scatter(x=[ts[0], ts[-1]], y=[thr, thr], mode="lines", name="threshold", line=dict(dash='dash', color='red')))
                    if r["blinks"]:
                        blink_vals = [sig[np.argmin(np.abs(np.array(ts) - bt))] for bt in r["blinks"]]
                        fig.add_trace(go.Scatter(x=r["blinks"], y=blink_vals, mode="markers", name="blinks", marker=dict(color="red", symbol="x", size=10)))
                    fig.update_layout(title=f"Minute {r['minute']} signal", xaxis_title="Time (s)", yaxis_title="Normalized closure", height=350)
                    st.plotly_chart(fig, use_container_width=True)

                    # CSV download for single minute
                    csv_txt = "timestamp_s,signal\n"
                    for t,s in zip(ts, sig):
                        csv_txt += f"{t:.3f},{s:.6f}\n"
                    st.download_button(f"Download minute {r['minute']} CSV", csv_txt, file_name=f"minute_{r['minute']:03d}.csv", mime="text/csv")

            # overall ZIP
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for r in results:
                    csv_txt = "timestamp_s,signal\n"
                    for t,s in zip(r["timestamps"], r["signal"]):
                        csv_txt += f"{t:.3f},{s:.6f}\n"
                    zf.writestr(f"minute_{r['minute']:03d}.csv", csv_txt)
                summary_csv = "minute,blinks\n"
                for r in results:
                    summary_csv += f"{r['minute']},{r['blink_count']}\n"
                zf.writestr("summary.csv", summary_csv)
            st.download_button("Download all results (ZIP)", zip_buf.getvalue(), file_name="analysis_results.zip", mime="application/zip")

    finally:
        # cleanup temp files
        try:
            if os.path.exists(input_path):
                os.unlink(input_path)
        except Exception:
            pass
        try:
            if rotated_path != input_path and os.path.exists(rotated_path):
                os.unlink(rotated_path)
        except Exception:
            pass

if __name__ == "__main__":
    main()
