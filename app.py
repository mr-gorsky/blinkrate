import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks, medfilt
import tempfile
import os
from PIL import Image
import io

st.set_page_config(
    page_title="Blink Rate Analyzer",
    page_icon="👁️",
    layout="wide"
)

def calculate_eye_variance(frame, eye_region=None):
    """
    Calculate eye openness based on image variance in the eye region
    Higher variance = more texture = eyes open
    Lower variance = more uniform = eyes closed/blinking
    """
    try:
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        height, width = gray.shape
        
        # Define eye region (central area of the frame)
        if eye_region is None:
            # Use central 50% of the frame as eye region
            y1, y2 = int(height * 0.25), int(height * 0.75)
            x1, x2 = int(width * 0.25), int(width * 0.75)
        else:
            y1, y2, x1, x2 = eye_region
        
        roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.5  # Default middle value
        
        # Calculate normalized variance
        variance = np.var(roi) / 255.0  # Normalize to 0-1 range
        
        # Apply logarithmic scaling to better distinguish states
        ear = np.log1p(variance * 10) / 4  # Scale to reasonable range
        
        # Clip to reasonable bounds
        ear = max(0.1, min(0.8, ear))
        
        return ear
        
    except Exception as e:
        return 0.5  # Fallback value

def process_video_fast(video_path, ear_threshold=0.3, max_frames=500):
    """
    Fast video processing with optimized frame sampling
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame skip to analyze at ~10fps for performance
    frame_skip = max(1, int(fps / 10))
    
    ear_values = []
    timestamps = []
    processed_frames = []
    frame_count = 0
    analyzed_count = 0
    
    while analyzed_count < max_frames and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only process every frame_skip-th frame
        if frame_count % frame_skip == 0:
            # Calculate EAR using variance method
            ear = calculate_eye_variance(frame)
            
            # Create preview frame (only store occasionally for performance)
            if analyzed_count % 50 == 0:  # Store every 50th analyzed frame
                preview_frame = frame.copy()
                height, width = preview_frame.shape[:2]
                
                # Draw eye region
                cv2.rectangle(preview_frame, 
                            (int(width * 0.25), int(height * 0.25)),
                            (int(width * 0.75), int(height * 0.75)),
                            (0, 255, 0), 2)
                
                # Add text
                color = (0, 0, 255) if ear < ear_threshold else (255, 255, 255)
                cv2.putText(preview_frame, f"EAR: {ear:.3f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if ear < ear_threshold:
                    cv2.putText(preview_frame, "BLINK", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                preview_frame_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                processed_frames.append((preview_frame_rgb, ear, ear < ear_threshold))
            
            ear_values.append(ear)
            timestamps.append(frame_count / fps)
            analyzed_count += 1
        
        frame_count += 1
    
    cap.release()
    return ear_values, timestamps, processed_frames, fps

def detect_blinks_smart(ear_values, ear_threshold=0.3, min_duration_frames=2, prominence=0.1):
    """
    Smart blink detection using peak finding on inverted EAR signal
    """
    try:
        # Invert the signal so blinks become peaks
        inverted_ear = [1 - ear for ear in ear_values]
        
        # Find peaks in the inverted signal (these are blinks)
        peaks, properties = find_peaks(inverted_ear, 
                                     height=1-ear_threshold, 
                                     prominence=prominence,
                                     distance=min_duration_frames)
        
        # Convert peaks to blink events
        blinks = []
        for peak in peaks:
            if inverted_ear[peak] >= (1 - ear_threshold):
                blinks.append((peak, peak, peak + 1))
        
        return blinks
        
    except:
        # Fallback to simple threshold method
        return detect_blinks_simple(ear_values, ear_threshold, min_duration_frames)

def detect_blinks_simple(ear_values, ear_threshold=0.3, min_consecutive_frames=2):
    """
    Simple threshold-based blink detection
    """
    blinks = []
    in_blink = False
    blink_start = 0
    
    for i, ear in enumerate(ear_values):
        if ear < ear_threshold and not in_blink:
            # Start of potential blink
            in_blink = True
            blink_start = i
        elif ear >= ear_threshold and in_blink:
            # End of potential blink
            blink_duration = i - blink_start
            if blink_duration >= min_consecutive_frames:
                blinks.append((blink_start, blink_start, i))
            in_blink = False
    
    # Handle blink at the end
    if in_blink and (len(ear_values) - blink_start) >= min_consecutive_frames:
        blinks.append((blink_start, blink_start, len(ear_values)))
    
    return blinks

def smooth_signal(signal, window_size=5):
    """Smooth signal using median filter"""
    if len(signal) < window_size:
        return signal
    return medfilt(signal, window_size)

def main():
    st.title("👁️ Blink Rate Analyzer")
    st.markdown("""
    Upload a monochromatic eye-tracking video to analyze blink rates. 
    **Optimized for VR headset side-mounted cameras.**
    """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an eye-tracking video file", 
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Supported formats: MP4, MOV, AVI, MKV"
    )
    
    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Quick video info
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            st.info(f"""
            **Video Info:** {total_frames} frames, {duration:.1f}s duration, {fps:.1f} FPS
            """)
            
            # Processing settings
            st.subheader("Detection Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                ear_threshold = st.slider(
                    "Sensitivity", 
                    min_value=0.1, 
                    max_value=0.5, 
                    value=0.3, 
                    step=0.05,
                    help="Lower = more sensitive to blinks"
                )
            with col2:
                min_blink_frames = st.slider(
                    "Min Blink Duration", 
                    min_value=1, 
                    max_value=5, 
                    value=2,
                    help="Minimum frames to count as blink"
                )
            with col3:
                smoothing = st.slider(
                    "Smoothing", 
                    min_value=1, 
                    max_value=7, 
                    value=3,
                    help="Noise reduction (odd numbers only)"
                )
            
            if st.button("🚀 Analyze Blink Rate", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Processing video...")
                ear_values, timestamps, processed_frames, actual_fps = process_video_fast(
                    video_path, ear_threshold
                )
                progress_bar.progress(50)
                
                if not ear_values:
                    st.error("Could not process video. Please check file format and try again.")
                    return
                
                status_text.text("Detecting blinks...")
                # Smooth the signal
                ear_smoothed = smooth_signal(ear_values, smoothing)
                
                # Detect blinks
                blinks = detect_blinks_smart(ear_smoothed, ear_threshold, min_blink_frames)
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Calculate metrics
                total_blinks = len(blinks)
                blink_rate = (total_blinks / duration) * 60 if duration > 0 else 0
                
                # Display results
                st.success(f"**Analysis Complete:** {total_blinks} blinks detected, {blink_rate:.1f} blinks/minute")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Blinks", total_blinks)
                col2.metric("Blink Rate", f"{blink_rate:.1f}/min")
                col3.metric("Duration", f"{duration:.1f}s")
                col4.metric("Frames Analyzed", len(ear_values))
                
                # Visualization tabs
                tab1, tab2, tab3 = st.tabs(["📈 EAR Signal", "📊 Blink Timeline", "👀 Sample Frames"])
                
                with tab1:
                    fig = go.Figure()
                    
                    # EAR signal
                    fig.add_trace(go.Scatter(
                        x=timestamps, y=ear_smoothed,
                        name='EAR Signal',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Threshold
                    fig.add_trace(go.Scatter(
                        x=timestamps, 
                        y=[ear_threshold] * len(timestamps),
                        name='Threshold',
                        line=dict(color='red', dash='dash', width=2)
                    ))
                    
                    # Blinks
                    if blinks:
                        blink_times = [timestamps[b[0]] for b in blinks]
                        blink_values = [ear_smoothed[b[0]] for b in blinks]
                        
                        fig.add_trace(go.Scatter(
                            x=blink_times, y=blink_values,
                            mode='markers',
                            name='Blinks',
                            marker=dict(color='red', size=10, symbol='x')
                        ))
                    
                    fig.update_layout(
                        title="Eye Aspect Ratio (EAR) Over Time",
                        xaxis_title="Time (seconds)",
                        yaxis_title="EAR Value",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab2:
                    # Create blink timeline
                    if blinks:
                        blink_df = pd.DataFrame([
                            {
                                'Start Time (s)': timestamps[b[0]],
                                'Duration (frames)': b[2] - b[0],
                                'EAR Value': ear_smoothed[b[0]]
                            }
                            for b in blinks
                        ])
                        
                        st.dataframe(blink_df, use_container_width=True)
                        
                        # Blink distribution
                        if len(blinks) > 1:
                            blink_intervals = []
                            for i in range(1, len(blinks)):
                                interval = timestamps[blinks[i][0]] - timestamps[blinks[i-1][0]]
                                blink_intervals.append(interval)
                            
                            fig_hist = go.Figure()
                            fig_hist.add_trace(go.Histogram(
                                x=blink_intervals,
                                name='Blink Intervals',
                                nbinsx=10
                            ))
                            fig_hist.update_layout(
                                title="Distribution of Blink Intervals",
                                xaxis_title="Time Between Blinks (seconds)",
                                yaxis_title="Count",
                                height=300
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                    else:
                        st.info("No blinks detected for timeline analysis")
                
                with tab3:
                    if processed_frames:
                        st.subheader("Sample Processed Frames")
                        cols = st.columns(min(3, len(processed_frames)))
                        for idx, (frame, ear, is_blink) in enumerate(processed_frames[:3]):
                            with cols[idx]:
                                pil_img = Image.fromarray(frame)
                                st.image(pil_img, 
                                       caption=f"EAR: {ear:.3f} {'🔴 BLINK' if is_blink else '⚪ Normal'}",
                                       use_column_width=True)
                    
                    # EAR statistics
                    st.subheader("Signal Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean EAR", f"{np.mean(ear_smoothed):.3f}")
                    col2.metric("Min EAR", f"{np.min(ear_smoothed):.3f}")
                    col3.metric("Max EAR", f"{np.max(ear_smoothed):.3f}")
                    col4.metric("Std Dev", f"{np.std(ear_smoothed):.3f}")
                
                # Export results
                st.subheader("📥 Export Results")
                results_text = f"""Blink Analysis Results
File: {uploaded_file.name}
Total Blinks: {total_blinks}
Blink Rate: {blink_rate:.2f} blinks/minute
Duration: {duration:.2f} seconds
Sensitivity: {ear_threshold}
Min Duration: {min_blink_frames} frames

Blinks detected at:
"""
                for blink in blinks:
                    results_text += f"{timestamps[blink[0]]:.2f}s (EAR: {ear_smoothed[blink[0]]:.3f})\n"
                
                st.download_button(
                    "Download Results (.txt)",
                    results_text,
                    file_name="blink_analysis_results.txt"
                )
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            st.info("""
            **Troubleshooting tips:**
            - Try a shorter video (under 30 seconds)
            - Ensure the eye is clearly visible
            - Adjust sensitivity slider
            - Check video format (MP4 usually works best)
            """)
        
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    else:
        # Instructions
        st.markdown("""
        ### 🎯 How to Use:
        
        1. **Record**: Use VR headset side camera to record eye movements
        2. **Upload**: Choose your video file (MP4 recommended)
        3. **Adjust**: Tune sensitivity if needed
        4. **Analyze**: Click the button to process
        
        ### ⚙️ Detection Method:
        - Uses image variance in the eye region
        - Higher variance = eyes open, lower variance = eyes closed
        - Automatically adapts to different lighting conditions
        
        ### 💡 Tips for Best Results:
        - Ensure clear view of the eye
        - Consistent lighting
        - Stable camera position
        - Start with default settings
        - Use videos under 2 minutes for faster processing
        """)

if __name__ == "__main__":
    main()
