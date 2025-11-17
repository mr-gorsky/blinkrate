import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import find_peaks
import tempfile
import os
import base64
from PIL import Image
import io
import time

# Try to import OpenCV with fallback
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.error("OpenCV is not available. Some features may be limited.")

# Try to import dlib with fallback  
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    st.warning("dlib is not available. Using alternative blink detection.")

st.set_page_config(
    page_title="Blink Rate Analyzer",
    page_icon="👁️",
    layout="wide"
)

def calculate_simple_ear(frame, eye_region=None):
    """
    Simple EAR calculation using basic image processing
    This works without facial landmarks
    """
    try:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        height, width = gray.shape
        
        # Define eye region (approximate)
        if eye_region is None:
            # Assume eye is in the central region
            roi = gray[height//3:2*height//3, width//4:3*width//4]
        else:
            y1, y2, x1, x2 = eye_region
            roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.3  # Default value
        
        # Normalize and calculate variance (open eyes have higher variance)
        roi_normalized = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
        variance = np.var(roi_normalized) / 1000  # Normalize variance
        
        # Convert to EAR-like metric
        ear = min(0.5, max(0.1, variance))
        return ear
        
    except Exception as e:
        return 0.3  # Fallback value

def process_video_simple(video_path, ear_threshold=0.2, max_frames=1000):
    """
    Process video using simple method (no facial landmarks)
    """
    if not OPENCV_AVAILABLE:
        return [], [], []
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ear_values = []
    timestamps = []
    processed_frames = []
    
    frame_count = 0
    success = True
    
    while success and frame_count < max_frames:
        success, frame = cap.read()
        
        if not success:
            break
            
        # Calculate simple EAR
        ear = calculate_simple_ear(frame)
        
        # Create preview frame
        preview_frame = frame.copy()
        height, width = preview_frame.shape[:2]
        
        # Draw eye region
        cv2.rectangle(preview_frame, 
                     (width//4, height//3), 
                     (3*width//4, 2*height//3), 
                     (0, 255, 0), 2)
        
        # Add EAR text
        color = (0, 0, 255) if ear < ear_threshold else (255, 255, 255)
        cv2.putText(preview_frame, f"EAR: {ear:.3f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if ear < ear_threshold:
            cv2.putText(preview_frame, "BLINK", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Convert BGR to RGB for display
        preview_frame_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
        processed_frames.append((preview_frame_rgb, ear, ear < ear_threshold))
        
        ear_values.append(ear)
        timestamps.append(frame_count / fps if fps > 0 else frame_count)
        frame_count += 1
    
    cap.release()
    return ear_values, timestamps, processed_frames

def detect_blinks(ear_values, ear_threshold=0.2, min_consecutive_frames=3):
    """
    Detect blinks from EAR values
    """
    blinks = []
    blink_start = None
    
    for i, ear in enumerate(ear_values):
        if ear < ear_threshold and blink_start is None:
            # Start of potential blink
            blink_start = i
        elif ear >= ear_threshold and blink_start is not None:
            # End of potential blink
            blink_duration = i - blink_start
            if blink_duration >= min_consecutive_frames:
                blinks.append((blink_start, blink_start, i))
            blink_start = None
    
    # Handle blink at the end
    if blink_start is not None and (len(ear_values) - blink_start) >= min_consecutive_frames:
        blinks.append((blink_start, blink_start, len(ear_values)))
    
    return blinks

def smooth_ear_values(ear_values, window_size=5):
    """Smooth EAR values using moving average"""
    if len(ear_values) < window_size:
        return ear_values
    
    smoothed = []
    for i in range(len(ear_values)):
        start = max(0, i - window_size // 2)
        end = min(len(ear_values), i + window_size // 2 + 1)
        window = ear_values[start:end]
        smoothed.append(np.mean(window))
    
    return smoothed

def main():
    st.title("👁️ Blink Rate Analyzer")
    st.markdown("""
    Upload a monochromatic eye-tracking video to analyze blink rates per minute and visualize blink patterns over time.
    This app is specifically designed for VR headset side-mounted eye-tracking cameras.
    """)
    
    # Show dependency status
    col1, col2 = st.columns(2)
    with col1:
        if OPENCV_AVAILABLE:
            st.success("✅ OpenCV available")
        else:
            st.error("❌ OpenCV not available")
    
    with col2:
        if DLIB_AVAILABLE:
            st.success("✅ dlib available")
        else:
            st.warning("⚠️ dlib not available - using simple detection")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an eye-tracking video file", 
        type=['mp4', 'mov', 'avi', 'mkv'],
        help="Upload a video of eye movements recorded from VR headset side camera"
    )
    
    if uploaded_file is not None:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Display video info
            if OPENCV_AVAILABLE:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = total_frames / fps if fps > 0 else 0
                cap.release()
                
                st.info(f"""
                **Video Information:**
                - FPS: {fps:.2f}
                - Total Frames: {total_frames}
                - Duration: {duration:.2f} seconds
                """)
            else:
                st.warning("OpenCV not available - using estimated video information")
                fps = 30
                total_frames = 1000
                duration = 33.33
            
            # Processing parameters
            st.subheader("Detection Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                ear_threshold = st.slider(
                    "EAR Threshold", 
                    min_value=0.1, 
                    max_value=0.5, 
                    value=0.25, 
                    step=0.01,
                    help="Threshold for blink detection (lower = more sensitive)"
                )
            with col2:
                min_blink_frames = st.slider(
                    "Minimum Blink Frames", 
                    min_value=1, 
                    max_value=10, 
                    value=3,
                    help="Minimum consecutive frames below threshold to count as blink"
                )
            with col3:
                smoothing = st.slider(
                    "Smoothing Window", 
                    min_value=1, 
                    max_value=10, 
                    value=3,
                    help="Moving average window size for smoothing EAR values"
                )
            
            if st.button("Analyze Blink Rate", type="primary"):
                with st.spinner("Processing video and detecting blinks..."):
                    # Process video
                    if OPENCV_AVAILABLE:
                        ear_values, timestamps, processed_frames = process_video_simple(
                            video_path, 
                            ear_threshold,
                            max_frames=min(1000, total_frames)  # Limit frames for performance
                        )
                    else:
                        st.error("OpenCV is required for video processing")
                        return
                    
                    if len(ear_values) == 0:
                        st.error("No video frames processed. Please check the video file.")
                        return
                    
                    # Smooth EAR values
                    ear_values_smoothed = smooth_ear_values(ear_values, smoothing)
                    
                    # Detect blinks
                    blinks = detect_blinks(ear_values_smoothed, ear_threshold, min_blink_frames)
                    
                    # Calculate blink rate per minute
                    if duration > 0:
                        total_blinks = len(blinks)
                        blink_rate_per_minute = (total_blinks / duration) * 60
                    else:
                        blink_rate_per_minute = 0
                        total_blinks = len(blinks)
                    
                    # Display results
                    st.success(f"Analysis Complete!")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Blinks", total_blinks)
                    with col2:
                        st.metric("Blink Rate", f"{blink_rate_per_minute:.2f} blinks/min")
                    with col3:
                        st.metric("Video Duration", f"{duration:.2f} seconds")
                    with col4:
                        st.metric("Frames Analyzed", len(ear_values))
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs(["EAR Signal", "Blink Rate Over Time", "Processing Preview"])
                    
                    with tab1:
                        # Plot EAR signal with blinks
                        fig_ear = go.Figure()
                        
                        # Original EAR signal
                        fig_ear.add_trace(go.Scatter(
                            x=timestamps,
                            y=ear_values,
                            mode='lines',
                            name='Raw EAR',
                            line=dict(color='lightblue', width=1),
                            opacity=0.6
                        ))
                        
                        # Smoothed EAR signal
                        fig_ear.add_trace(go.Scatter(
                            x=timestamps,
                            y=ear_values_smoothed,
                            mode='lines',
                            name='Smoothed EAR',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Threshold line
                        fig_ear.add_trace(go.Scatter(
                            x=timestamps,
                            y=[ear_threshold] * len(timestamps),
                            mode='lines',
                            name='Threshold',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        # Blink markers
                        if blinks:
                            blink_times = [timestamps[blink[0]] for blink in blinks]
                            blink_ears = [ear_values_smoothed[blink[0]] for blink in blinks]
                            
                            fig_ear.add_trace(go.Scatter(
                                x=blink_times,
                                y=blink_ears,
                                mode='markers',
                                name='Blinks',
                                marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
                            ))
                        
                        fig_ear.update_layout(
                            title="Eye Aspect Ratio (EAR) Signal with Blink Detection",
                            xaxis_title="Time (seconds)",
                            yaxis_title="EAR Value",
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_ear, use_container_width=True)
                    
                    with tab2:
                        # Calculate blink rate per minute throughout the video
                        if duration >= 60:  # Only if video is at least 1 minute
                            time_windows = np.arange(0, duration, 60)  # 1-minute windows
                            if time_windows[-1] < duration:
                                time_windows = np.append(time_windows, duration)
                            
                            blink_rates = []
                            time_labels = []
                            
                            for i in range(len(time_windows) - 1):
                                start_time = time_windows[i]
                                end_time = time_windows[i + 1]
                                
                                # Count blinks in this window
                                blinks_in_window = 0
                                for blink in blinks:
                                    blink_time = timestamps[blink[0]]
                                    if start_time <= blink_time < end_time:
                                        blinks_in_window += 1
                                
                                # Calculate rate per minute
                                window_duration = end_time - start_time
                                if window_duration > 0:
                                    rate = (blinks_in_window / window_duration) * 60
                                else:
                                    rate = 0
                                
                                blink_rates.append(rate)
                                time_labels.append(f"{start_time/60:.1f}-{end_time/60:.1f}")
                            
                            # Create blink rate over time plot
                            fig_rate = go.Figure()
                            
                            fig_rate.add_trace(go.Bar(
                                x=time_labels,
                                y=blink_rates,
                                name='Blinks per Minute',
                                marker_color='green',
                                opacity=0.7
                            ))
                            
                            fig_rate.update_layout(
                                title="Blink Rate Over Time (per Minute Windows)",
                                xaxis_title="Time Window (minutes)",
                                yaxis_title="Blinks per Minute",
                                height=500
                            )
                            
                            st.plotly_chart(fig_rate, use_container_width=True)
                        else:
                            st.info("Video is shorter than 1 minute. Minute-by-minute analysis requires longer videos.")
                            
                            # Show blink distribution anyway
                            if blinks:
                                blink_times = [timestamps[blink[0]] for blink in blinks]
                                
                                fig_dist = go.Figure()
                                fig_dist.add_trace(go.Histogram(
                                    x=blink_times,
                                    nbinsx=10,
                                    name='Blink Distribution',
                                    marker_color='orange'
                                ))
                                
                                fig_dist.update_layout(
                                    title="Blink Time Distribution",
                                    xaxis_title="Time (seconds)",
                                    yaxis_title="Number of Blinks",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_dist, use_container_width=True)
                    
                    with tab3:
                        # Show sample processed frames
                        st.subheader("Sample Processed Frames")
                        
                        if processed_frames:
                            # Show frames with blinks if any, otherwise random frames
                            blink_frames = [f for f in processed_frames if f[2]]
                            display_frames = blink_frames if blink_frames else processed_frames
                            
                            # Take up to 3 frames
                            sample_frames = display_frames[:3]
                            
                            cols = st.columns(len(sample_frames))
                            for idx, (frame, ear, is_blink) in enumerate(sample_frames):
                                with cols[idx]:
                                    # Convert numpy array to PIL Image
                                    pil_img = Image.fromarray(frame)
                                    st.image(pil_img, 
                                           caption=f"EAR: {ear:.3f} {'🔴 BLINK' if is_blink else '⚪ Normal'}", 
                                           use_column_width=True)
                        
                        # Show EAR statistics
                        st.subheader("EAR Statistics")
                        if ear_values_smoothed:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Mean EAR", f"{np.mean(ear_values_smoothed):.3f}")
                            with col2:
                                st.metric("Min EAR", f"{np.min(ear_values_smoothed):.3f}")
                            with col3:
                                st.metric("Max EAR", f"{np.max(ear_values_smoothed):.3f}")
                            with col4:
                                st.metric("Std EAR", f"{np.std(ear_values_smoothed):.3f}")
                    
                    # Download results
                    st.subheader("Export Results")
                    
                    # Create results summary
                    results_text = f"""Blink Analysis Results
=====================
Video: {uploaded_file.name}
Total Blinks: {total_blinks}
Blink Rate: {blink_rate_per_minute:.2f} blinks/minute
Video Duration: {duration:.2f} seconds
EAR Threshold: {ear_threshold}
Minimum Blink Frames: {min_blink_frames}
Frames Analyzed: {len(ear_values)}

Blink Events:
Frame | Time(s) | EAR Value
"""
                    for blink in blinks:
                        frame_idx, start, end = blink
                        results_text += f"{frame_idx:6d} | {timestamps[frame_idx]:7.2f} | {ear_values_smoothed[frame_idx]:.3f}\n"
                    
                    st.download_button(
                        label="Download Results as TXT",
                        data=results_text,
                        file_name="blink_analysis_results.txt",
                        mime="text/plain"
                    )
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            
            st.info("""
            **Troubleshooting tips:**
            1. Try a shorter video (under 30 seconds)
            2. Ensure the eye is clearly visible
            3. Adjust the EAR threshold
            4. Check video format compatibility
            """)
        
        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    else:
        # Show instructions when no file is uploaded
        st.markdown("""
        ### Instructions:
        
        1. **Record your video** using a monochromatic eye-tracking camera mounted in VR headset
        2. **Upload the video** (MP4, MOV, AVI, or MKV formats supported)
        3. **Adjust parameters** if needed
        4. **Click "Analyze Blink Rate"** to process the video
        
        ### Detection Method:
        This app uses image variance in the eye region to detect blinks. When eyes are open, 
        there's more texture and variation. During blinks, the region becomes more uniform.
        
        ### Tips for Best Results:
        - Ensure clear view of the eye
        - Consistent lighting
        - Stable camera position
        - Minimum 15 FPS recommended
        - Start with default parameters and adjust if needed
        """)

if __name__ == "__main__":
    main()
