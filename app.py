import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import tempfile
import os
from scipy.signal import find_peaks, medfilt
import pandas as pd

st.set_page_config(
    page_title="VR Eye Blink Analyzer",
    page_icon="👁️",
    layout="wide"
)

def preprocess_vr_frame(frame):
    """Preprocessing optimized for VR side-camera footage"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    
    # VR cameras often have specific lighting - enhance contrast
    gray = cv2.equalizeHist(gray)
    
    # Reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    return gray

def detect_vr_eye_region(frame):
    """
    Detect eye region specifically for VR side-camera footage
    Based on the screenshot, the eye is in the left portion of the frame
    """
    height, width = frame.shape[:2]
    
    # For VR side-camera: eye is typically in left 2/3 of frame, centered vertically
    # Adjust these based on the actual camera position
    eye_region = (
        int(width * 0.1),   # x1 - start from left
        int(height * 0.2),  # y1 - from top
        int(width * 0.7),   # x2 - to right
        int(height * 0.8)   # y2 - to bottom
    )
    
    return eye_region

def calculate_vr_eye_state(frame):
    """
    Calculate eye state specifically for VR side-camera footage
    Optimized for the angled view and lighting conditions
    """
    gray = preprocess_vr_frame(frame)
    eye_region = detect_vr_eye_region(frame)
    x1, y1, x2, y2 = eye_region
    
    roi = gray[y1:y2, x1:x2]
    
    if roi.size == 0:
        return 0.4, eye_region
    
    # For VR side-cameras, we need different metrics
    height, width = roi.shape
    
    # Split into subregions for better analysis
    # Upper part (eyelid area) and lower part (pupil/iris area)
    upper_roi = roi[0:int(height*0.4), :]
    lower_roi = roi[int(height*0.4):, :]
    
    metrics = {}
    
    # 1. Upper region analysis (eyelid movement)
    if upper_roi.size > 0:
        metrics['upper_variance'] = np.var(upper_roi) / 500  # Normalized
        metrics['upper_edges'] = np.sum(cv2.Canny(upper_roi, 30, 100) > 0) / upper_roi.size
    else:
        metrics['upper_variance'] = 0.3
        metrics['upper_edges'] = 0.2
    
    # 2. Lower region analysis (pupil/iris)
    if lower_roi.size > 0:
        metrics['lower_variance'] = np.var(lower_roi) / 500
        metrics['lower_edges'] = np.sum(cv2.Canny(lower_roi, 30, 100) > 0) / lower_roi.size
        
        # Brightness analysis - pupil is typically darker
        lower_mean = np.mean(lower_roi) / 255
        metrics['lower_brightness'] = 1.0 - lower_mean  # Invert - darker = more open
    else:
        metrics['lower_variance'] = 0.3
        metrics['lower_edges'] = 0.2
        metrics['lower_brightness'] = 0.5
    
    # 3. Whole region metrics
    metrics['whole_variance'] = np.var(roi) / 500
    metrics['whole_edges'] = np.sum(cv2.Canny(roi, 30, 100) > 0) / roi.size
    
    # Weighted combination for VR side-camera
    # Emphasize upper region for blink detection (eyelid movement)
    openness = (
        metrics['upper_variance'] * 0.35 +      # Eyelid texture
        metrics['upper_edges'] * 0.25 +         # Eyelid edges
        metrics['lower_brightness'] * 0.20 +    # Pupil darkness
        metrics['whole_variance'] * 0.15 +      # Overall texture
        metrics['whole_edges'] * 0.05           # Overall edges
    )
    
    # Adjust for VR camera characteristics
    # VR footage typically has lower overall values due to angle and lighting
    openness = openness * 1.3  # Compensate for generally lower values
    
    return max(0.1, min(0.9, openness)), eye_region

def calculate_vr_adaptive_threshold(openness_values):
    """
    Calculate adaptive threshold specifically for VR footage
    """
    if len(openness_values) < 20:
        return 0.22  # Good default for VR side-cameras
    
    values = np.array(openness_values)
    
    # VR footage typically has:
    # - Lower mean values (0.3-0.5 range)
    # - Smaller variance
    mean_val = np.mean(values)
    std_val = np.std(values)
    q25 = np.percentile(values, 25)
    
    # For VR: use lower threshold based on distribution
    if mean_val < 0.4:
        # Very angled view - use very conservative threshold
        threshold = q25 * 0.9
    else:
        # More visible - use standard approach
        threshold = mean_val - (std_val * 0.7)
    
    # VR-specific bounds
    threshold = max(0.18, min(0.30, threshold))
    
    return threshold

def process_vr_video(video_path, user_threshold=None):
    """Process video with VR-optimized parameters"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    openness_values = []
    timestamps = []
    processed_frames = []
    
    frame_count = 0
    max_frames = min(2000, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate VR-optimized eye state
        openness, eye_region = calculate_vr_eye_state(frame)
        
        openness_values.append(openness)
        timestamps.append(frame_count / fps)
        
        # Store sample frames (every 40 frames for performance)
        if frame_count % 40 == 0:
            preview = frame.copy()
            x1, y1, x2, y2 = eye_region
            
            # Draw detection area
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw subregions
            cv2.rectangle(preview, 
                         (x1, y1), 
                         (x2, int(y1 + (y2-y1)*0.4)), 
                         (255, 255, 0), 1)  # Upper region
            
            # Add info text
            color = (0, 0, 255) if openness < (user_threshold or 0.22) else (255, 255, 255)
            cv2.putText(preview, f"VR Openness: {openness:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(preview, "Green: Eye ROI | Blue: Eyelid area", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if openness < (user_threshold or 0.22):
                cv2.putText(preview, "BLINK DETECTED", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            processed_frames.append((preview_rgb, openness, openness < (user_threshold or 0.22)))
        
        frame_count += 1
    
    cap.release()
    
    return openness_values, timestamps, processed_frames, fps

def detect_vr_blinks(openness_values, threshold=0.22, min_duration=2):
    """Blink detection optimized for VR footage"""
    # Smooth specifically for VR signal characteristics
    openness_smoothed = medfilt(openness_values, 5)
    
    # Find blinks as valleys in the signal
    valleys, properties = find_peaks(
        1 - np.array(openness_smoothed),
        height=1-threshold,
        distance=min_duration,
        prominence=0.08,  # Lower prominence for VR
        width=min_duration
    )
    
    blinks = []
    for valley in valleys:
        if openness_smoothed[valley] < threshold:
            # Simple blink event - valley frame
            blinks.append((valley, valley, valley))
    
    return blinks, openness_smoothed

def main():
    st.title("👁️ VR Eye Blink Analyzer")
    st.markdown("**Optimizirano za VR headset side-camera snimke**")
    
    uploaded_file = st.file_uploader(
        "Uploadaj VR eye-tracking video", 
        type=['mp4', 'mov', 'avi', 'mkv', 'webm']
    )
    
    if uploaded_file is not None:
        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("Video prevelik! Maksimalno 200MB. Smanji video prije uploada.")
            return
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Video info
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            st.info(f"""
            **VR Video Analysis Ready**  
            📁 {uploaded_file.name}  
            ⏱️ {duration:.1f}s duration  
            🎞️ {total_frames} frames  
            🚀 {fps:.1f} FPS
            """)
            
            # VR-specific settings
            st.subheader("🎯 VR-Specific Settings")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Threshold Mode**")
                auto_mode = st.radio("", ["Auto VR", "Manual"], horizontal=True)
            with col2:
                if auto_mode == "Manual":
                    threshold = st.slider("Threshold", 0.15, 0.35, 0.22, 0.01,
                                        help="Niže vrijednosti = osjetljivije")
                else:
                    threshold = st.slider("Base Sensitivity", 0.18, 0.28, 0.22, 0.01,
                                        help="Početna osjetljivost za auto-mode")
            with col3:
                min_duration = st.slider("Min Duration", 1, 5, 2,
                                       help="Minimalno trajanje treptaja (frejmovi)")
            
            if st.button("🚀 START VR BLINK ANALYSIS", type="primary"):
                with st.spinner("Processing VR footage..."):
                    # Process with VR-optimized algorithm
                    openness_values, timestamps, processed_frames, actual_fps = process_vr_video(
                        video_path, threshold if auto_mode == "Manual" else None
                    )
                
                if not openness_values:
                    st.error("Nema podataka za analizu. Provjeri video format.")
                    return
                
                # Calculate VR-optimized threshold
                if auto_mode == "Auto VR":
                    vr_threshold = calculate_vr_adaptive_threshold(openness_values)
                    final_threshold = min(threshold + 0.02, vr_threshold)
                    st.success(f"🤖 VR Auto-threshold: **{final_threshold:.3f}**")
                else:
                    final_threshold = threshold
                
                # Detect blinks
                blinks, openness_smoothed = detect_vr_blinks(
                    openness_values, final_threshold, min_duration
                )
                
                # Calculate results
                total_blinks = len(blinks)
                blink_rate = (total_blinks / duration) * 60 if duration > 0 else 0
                
                # Display results
                st.success(f"**🎯 VR ANALYSIS COMPLETE: {total_blinks} blinks detected**")
                
                # Metrics dashboard
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Blinks", total_blinks)
                col2.metric("Blink Rate", f"{blink_rate:.1f}/min")
                col3.metric("VR Threshold", f"{final_threshold:.3f}")
                col4.metric("Signal Quality", f"{np.mean(openness_smoothed):.3f}")
                
                # Detailed analysis
                tab1, tab2, tab3 = st.tabs(["📊 Signal Analysis", "📈 Blink Timeline", "👁️ Frame Preview"])
                
                with tab1:
                    # Create VR-optimized plot
                    fig = go.Figure()
                    
                    # Raw signal (light)
                    fig.add_trace(go.Scatter(
                        x=timestamps, y=openness_values,
                        name='Raw Signal',
                        line=dict(color='lightblue', width=1),
                        opacity=0.6
                    ))
                    
                    # Smoothed signal (bold)
                    fig.add_trace(go.Scatter(
                        x=timestamps, y=openness_smoothed,
                        name='Smoothed',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # Threshold line
                    fig.add_trace(go.Scatter(
                        x=timestamps, y=[final_threshold] * len(timestamps),
                        name='Threshold',
                        line=dict(color='red', dash='dash', width=2)
                    ))
                    
                    # Blink markers
                    if blinks:
                        blink_times = [timestamps[b[0]] for b in blinks]
                        blink_values = [openness_smoothed[b[0]] for b in blinks]
                        
                        fig.add_trace(go.Scatter(
                            x=blink_times, y=blink_values,
                            mode='markers',
                            name='Blinks',
                            marker=dict(color='red', size=10, symbol='x', line=dict(width=2))
                        ))
                    
                    fig.update_layout(
                        title="VR Eye Openness Signal with Blink Detection",
                        xaxis_title="Time (seconds)",
                        yaxis_title="Eye Openness Score",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Signal statistics
                    st.subheader("Signal Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Mean", f"{np.mean(openness_smoothed):.3f}")
                    col2.metric("Std Dev", f"{np.std(openness_smoothed):.3f}")
                    col3.metric("Dynamic Range", f"{np.ptp(openness_smoothed):.3f}")
                    col4.metric("Blinks/Min", f"{blink_rate:.1f}")
                
                with tab2:
                    if blinks:
                        st.subheader("Blink Events Timeline")
                        
                        # Create timeline data
                        blink_data = []
                        for i, blink in enumerate(blinks):
                            blink_time = timestamps[blink[0]]
                            blink_data.append({
                                'Blink #': i + 1,
                                'Time (s)': f"{blink_time:.2f}",
                                'Openness': f"{openness_smoothed[blink[0]]:.3f}",
                                'Frame': blink[0]
                            })
                        
                        # Display as dataframe
                        df = pd.DataFrame(blink_data)
                        st.dataframe(df, use_container_width=True)
                        
                        # Blink intervals analysis
                        if len(blinks) > 1:
                            intervals = []
                            for i in range(1, len(blinks)):
                                interval = timestamps[blinks[i][0]] - timestamps[blinks[i-1][0]]
                                intervals.append(interval)
                            
                            st.subheader("Blink Interval Analysis")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Avg Interval", f"{np.mean(intervals):.2f}s")
                            col2.metric("Min Interval", f"{np.min(intervals):.2f}s")
                            col3.metric("Max Interval", f"{np.max(intervals):.2f}s")
                    
                    else:
                        st.info("No blinks detected. Try adjusting the threshold or check video quality.")
                
                with tab3:
                    if processed_frames:
                        st.subheader("VR Frame Analysis Preview")
                        st.info("Green: Eye detection area | Blue: Eyelid region")
                        
                        cols = st.columns(min(3, len(processed_frames)))
                        for idx, (frame, openness, is_blink) in enumerate(processed_frames[:3]):
                            with cols[idx]:
                                st.image(Image.fromarray(frame),
                                       caption=f"Openness: {openness:.3f} {'🔴 BLINK' if is_blink else '⚪ OPEN'}",
                                       use_column_width=True)
                
                # Export results
                st.subheader("📥 Export VR Analysis Results")
                results_text = f"""VR EYE BLINK ANALYSIS RESULTS
================================
Video: {uploaded_file.name}
Total Blinks: {total_blinks}
Blink Rate: {blink_rate:.1f} blinks/minute
Video Duration: {duration:.2f} seconds
VR Threshold: {final_threshold:.3f}
Frames Analyzed: {len(openness_values)}
Analysis Mode: {'Auto VR' if auto_mode == 'Auto VR' else 'Manual'}

DETECTED BLINKS:
Time(s)    Openness
"""
                for blink in blinks:
                    results_text += f"{timestamps[blink[0]]:7.2f}    {openness_smoothed[blink[0]]:.3f}\n"
                
                st.download_button(
                    "💾 Download Full Analysis",
                    results_text,
                    file_name=f"vr_blink_analysis_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain"
                )
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
            st.info("""
            **VR-Specific Troubleshooting:**
            - Ensure eye is clearly visible in the video
            - Try shorter videos (10-30 seconds)
            - Adjust threshold manually if auto-mode doesn't work
            - Check lighting consistency in VR footage
            """)
        
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    else:
        st.markdown("""
        ### 🎯 VR-Specific Features:
        
        **Optimized for Side-Camera Footage:**
        - 🤖 **Auto VR-threshold** - specifically tuned for angled views
        - 👁️ **Eyelid-focused analysis** - better for partial visibility
        - 🎯 **VR region detection** - optimized for side-camera positioning
        - 📊 **Adaptive algorithms** - adjust to VR lighting conditions
        
        **Expected VR Signal Characteristics:**
        - Open eye: 0.4-0.7 range
        - Closed eye: 0.15-0.3 range  
        - Typical threshold: 0.18-0.28
        
        **Upload Tips:**
        - Short videos work best (10-30 seconds)
        - Ensure consistent lighting
        - Eye should be visible in left portion of frame
        - Use MP4 or WEBM format
        """)

if __name__ == "__main__":
    main()
