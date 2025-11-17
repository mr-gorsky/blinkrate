import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.signal import find_peaks
import tempfile
import os
from utils import process_video, detect_blinks, calculate_ear

st.set_page_config(
    page_title="Blink Rate Analyzer",
    page_icon="👁️",
    layout="wide"
)

def main():
    st.title("👁️ Blink Rate Analyzer")
    st.markdown("""
    Upload a monochromatic eye-tracking video to analyze blink rates per minute and visualize blink patterns over time.
    This app is specifically designed for VR headset side-mounted eye-tracking cameras.
    """)
    
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
            
            # Processing parameters
            col1, col2 = st.columns(2)
            with col1:
                ear_threshold = st.slider(
                    "EAR Threshold", 
                    min_value=0.1, 
                    max_value=0.5, 
                    value=0.2, 
                    step=0.01,
                    help="Eye Aspect Ratio threshold for blink detection (lower = more sensitive)"
                )
            with col2:
                min_blink_frames = st.slider(
                    "Minimum Blink Frames", 
                    min_value=1, 
                    max_value=10, 
                    value=3,
                    help="Minimum consecutive frames below threshold to count as blink"
                )
            
            if st.button("Analyze Blink Rate", type="primary"):
                with st.spinner("Processing video and detecting blinks..."):
                    # Process video
                    ear_values, timestamps, processed_frames = process_video(
                        video_path, 
                        ear_threshold
                    )
                    
                    if len(ear_values) == 0:
                        st.error("No face/eye detected in the video. Please check the video quality and positioning.")
                        return
                    
                    # Detect blinks
                    blinks = detect_blinks(ear_values, ear_threshold, min_blink_frames)
                    
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
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Blinks", total_blinks)
                    with col2:
                        st.metric("Blink Rate", f"{blink_rate_per_minute:.2f} blinks/min")
                    with col3:
                        st.metric("Video Duration", f"{duration:.2f} seconds")
                    
                    # Create tabs for different visualizations
                    tab1, tab2, tab3 = st.tabs(["EAR Signal", "Blink Rate Over Time", "Processing Preview"])
                    
                    with tab1:
                        # Plot EAR signal with blinks
                        fig_ear = go.Figure()
                        
                        # EAR signal
                        fig_ear.add_trace(go.Scatter(
                            x=timestamps,
                            y=ear_values,
                            mode='lines',
                            name='EAR Signal',
                            line=dict(color='blue', width=1)
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
                        blink_times = [timestamps[blink[0]] for blink in blinks]
                        blink_ears = [ear_values[blink[0]] for blink in blinks]
                        
                        fig_ear.add_trace(go.Scatter(
                            x=blink_times,
                            y=blink_ears,
                            mode='markers',
                            name='Blinks',
                            marker=dict(color='red', size=8, symbol='x')
                        ))
                        
                        fig_ear.update_layout(
                            title="Eye Aspect Ratio (EAR) Signal with Blink Detection",
                            xaxis_title="Time (seconds)",
                            yaxis_title="EAR Value",
                            height=500
                        )
                        
                        st.plotly_chart(fig_ear, use_container_width=True)
                    
                    with tab2:
                        # Calculate blink rate per minute throughout the video
                        if duration > 0:
                            time_windows = np.arange(0, duration, 60)  # 1-minute windows
                            blink_rates = []
                            
                            for i in range(len(time_windows) - 1):
                                start_time = time_windows[i]
                                end_time = time_windows[i + 1]
                                
                                # Count blinks in this window
                                blinks_in_window = sum(1 for blink in blinks 
                                                     if start_time <= timestamps[blink[0]] < end_time)
                                blink_rates.append(blinks_in_window)
                            
                            # Create blink rate over time plot
                            if len(blink_rates) > 0:
                                fig_rate = go.Figure()
                                
                                fig_rate.add_trace(go.Scatter(
                                    x=time_windows[1:],
                                    y=blink_rates,
                                    mode='lines+markers',
                                    name='Blinks per Minute',
                                    line=dict(color='green', width=3),
                                    marker=dict(size=8)
                                ))
                                
                                fig_rate.update_layout(
                                    title="Blink Rate Over Time (per Minute)",
                                    xaxis_title="Time (minutes)",
                                    yaxis_title="Blinks per Minute",
                                    height=500
                                )
                                
                                # Convert x-axis to minutes
                                fig_rate.update_xaxes(tickvals=time_windows[1:], 
                                                    ticktext=[f"{t/60:.1f}" for t in time_windows[1:]])
                                
                                st.plotly_chart(fig_rate, use_container_width=True)
                            else:
                                st.info("Video is too short for minute-by-minute analysis")
                        else:
                            st.info("Could not calculate video duration")
                    
                    with tab3:
                        # Show sample processed frames
                        st.subheader("Sample Processed Frames")
                        
                        if processed_frames:
                            cols = st.columns(min(3, len(processed_frames)))
                            for idx, (frame, ear, is_blink) in enumerate(processed_frames[:3]):
                                with cols[idx]:
                                    st.image(frame, caption=f"EAR: {ear:.3f} {'🔴 BLINK' if is_blink else '⚪ Normal'}", 
                                            use_column_width=True)
                        
                        # Show EAR statistics
                        st.subheader("EAR Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Mean EAR", f"{np.mean(ear_values):.3f}")
                        with col2:
                            st.metric("Min EAR", f"{np.min(ear_values):.3f}")
                        with col3:
                            st.metric("Max EAR", f"{np.max(ear_values):.3f}")
                        with col4:
                            st.metric("Std EAR", f"{np.std(ear_values):.3f}")
                    
                    # Download results
                    st.subheader("Export Results")
                    
                    # Create results summary
                    results_text = f"""
Blink Analysis Results
=====================
Video: {uploaded_file.name}
Total Blinks: {total_blinks}
Blink Rate: {blink_rate_per_minute:.2f} blinks/minute
Video Duration: {duration:.2f} seconds
EAR Threshold: {ear_threshold}
Minimum Blink Frames: {min_blink_frames}

Blink Events:
Frame | Time(s) | EAR Value
"""
                    for blink in blinks:
                        frame_idx, start, end = blink
                        results_text += f"{frame_idx:6d} | {timestamps[frame_idx]:7.2f} | {ear_values[frame_idx]:.3f}\n"
                    
                    st.download_button(
                        label="Download Results as TXT",
                        data=results_text,
                        file_name="blink_analysis_results.txt",
                        mime="text/plain"
                    )
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.info("""
            Common issues:
            1. No face/eyes detected in the video
            2. Poor lighting or video quality
            3. Face not properly visible to camera
            4. Video format/codec issues
            
            Try adjusting the EAR threshold or ensure the eye is clearly visible throughout the video.
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
        3. **Adjust parameters** if needed:
           - EAR Threshold: Sensitivity for blink detection (default: 0.2)
           - Minimum Blink Frames: Minimum consecutive frames to count as blink (default: 3)
        4. **Click "Analyze Blink Rate"** to process the video
        
        ### Expected Video Characteristics:
        - Clear view of the eye
        - Consistent lighting
        - Monochromatic/grayscale preferred
        - Minimum 15 FPS recommended
        - Stable camera position
        
        ### How it works:
        The app uses Eye Aspect Ratio (EAR) to detect blinks. EAR measures the ratio of eye landmarks.
        During a blink, EAR decreases significantly. The algorithm detects these drops as blinks.
        """)

if __name__ == "__main__":
    main()