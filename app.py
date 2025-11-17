import streamlit as st
import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import tempfile
import os
import time

st.set_page_config(
    page_title="Blink Rate Analyzer",
    page_icon="👁️",
    layout="wide"
)

def calculate_eye_openness(frame):
    """
    Calculate eye openness based on image analysis
    Returns value between 0 (closed) and 1 (open)
    """
    # Convert to grayscale
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    height, width = gray.shape
    
    # Define eye region (center of frame)
    eye_region = gray[int(height*0.3):int(height*0.7), int(width*0.3):int(width*0.7)]
    
    if eye_region.size == 0:
        return 0.5
    
    # Calculate metrics for eye openness
    # 1. Variance (open eyes have more texture/variance)
    variance = np.var(eye_region) / 1000
    
    # 2. Edge density (open eyes have more edges)
    edges = cv2.Canny(eye_region, 50, 150)
    edge_density = np.sum(edges > 0) / eye_region.size
    
    # 3. Brightness distribution
    hist = cv2.calcHist([eye_region], [0], None, [64], [0, 256])
    hist_norm = hist / hist.sum()
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-8))
    entropy_norm = entropy / 6  # Normalize
    
    # Combine metrics
    openness = min(1.0, max(0.1, (variance * 0.4 + edge_density * 0.4 + entropy_norm * 0.2)))
    
    return openness

def process_video_for_blinks(video_path, threshold=0.3):
    """
    Process video and detect blinks
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    openness_values = []
    timestamps = []
    blink_events = []
    processed_frames = []
    
    frame_count = 0
    max_frames = 1000  # Limit for performance
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Calculate eye openness
        openness = calculate_eye_openness(frame)
        openness_values.append(openness)
        timestamps.append(frame_count / fps)
        
        # Detect if this is a blink frame
        is_blink = openness < threshold
        
        # Create preview frame (store every 50th frame for performance)
        if frame_count % 50 == 0:
            preview = frame.copy()
            height, width = preview.shape[:2]
            
            # Draw eye region
            cv2.rectangle(preview, 
                         (int(width*0.3), int(height*0.3)),
                         (int(width*0.7), int(height*0.7)),
                         (0, 255, 0), 2)
            
            # Add text
            color = (0, 0, 255) if is_blink else (255, 255, 255)
            cv2.putText(preview, f"Openness: {openness:.3f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            if is_blink:
                cv2.putText(preview, "BLINK DETECTED", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            processed_frames.append((preview_rgb, openness, is_blink))
        
        frame_count += 1
    
    cap.release()
    
    # Detect blink events from openness values
    blinks = detect_blink_events(openness_values, threshold)
    
    return openness_values, timestamps, blinks, processed_frames, fps

def detect_blink_events(openness_values, threshold=0.3, min_duration=3):
    """
    Detect blink events from openness values
    """
    blinks = []
    in_blink = False
    blink_start = 0
    
    for i, openness in enumerate(openness_values):
        if openness < threshold and not in_blink:
            # Start of blink
            in_blink = True
            blink_start = i
        elif openness >= threshold and in_blink:
            # End of blink
            blink_duration = i - blink_start
            if blink_duration >= min_duration:
                blinks.append((blink_start, blink_start, i))
            in_blink = False
    
    # Handle blink at the end
    if in_blink and (len(openness_values) - blink_start) >= min_duration:
        blinks.append((blink_start, blink_start, len(openness_values)))
    
    return blinks

def smooth_values(values, window_size=5):
    """Smooth values using moving average"""
    if len(values) < window_size:
        return values
    
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        window = values[start:end]
        smoothed.append(np.mean(window))
    
    return smoothed

def main():
    st.title("👁️ Blink Rate Analyzer - STVARNA ANALIZA")
    st.markdown("**Uploadaj video oka za stvarnu analizu treptaja**")
    
    uploaded_file = st.file_uploader(
        "Odaberi video snimku oka", 
        type=['mp4', 'mov', 'avi', 'mkv']
    )
    
    if uploaded_file is not None:
        # Save uploaded file
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
            **Informacije o videu:**
            - FPS: {fps:.1f}
            - Ukupno frejmova: {total_frames}
            - Trajanje: {duration:.2f} sekundi
            """)
            
            # Parameters
            st.subheader("Postavke detekcije")
            col1, col2 = st.columns(2)
            with col1:
                sensitivity = st.slider(
                    "Osjetljivost detekcije", 
                    min_value=0.1, 
                    max_value=0.5, 
                    value=0.3, 
                    step=0.05,
                    help="Niže vrijednosti = osjetljivije na treptaje"
                )
            with col2:
                min_duration = st.slider(
                    "Minimalno trajanje treptaja (frejmovi)", 
                    min_value=1, 
                    max_value=10, 
                    value=3,
                    help="Minimalni broj frejmova da se računa kao treptaj"
                )
            
            if st.button("🎯 POKRENI ANALIZU TREPTAJA", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Analyze video
                status_text.text("🔄 Procesiram video...")
                openness_values, timestamps, blinks, processed_frames, actual_fps = process_video_for_blinks(
                    video_path, sensitivity
                )
                progress_bar.progress(50)
                
                status_text.text("🔄 Detektiram treptaje...")
                # Smooth values
                openness_smoothed = smooth_values(openness_values, 5)
                
                # Re-detect blinks with smoothed values
                blinks = detect_blink_events(openness_smoothed, sensitivity, min_duration)
                progress_bar.progress(100)
                
                # Calculate results
                total_blinks = len(blinks)
                blink_rate = (total_blinks / duration) * 60 if duration > 0 else 0
                
                status_text.text(f"✅ Analiza završena! Pronađeno {total_blinks} treptaja")
                
                # Display results
                st.success(f"**REZULTATI ANALIZE:** {total_blinks} treptaja detektirano = {blink_rate:.1f} treptaja/minutu")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Ukupno treptaja", total_blinks)
                col2.metric("Stopa treptaja", f"{blink_rate:.1f}/min")
                col3.metric("Trajanje videa", f"{duration:.1f}s")
                col4.metric("Analiziranih frejmova", len(openness_values))
                
                # Tabs for different views
                tab1, tab2, tab3 = st.tabs(["📊 Graf otvorenosti oka", "⏰ Vremenska linija", "👀 Primjer frejmova"])
                
                with tab1:
                    # Create plot
                    fig = go.Figure()
                    
                    # Openness values
                    fig.add_trace(go.Scatter(
                        x=timestamps, y=openness_smoothed,
                        name='Otvorenost oka',
                        line=dict(color='blue', width=2),
                        hovertemplate='Vrijeme: %{x:.2f}s<br>Otvorenost: %{y:.3f}<extra></extra>'
                    ))
                    
                    # Threshold line
                    fig.add_trace(go.Scatter(
                        x=timestamps, 
                        y=[sensitivity] * len(timestamps),
                        name='Granica treptaja',
                        line=dict(color='red', dash='dash', width=2)
                    ))
                    
                    # Blink markers
                    if blinks:
                        blink_times = [timestamps[b[0]] for b in blinks]
                        blink_values = [openness_smoothed[b[0]] for b in blinks]
                        
                        fig.add_trace(go.Scatter(
                            x=blink_times, y=blink_values,
                            mode='markers',
                            name='Treptaji',
                            marker=dict(color='red', size=10, symbol='x', line=dict(width=2)),
                            hovertemplate='Treptaj @ %{x:.2f}s<extra></extra>'
                        ))
                    
                    fig.update_layout(
                        title="Otvorenost oka kroz vrijeme s detektiranim treptajima",
                        xaxis_title="Vrijeme (sekunde)",
                        yaxis_title="Otvorenost oka (0=zatvoreno, 1=otvoreno)",
                        height=500,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistics
                    st.subheader("Statistika otvorenosti oka")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Prosjek", f"{np.mean(openness_smoothed):.3f}")
                    col2.metric("Minimum", f"{np.min(openness_smoothed):.3f}")
                    col3.metric("Maksimum", f"{np.max(openness_smoothed):.3f}")
                    col4.metric("Standardna devijacija", f"{np.std(openness_smoothed):.3f}")
                
                with tab2:
                    if blinks:
                        st.subheader("Detektirani treptaji")
                        
                        # Create timeline table
                        blink_data = []
                        for i, blink in enumerate(blinks):
                            start_time = timestamps[blink[0]]
                            duration_frames = blink[2] - blink[0]
                            duration_sec = duration_frames / actual_fps
                            openness_val = openness_smoothed[blink[0]]
                            
                            blink_data.append({
                                'Treptaj #': i + 1,
                                'Vrijeme (s)': f"{start_time:.2f}",
                                'Trajanje (s)': f"{duration_sec:.2f}",
                                'Otvorenost': f"{openness_val:.3f}"
                            })
                        
                        # Display as table
                        for blink in blink_data:
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Treptaj", blink['Treptaj #'])
                            col2.metric("Vrijeme", blink['Vrijeme (s)'] + "s")
                            col3.metric("Trajanje", blink['Trajanje (s)'] + "s")
                            col4.metric("Otvorenost", blink['Otvorenost'])
                    
                    else:
                        st.info("Nije detektiran nijedan treptaj. Pokušaj smanjiti osjetljivost.")
                
                with tab3:
                    if processed_frames:
                        st.subheader("Primjer procesiranih frejmova")
                        
                        # Show sample frames
                        cols = st.columns(min(3, len(processed_frames)))
                        for idx, (frame, openness, is_blink) in enumerate(processed_frames[:3]):
                            with cols[idx]:
                                pil_img = Image.fromarray(frame)
                                st.image(pil_img, 
                                       caption=f"Otvorenost: {openness:.3f} {'🔴 TREPTAJ' if is_blink else '⚪ Normal'}",
                                       use_column_width=True)
                    
                    # Analysis info
                    st.subheader("Informacije o analizi")
                    st.write(f"""
                    **Kako funkcionira analiza:**
                    - Analizira se **stvarni video** frejm po frejm
                    - Mjeri se **otvorenost oka** bazirano na teksturi i edge density
                    - **Niska otvorenost** = vjerojatno treptaj
                    - **Granica** ({sensitivity}) određuje što se smatra treptajem
                    - **Minimalno trajanje** ({min_duration} frejmova) eliminira lažne detekcije
                    
                    **Broj analiziranih frejmova:** {len(openness_values)}
                    **Stopa uzorkovanja:** ~{actual_fps:.1f} FPS
                    """)
                
                # Export results
                st.subheader("📥 Preuzmi rezultate")
                results_text = f"""REZULTATI ANALIZE TREPTAJA
================================
Datoteka: {uploaded_file.name}
Ukupno treptaja: {total_blinks}
Stopa treptaja: {blink_rate:.2f} treptaja/minutu
Trajanje videa: {duration:.2f} sekundi
Osjetljivost: {sensitivity}
Minimalno trajanje: {min_duration} frejmova

VREMENA TREPTAJA:
"""
                for i, blink in enumerate(blinks):
                    results_text += f"{i+1:2d}. {timestamps[blink[0]]:.2f}s (otvorenost: {openness_smoothed[blink[0]]:.3f})\n"
                
                st.download_button(
                    "💾 Preuzmi rezultate (.txt)",
                    results_text,
                    file_name=f"blink_analysis_{uploaded_file.name.split('.')[0]}.txt"
                )
                
        except Exception as e:
            st.error(f"Greška pri procesiranju: {str(e)}")
            st.info("""
            **Rješenje problema:**
            1. Pokušaj s kraćim videom (do 30 sekundi)
            2. Provjeri je li oko vidljivo u videu
            3. Podesi osjetljivost (povećaj za manje treptaje, smanji za više)
            4. Pokušaj s MP4 formatom
            """)
        
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
    
    else:
        st.markdown("""
        ### 📋 Upute za korištenje:
        
        1. **Snimi video** oka pomoću VR kamere sa strane headseta
        2. **Uploadaj video** (MP4, MOV, AVI, MKV)
        3. **Podesi postavke** ako je potrebno
        4. **Klikni "POKRENI ANALIZU TREPTAJA"**
        
        ### 🔍 Kako radi analiza:
        - **Stvarno procesira video** frejm po frejm
        - **Analizira otvorenost oka** koristeći computer vision
        - **Detektira treptaje** kada otvorenost padne ispod granice
        - **Računa stopu treptaja** (treptaji/minutu)
        - **Generira grafove** i detaljne rezultate
        
        ### 💡 Savjeti za najbolje rezultate:
        - **Jasno vidljivo oko** u videu
        - **Konzistentno osvjetljenje**
        - **Stabilna kamera**
        - **Video do 2 minute** za bržu obradu
        - **Počni s default postavkama**
        """)

if __name__ == "__main__":
    main()
