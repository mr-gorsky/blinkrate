import streamlit as st
import base64

st.set_page_config(
    page_title="Blink Rate Analyzer",
    page_icon="👁️",
    layout="wide"
)

def main():
    st.title("👁️ Blink Rate Analyzer")
    
    st.markdown("""
    ## Upload your eye-tracking video for analysis
    
    This app analyzes blink rates from VR headset eye-tracking videos.
    
    **Features:**
    - Blink detection using computer vision
    - Blink rate calculation (blinks per minute)
    - Time-series visualization
    - Exportable results
    
    **Supported formats:** MP4, MOV, AVI
    """)
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])
    
    if uploaded_file is not None:
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Show file info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        st.write(f"File size: {file_size:.2f} MB")
        
        # Demo analysis
        st.subheader("Demo Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Estimated Blinks", "24")
        col2.metric("Blink Rate", "18.5/min") 
        col2.metric("Analysis Status", "Complete")
        
        st.info("""
        **Note:** This is a demo interface. The full version includes:
        - Actual video processing with OpenCV
        - Real blink detection algorithms
        - Interactive graphs
        - Detailed frame-by-frame analysis
        """)

if __name__ == "__main__":
    main()
