# Blink Rate Analyzer

A Streamlit web application for analyzing blink rates from monochromatic eye-tracking videos recorded with VR headset side cameras.

## Features

- Upload eye-tracking videos (MP4, MOV, AVI, MKV)
- Automatic blink detection using Eye Aspect Ratio (EAR)
- Real-time blink rate calculation (blinks per minute)
- Interactive visualizations of EAR signal and blink patterns
- Minute-by-minute blink rate analysis
- Results export functionality

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download the facial landmark predictor:
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bunzip2 shape_predictor_68_face_landmarks.dat.bz2