import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import matplotlib.pyplot as plt

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def calculate_ear(eye):
    """
    Calculate Eye Aspect Ratio (EAR)
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Compute vertical distances
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    
    # Compute horizontal distance
    C = distance.euclidean(eye[0], eye[3])
    
    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    return ear

def process_video(video_path, ear_threshold=0.2, max_frames=None):
    """
    Process video and extract EAR values
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ear_values = []
    timestamps = []
    processed_frames = []
    
    frame_count = 0
    success = True
    
    while success:
        success, frame = cap.read()
        
        if not success:
            break
            
        if max_frames and frame_count >= max_frames:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray, 0)
        
        if len(faces) > 0:
            # Assume first face is the target
            face = faces[0]
            
            # Get facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)
            
            # Extract eye coordinates
            left_eye = shape[42:48]
            right_eye = shape[36:42]
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            
            # Average EAR
            ear = (left_ear + right_ear) / 2.0
            
            # Check if this is a blink frame
            is_blink = ear < ear_threshold
            
            # Draw eyes and EAR on frame for preview
            preview_frame = frame.copy()
            
            # Draw left eye
            left_eye_hull = cv2.convexHull(left_eye)
            cv2.drawContours(preview_frame, [left_eye_hull], -1, (0, 255, 0), 1)
            
            # Draw right eye
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(preview_frame, [right_eye_hull], -1, (0, 255, 0), 1)
            
            # Add EAR text
            cv2.putText(preview_frame, f"EAR: {ear:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if is_blink:
                cv2.putText(preview_frame, "BLINK", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            processed_frames.append((preview_frame, ear, is_blink))
            
        else:
            # No face detected
            ear = 0
            preview_frame = frame.copy()
            cv2.putText(preview_frame, "NO FACE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            processed_frames.append((preview_frame, 0, False))
        
        ear_values.append(ear)
        timestamps.append(frame_count / fps if fps > 0 else frame_count)
        frame_count += 1
    
    cap.release()
    return ear_values, timestamps, processed_frames

def detect_blinks(ear_values, ear_threshold=0.2, min_blink_frames=3):
    """
    Detect blinks from EAR values
    Returns list of blinks as (frame_index, start_frame, end_frame)
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
            in_blink = False
            blink_duration = i - blink_start
            
            if blink_duration >= min_blink_frames:
                # Valid blink detected
                blinks.append((blink_start, blink_start, i))
    
    # Handle case where blink continues until end of video
    if in_blink:
        blink_duration = len(ear_values) - blink_start
        if blink_duration >= min_blink_frames:
            blinks.append((blink_start, blink_start, len(ear_values)))
    
    return blinks

# Alternative simple EAR calculation for when dlib fails
def simple_ear_calculation(frame):
    """
    Simple alternative EAR calculation using basic image processing
    This is a fallback when facial landmarks aren't available
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Simple thresholding to find dark regions (pupil/closed eye)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Calculate ratio of dark pixels in central region (rough eye area estimate)
    height, width = gray.shape
    roi = thresh[height//3:2*height//3, width//3:2*width//3]
    
    if roi.size > 0:
        dark_ratio = np.sum(roi == 255) / roi.size
        # Convert to EAR-like metric (inverse relationship)
        ear = max(0.1, 1.0 - dark_ratio)
        return ear
    
    return 0.2  # Default value