# main.py
import cv2
import time
import os
from src.detector import FaceDetector
from src.smoother import OmniSmoother

# ============================================================================
# CONFIGURATION
# ============================================================================
AGE_MODEL = "weights/age_net.caffemodel"
AGE_PROTO = "weights/deploy_age2.prototxt"
GENDER_MODEL = "weights/gender_net.caffemodel"
GENDER_PROTO = "weights/deploy_gender2.prototxt"

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

# Performance settings
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
PREDICTION_INTERVAL = 2  # Process every Nth frame (set to 1 for real-time)
MIN_FACE_CONFIDENCE = 0.7
MIN_PREDICTION_CONFIDENCE = 0.3  # Threshold for age/gender confidence

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_model_files():
    """Check if all required model files exist"""
    files = [AGE_MODEL, AGE_PROTO, GENDER_MODEL, GENDER_PROTO]
    for file_path in files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
    print("[✓] All model files validated")

def draw_cyberpunk_box(frame, x, y, w, h, color, label, confidence=None):
    """
    Draws a stylish cyberpunk-style bounding box with label.
    
    Args:
        frame: Image to draw on
        x, y, w, h: Bounding box coordinates
        color: RGB color tuple
        label: Text to display
        confidence: Optional confidence score to display
    """
    l = 30  # Corner line length
    thickness = 3
    
    # Draw corner lines (top-left, top-right, bottom-left, bottom-right)
    cv2.line(frame, (x, y), (x + l, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + l), color, thickness)
    cv2.line(frame, (x + w, y), (x + w - l, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + l), color, thickness)
    cv2.line(frame, (x, y + h), (x + l, y + h), color, thickness)
    cv2.line(frame, (x, y + h), (x, y + h - l), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w - l, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - l), color, thickness)
    
    # Draw label background
    label_text = label
    if confidence:
        label_text += f" [{confidence:.0%}]"
    
    (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    label_w = text_w + 20
    
    cv2.rectangle(frame, (x, y - 40), (x + label_w, y), color, -1)
    cv2.putText(frame, label_text, (x + 10, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

def get_face_id(x, y, grid_size=50):
    """
    Creates a simple face ID based on grid position for tracking.
    
    Args:
        x, y: Face position
        grid_size: Size of grid cells for ID generation
        
    Returns:
        String identifier for the face
    """
    return f"{x // grid_size}_{y // grid_size}"

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    print("=" * 60)
    print("  OmniSight-Core v2.0 - Age & Gender Detection System")
    print("=" * 60)
    
    # Validate model files
    try:
        validate_model_files()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    
    # Initialize face detector
    print("[INFO] Initializing Face Detector...")
    detector = FaceDetector(min_confidence=MIN_FACE_CONFIDENCE)
    
    # Track smoothers per face (for multiple face support)
    face_smoothers = {}
    face_predictions = {}  # Store last predictions per face
    
    # Load Deep Neural Networks
    print("[INFO] Loading AI Models...")
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    
    # Optimize DNN performance
    age_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    age_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("[✓] Models loaded and optimized")
    
    # Initialize video capture
    print("[INFO] Starting camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot access camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("[✓] Camera initialized")
    print("\n[INFO] System ready! Press 'ESC' to quit, 'R' to reset tracking\n")
    
    # Performance tracking
    frame_count = 0
    prev_time = time.time()
    fps = 0
    
    # Main processing loop
    while True:
        success, frame = cap.read()
        if not success:
            print("[WARNING] Failed to read frame")
            break
        
        frame_count += 1
        
        # Apply light image enhancement
        frame = cv2.bilateralFilter(frame, 5, 50, 50)
        
        # Detect faces in every frame
        faces = detector.find_faces(frame)
        
        # Process predictions based on interval
        should_predict = (frame_count % PREDICTION_INTERVAL == 0)
        
        for face in faces:
            x, y, w, h = face['bbox']
            roi = face['roi']
            face_conf = face.get('confidence', 0)
            
            # Validate ROI size
            if roi.shape[0] < 20 or roi.shape[1] < 20:
                continue
            
            # Generate face ID for tracking
            face_id = get_face_id(x, y)
            
            # Initialize smoother for new faces
            if face_id not in face_smoothers:
                face_smoothers[face_id] = OmniSmoother(buffer_size=8)
                face_predictions[face_id] = {'age': None, 'gender': None}
            
            # Run prediction at specified intervals
            if should_predict:
                try:
                    # Prepare image blob for neural network
                    roi_resized = cv2.resize(roi, (227, 227))
                    blob = cv2.dnn.blobFromImage(
                        roi_resized, 1.0, (227, 227),
                        MODEL_MEAN, swapRB=False
                    )
                except Exception as e:
                    print(f"[WARNING] Blob creation failed: {e}")
                    continue

                # Predict Gender
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender_conf = float(gender_preds[0].max())
                gender = GENDER_LIST[gender_preds[0].argmax()]

                # Predict Age
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age_conf = float(age_preds[0].max())
                age = AGE_LIST[age_preds[0].argmax()]
                
                # Only use predictions with sufficient confidence
                if gender_conf > MIN_PREDICTION_CONFIDENCE and age_conf > MIN_PREDICTION_CONFIDENCE:
                    # Update smoother with new predictions
                    final_age, final_gender = face_smoothers[face_id].update_and_get_average(age, gender)
                    face_predictions[face_id] = {
                        'age': final_age, 
                        'gender': final_gender,
                        'confidence': (gender_conf + age_conf) / 2
                    }
                elif face_predictions[face_id]['age'] is None:
                    # First prediction, use it even if confidence is low
                    final_age, final_gender = face_smoothers[face_id].update_and_get_average(age, gender)
                    face_predictions[face_id] = {
                        'age': final_age, 
                        'gender': final_gender,
                        'confidence': (gender_conf + age_conf) / 2
                    }
            
            # Retrieve stored predictions
            if face_predictions[face_id]['age'] is not None:
                final_age = face_predictions[face_id]['age']
                final_gender = face_predictions[face_id]['gender']
                avg_confidence = face_predictions[face_id].get('confidence', 0)
                
                # Choose color based on gender
                color = (255, 191, 0) if final_gender == 'Male' else (203, 192, 255)
                
                # Draw visualization
                label = f"{final_gender} | {final_age}"
                draw_cyberpunk_box(frame, x, y, w, h, color, label, avg_confidence)
        
        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else fps
        prev_time = curr_time
        
        # Draw status overlay
        status_color = (0, 255, 0)
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Faces: {len(faces)}", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Display frame
        cv2.imshow("OmniSight-Core v2.0", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to quit
            print("\n[INFO] Shutting down...")
            break
        elif key == ord('r') or key == ord('R'):  # Reset tracking
            face_smoothers.clear()
            face_predictions.clear()
            print("[INFO] Tracking reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("[✓] System shutdown complete")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Fatal error: {e}")
        import traceback
        traceback.print_exc()