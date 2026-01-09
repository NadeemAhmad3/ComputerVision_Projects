# src/detector.py
import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self, min_confidence=0.7):
        """
        Initialize face detector with MediaPipe.
        
        Args:
            min_confidence: Minimum detection confidence (0.0 to 1.0)
        """
        self.mp_face_detection = mp.solutions.face_detection
        # model_selection=1 for full range detection (better for varying distances)
        # model_selection=0 for close range (within 2 meters)
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_confidence,
            model_selection=1  # Better for general use
        )

    def find_faces(self, img):
        """
        Detects faces in an image and returns their bounding boxes and ROIs.
        
        Args:
            img: BGR image from OpenCV
            
        Returns:
            List of dictionaries containing 'bbox' and 'roi' for each face
            Format: [{'bbox': (x, y, w, h), 'roi': cropped_face_img}, ...]
        """
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(img_rgb)
        
        faces_data = []
        
        if results.detections:
            h, w, _ = img.shape
            
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                
                # Convert relative coordinates to pixel values
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)

                # Add intelligent padding to capture full head
                # Use percentage-based padding for better scaling
                pad_w = int(width * 0.2)   # 20% horizontal padding
                pad_h = int(height * 0.3)  # 30% vertical padding (capture forehead)
                
                # Calculate padded coordinates with bounds checking
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(w, x + width + pad_w)
                y2 = min(h, y + height + pad_h)
                
                # Extract the face ROI (Region of Interest)
                face_img = img[y1:y2, x1:x2]
                
                # Validate ROI dimensions
                # Ensure the face region is large enough to be useful
                if face_img.size > 0 and face_img.shape[0] > 20 and face_img.shape[1] > 20:
                    faces_data.append({
                        'bbox': (x1, y1, x2 - x1, y2 - y1),
                        'roi': face_img,
                        'confidence': detection.score[0]  # Store confidence score
                    })
                    
        return faces_data

    def close(self):
        """Clean up resources"""
        self.face_detection.close()