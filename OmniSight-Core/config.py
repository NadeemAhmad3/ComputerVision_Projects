# config.py
"""
Configuration file for OmniSight-Core System
Centralized settings for easy tuning and optimization
"""

# ============================================================================
# MODEL PATHS
# ============================================================================
AGE_MODEL = "weights/age_net.caffemodel"
AGE_PROTO = "weights/deploy_age2.prototxt"
GENDER_MODEL = "weights/gender_net.caffemodel"
GENDER_PROTO = "weights/deploy_gender2.prototxt"

# ============================================================================
# MODEL CONSTANTS
# ============================================================================
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']
MODEL_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

# ============================================================================
# DETECTION SETTINGS
# ============================================================================
# Face detection confidence threshold (0.0 to 1.0)
# Higher = fewer false positives, but might miss some faces
MIN_FACE_CONFIDENCE = 0.7

# Prediction confidence threshold
# Predictions below this will be ignored
MIN_PREDICTION_CONFIDENCE = 0.3

# Smoothing buffer size
# Higher = smoother but slower to respond to changes
SMOOTHER_BUFFER_SIZE = 8

# Gender consensus threshold (0.0 to 1.0)
# Percentage required to change gender prediction
GENDER_CONSENSUS_THRESHOLD = 0.6

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================
# Camera resolution
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Process every Nth frame (1 = every frame, 2 = every other frame, etc.)
# Higher = better performance but less responsive
PREDICTION_INTERVAL = 2

# Target FPS (for camera capture)
TARGET_FPS = 30

# Apply noise reduction filter
USE_BILATERAL_FILTER = True

# ============================================================================
# UI SETTINGS
# ============================================================================
# Corner line length for bounding box
CORNER_LINE_LENGTH = 30

# Bounding box line thickness
BOX_THICKNESS = 3

# Colors (BGR format)
COLOR_MALE = (255, 191, 0)      # Cyber Yellow/