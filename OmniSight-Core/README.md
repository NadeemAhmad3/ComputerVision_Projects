# OmniSight-Core 👁️

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=flat&logo=opencv)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Detection-orange?style=flat&logo=google)](https://developers.google.com/mediapipe)

**OmniSight-Core** is a high-performance, real-time computer vision system designed for accurate demographic analysis. It leverages Google's MediaPipe for robust face detection and Caffe Deep Learning models to classify age and gender from live webcam feeds.

Unlike traditional implementations that suffer from prediction "jitter," OmniSight includes a custom smoothing engine (`OmniSmoother`) to provide stable, reliable outputs in real-time environments.

## ✨ Features

- **🚀 Real-Time Performance:** Optimized for low latency inference on standard CPU hardware.
- **🎯 Robust Detection:** Uses MediaPipe (SSD architecture) for superior face tracking compared to Haar Cascades.
- **🧠 Stable Predictions:** Custom temporal smoothing algorithm reduces flickering of age/gender labels.
- **🎨 Cyberpunk UI:** Custom visual overlay with futuristic bounding boxes and HUD elements.
- **⚙️ Configurable:** Centralized `config.py` for easy tuning of thresholds, resolution, and model parameters.

## 📂 Project Structure

```
OmniSight-Core/
├── config.py           # Central configuration settings
├── main.py             # Entry point of the application
├── requirements.txt    # Python dependencies
├── src/                # Source code modules
│   ├── detector.py     # MediaPipe face detection wrapper
│   └── smoother.py     # Temporal data smoothing logic
└── weights/            # (Required) Deep learning model files
    ├── age_net.caffemodel      # <--- Download this
    ├── gender_net.caffemodel   # <--- Download this
    ├── deploy_age2.prototxt
    └── deploy_gender2.prototxt
```

## 🛠️ Installation

### 1. Prerequisite
Ensure you have **Python 3.10** or higher installed.

### 2. Clone the Repository
```bash
git clone https://github.com/yourusername/OmniSight-Core.git
cd OmniSight-Core
```

### 3. Set up Virtual Environment
It is recommended to use a virtual environment.
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## ⚠️ Important: Model Weights

**Note:** The bulky `.caffemodel` weights files are **NOT** included in this repository to keep it light. You must download them manually and place them in the `weights/` directory.

| File | Description | Download Link |
|------|-------------|---------------|
| `age_net.caffemodel` | Age Classification Weights | [Download (Dropbox)](https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=0) |
| `gender_net.caffemodel` | Gender Classification Weights | [Download (Dropbox)](https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=0) |

The `.prototxt` files (architecture definitions) *should* be included in the repo. If they are missing, you can find them in the [OpenCV AgeGender repo](https://github.com/spmallick/learnopencv/tree/master/AgeGender).

Ensure your `weights/` folder looks like this before running:
```
weights/
├── age_net.caffemodel
├── deploy_age2.prototxt
├── deploy_gender2.prototxt
└── gender_net.caffemodel
```

## 🚀 Usage

Once the environment is set up and weights are downloaded, run the application:

```bash
python main.py
```

### Controls
- **`q`**: Quit the application.

## ⚙️ Configuration

You can fine-tune the system performance in `config.py`:
- **`PREDICTION_INTERVAL`**: Set to 1 for every frame, or higher to improve FPS.
- **`MIN_FACE_CONFIDENCE`**: Adjust detection sensitivity (0.7 default).
- **`SMOOTHER_BUFFER_SIZE`**: Increase for less jitter, decrease for faster response.

## 📚 Credits

- **Face Detection:** [Google MediaPipe](https://developers.google.com/mediapipe)
- **Age & Gender Models:** Gil Levi and Tal Hassner (OpenCV)
- **Inspiration:** [LearnOpenCV](https://learnopencv.com/age-gender-classification-using-opencv-deep-learning-c-python/)

## 📄 License
MIT License - feel free to use and modify for your own projects.
