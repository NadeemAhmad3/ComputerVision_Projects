# OmniSight-Core v1.0 👁️

**OmniSight-Core** is a high-performance, real-time demographic analysis system. It utilizes Google's MediaPipe for robust face detection and Caffe Deep Learning models for Age and Gender classification.

## 🚀 Features
- **Real-Time Processing:** Works instantly on live webcam feeds.
- **Robust Detection:** Uses MediaPipe (SSD) instead of Haar Cascades for superior accuracy.
- **Jitter Smoothing:** Custom `OmniSmoother` engine to stabilize AI predictions over time.
- **Modular Architecture:** Clean separation of concerns (Detector, Smoother, Main).

## 🛠️ Installation

1. **Clone the repo**
2. **Create a Virtual Environment (Python 3.10)**
   ```bash
   py -3.10 -m venv venv
   .\venv\Scripts\activate