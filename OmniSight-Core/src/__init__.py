# src/__init__.py
"""
OmniSight-Core - Age and Gender Detection System
"""

from .detector import FaceDetector
from .smoother import OmniSmoother

__all__ = ['FaceDetector', 'OmniSmoother']
__version__ = '2.0.0'
__author__ = 'OmniSight Team'