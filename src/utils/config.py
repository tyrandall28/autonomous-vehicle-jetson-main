#!/usr/bin/env python3
"""
Configuration settings for the autonomous vehicle system
"""

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640
CAMERA_FPS = 30

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.8
INPUT_SIZE = (640, 640)

# Target objects for autonomous driving
TARGET_OBJECTS = {
    'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 
    'traffic light', 'stop sign', 'bottle', 'cup'
}

# Model paths
DEFAULT_MODEL_PATH = 'models/yolov5n_fp16.engine'

# Performance settings
PROCESS_EVERY_N_FRAMES = 8  # Process every 8th frame for performance 