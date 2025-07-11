# Autonomous Vehicle Vision System

This is my attempt at building a real-time object detection system for autonomous vehicles using a Jetson Nano. The goal is to detect people, cars, traffic signs, and other relevant objects at a decent frame rate. Ultimately, the goal is to actually construct a fully autonomous vehicle that can be given a "mission" to search for a specific object and return to the starting location either directly after successfully locating the target, or after searching the whole area and determining the target is not present.

## Current Status

Started with a YOLOv5 model that was running terribly slow (2.7 FPS peak) and wasn't showing any bounding boxes. After a lot of debugging and optimization work:

- Fixed the coordinate scaling bug that was preventing bounding boxes from appearing
- Cleaned up a bunch of unnecessary code complexity 
- Replaced slow custom NMS with OpenCV's vectorized version
- Got performance up to around 7.5-12 FPS on the Jetson Nano Dev Kit 4GB, which is much more usable

The system now properly detects and displays bounding boxes for target objects like people, vehicles, traffic lights, stop signs, etc.

## What's Working

- Camera input from CSI camera on Jetson Nano
- TensorRT optimized inference (FP16)
- Real-time object detection with bounding box visualization
- Targets relevant objects for autonomous driving scenarios

## Files

- `vision.py` - Main vision system (original messy version)
- `vision_clean.py` - Cleaned up version with optimizations
- `convert_to_tensorrt.py` - Script to convert ONNX models to TensorRT engines
- Other utility scripts for testing and configuration
- Placeholder files for future implementation of the driving functions

## Notes

The biggest issue was that TensorRT was outputting pixel coordinates but the code was treating them as normalized coordinates, leading to completely wrong bounding box positions. Once that was fixed, everything started working much better.

Still working on getting closer to the target 15 FPS, but it's functional now for basic autonomous vehicle vision tasks.