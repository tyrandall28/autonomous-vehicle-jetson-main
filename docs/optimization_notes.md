# Jetson Nano Vision System Optimization Guide

## Overview
This guide outlines the steps taken to optimize the autonomous vehicle vision system to achieve **5-15 FPS** (up from 1.1 FPS) on Jetson Nano 4GB.

## Performance Improvements Summary

| Optimization | Expected FPS Gain | Implementation Difficulty |
|-------------|------------------|-------------------------|
| TensorRT Engine | 3-5x faster | Medium |
| YOLOv5n Model | 2-3x faster | Easy |
| Input Resolution Reduction | 2x faster | Easy |
| Frame Skipping | 1.5-2x faster | Easy |
| Threading | 1.2-1.5x faster | Medium |
| System Optimization | 1.2-1.3x faster | Easy |

**Combined Expected Performance: 8-15 FPS** (vs original 1.1 FPS)

## Quick Start

### 1. Install Dependencies
```bash
pip3 install -r requirements_optimized.txt
```

### 2. Convert Model to TensorRT
```bash
# Download optimized YOLOv5n model and convert
python3 convert_to_tensorrt.py --download --precision fp16

# Or convert existing model
python3 convert_to_tensorrt.py --onnx yolov5s.onnx --precision fp16
```

### 3. Run Optimized Vision System
```bash
# With TensorRT engine (best performance)
python3 vision.py --model yolov5n_fp16.engine --tensorrt

# Or with optimized OpenCV DNN
python3 vision.py --model yolov5n.onnx --confidence 0.6
```

## Detailed Optimizations

### 1. Model Optimizations

#### Switch to YOLOv5n (Nano)
- **Gain**: 2-3x faster inference
- **Trade-off**: Slightly lower accuracy
- **Implementation**: Use `--download` flag in conversion script

#### TensorRT Conversion
- **Gain**: 3-5x faster inference
- **Benefits**: GPU-optimized inference, FP16 precision
- **Memory**: Reduced from FP32 to FP16

#### Reduced Input Resolution
- **Original**: 640x640 pixels
- **Optimized**: 416x416 pixels
- **Gain**: ~2x faster processing

### 2. System-Level Optimizations

#### Frame Processing Strategy
```python
# Process every 3rd frame for detection
self.process_every_n_frames = 3

# Use threading for async processing
detection_thread = threading.Thread(target=self.async_detection_worker, daemon=True)
```

#### Camera Pipeline Optimization
- Reduced resolution in GStreamer pipeline
- Frame dropping when processing is slow
- Smaller buffer sizes

#### GPU Memory Management
- Use CUDA preprocessing when available
- Optimized memory allocation
- Reduced workspace memory for TensorRT

### 3. Algorithm Optimizations

#### Early Exit Strategies
```python
# Skip low-confidence detections early
if confidence <= self.confidence_threshold:
    continue

# Filter to autonomous driving relevant objects only
if class_name not in self.target_objects:
    continue
```

#### Simplified Post-Processing
- Reduced NMS overhead
- Simplified drawing operations
- Focused on essential objects only

## Performance Monitoring

### Monitor System Performance
```bash
# Real-time system stats
sudo tegrastats

# Detailed monitoring
jtop

# Monitor GPU utilization
nvidia-smi
```

### Expected Thermal Behavior
- **Idle**: ~45-50°C
- **Under Load**: ~60-70°C
- **Throttling**: Starts at ~80°C

## Hardware Recommendations

### Essential for Maximum Performance
1. **Active Cooling**: Fan + heatsink
2. **Power Supply**: 5V 4A minimum
3. **High-Speed SD Card**: Class 10, U3
4. **Good Ventilation**: Prevent thermal throttling

### Camera Optimization
```python
# CSI Camera (recommended)
gst_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=416, height=416 ! "
    "videoconvert ! "
    "appsink max-buffers=1 drop=true"
)
```

## Troubleshooting

### Low FPS Issues
1. **Check thermal throttling**: `sudo tegrastats`
2. **Verify power mode**: `sudo nvpmodel -q`
3. **Monitor GPU usage**: `nvidia-smi`
4. **Check memory usage**: `free -h`

### TensorRT Issues
```bash
# Verify TensorRT installation
python3 -c "import tensorrt; print(tensorrt.__version__)"

# Check CUDA availability
python3 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

### Memory Issues
```bash
# Check available memory
free -h

# Disable swap if needed
sudo swapoff -a

# Increase GPU memory split
sudo vi /boot/config.txt  # Add gpu_mem=128
```

## Code Architecture Changes

### Original vs Optimized
| Component | Original | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| Model | YOLOv5s 640x640 | YOLOv5n 416x416 + TensorRT | 5-8x faster |
| Processing | Every frame | Every 3rd frame + threading | 2-3x faster |
| Post-processing | Full pipeline | Early exits + filtering | 1.5x faster |
| Camera | Basic pipeline | Optimized GStreamer | 1.2x faster |

### Threading Architecture
```
Main Thread (Camera) -> Frame Queue -> Detection Thread
                    \-> Display Thread
                    \-> Navigation Thread
```

## Expected Results

### Performance Targets
- **Minimum**: 5 FPS (sufficient for basic autonomy)
- **Target**: 8-12 FPS (good for autonomous vehicle)
- **Optimal**: 12-15 FPS (excellent real-time performance)

### Accuracy Trade-offs
- **YOLOv5s -> YOLOv5n**: ~5-10% accuracy reduction
- **640x640 -> 416x416**: ~3-5% accuracy reduction
- **Frame skipping**: No accuracy loss (temporal redundancy)

## Next Steps for Further Optimization

1. **INT8 Quantization**: Further 1.5-2x speedup
2. **Custom Model Training**: Smaller, task-specific models
3. **Multi-threading**: Separate camera, processing, and control threads
4. **Hardware Upgrade**: Consider running models on more powerful hardware like Jetson Orin

## Files Overview

- `vision.py` - Optimized vision system with TensorRT support
- `convert_to_tensorrt.py` - Model conversion utility
- `jetson_performance_setup.sh` - System optimization script
- `requirements_optimized.txt` - Python dependencies
- `OPTIMIZATION_README.md` - This guide

## Support

For issues or questions:
1. Check the troubleshooting section
2. Monitor system performance with `jtop` or `tegrastats`
3. Verify all optimizations are applied
4. Consider hardware limitations