#!/usr/bin/env python3
"""
Debug version to fix TensorRT output format issues
"""

import cv2
import numpy as np
import time
import argparse

# TensorRT and PyCUDA imports
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TensorRTInference:
    def __init__(self, engine_path):
        """Initialize TensorRT inference engine"""
        self.logger = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            size = trt.volume(self.engine.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            
            print(f"Binding: {binding}, Index: {binding_idx}, Size: {size}, Shape: {self.engine.get_binding_shape(binding_idx)}, DType: {dtype}")
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def infer(self, input_data):
        """Run inference"""
        # Copy input data to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        # Run inference
        self.context.execute_v2(bindings=self.bindings)
        
        # Copy output back to host
        output = []
        for out in self.outputs:
            cuda.memcpy_dtoh(out['host'], out['device'])
            output.append(out['host'].copy())
        
        return output

class DebugVisionSystem:
    def __init__(self, model_path="yolov5n_fp16.engine", confidence_threshold=0.6):
        """Initialize debug vision system with higher confidence threshold"""
        self.confidence_threshold = confidence_threshold  # Higher threshold to reduce spam
        self.input_size = (640, 640)
        
        print("Loading TensorRT engine...")
        self.inference_engine = TensorRTInference(model_path)
        print("TensorRT engine loaded successfully!")
        
        # COCO class names (simplified)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
        
        # Target objects
        self.target_objects = {
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 
            'traffic light', 'stop sign', 'bottle', 'cup'
        }
        
        # Initialize camera
        self.cap = None
        self.initialize_camera()
        
    def initialize_camera(self):
        """Initialize camera"""
        print("Initializing camera...")
        
        gst_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=(int)640, height=(int)640, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        )
        
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("CSI camera failed, trying USB camera...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open any camera")
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Camera initialized successfully")
        
    def debug_raw_output(self, outputs, frame_shape):
        """Debug raw TensorRT output to understand format"""
        print(f"\n=== RAW OUTPUT DEBUG ===")
        print(f"Frame shape: {frame_shape}")
        print(f"Number of outputs: {len(outputs)}")
        
        for i, output in enumerate(outputs):
            print(f"Output {i}: shape={output.shape}, dtype={output.dtype}")
            print(f"Output {i}: min={output.min():.3f}, max={output.max():.3f}")
            
            # Show first few detections for analysis
            if output.ndim == 1:
                output_2d = output.reshape(-1, 85) if len(output) % 85 == 0 else output.reshape(-1, output.shape[0]//25200)
            elif output.ndim == 3:
                output_2d = output[0]
            else:
                output_2d = output
                
            print(f"Reshaped to: {output_2d.shape}")
            
            # Show first 5 detections
            for j in range(min(5, len(output_2d))):
                det = output_2d[j]
                if len(det) >= 6:
                    print(f"  Detection {j}: x={det[0]:.3f}, y={det[1]:.3f}, w={det[2]:.3f}, h={det[3]:.3f}, conf={det[4]:.3f}, max_class_conf={det[5:].max():.3f}")
        
        print("========================\n")
    
    def detect_objects_debug(self, frame):
        """Debug object detection"""
        # Create blob
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1/255.0,
            size=self.input_size,
            swapRB=True,
            crop=False
        )
        
        # Run inference
        outputs = self.inference_engine.infer(blob)
        
        # Debug raw output ONLY for first frame
        if not hasattr(self, 'debug_done'):
            self.debug_raw_output(outputs, frame.shape[:2])
            self.debug_done = True
        
        return self._process_detections_debug(outputs, frame.shape[:2])
    
    def _process_detections_debug(self, outputs, frame_shape):
        """Process detections with extensive debugging"""
        height, width = frame_shape
        detections = []
        high_conf_count = 0
        total_count = 0
        
        print(f"Processing frame {width}x{height}")
        
        for output in outputs:
            # Handle TensorRT output format
            if output.ndim == 1:
                output = output.reshape(-1, 85)
            elif output.ndim == 3:
                output = output[0]
            
            for detection in output:
                total_count += 1
                
                if len(detection) < 85:
                    continue
                
                # Get confidence scores
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Count high confidence
                if confidence > 0.3:
                    high_conf_count += 1
                    
                    # Debug first few high confidence detections
                    if high_conf_count <= 3:
                        print(f"High conf detection: raw_coords=({detection[0]:.3f},{detection[1]:.3f},{detection[2]:.3f},{detection[3]:.3f}), conf={confidence:.3f}")
                
                # Early exit for low confidence
                if confidence <= self.confidence_threshold:
                    continue
                
                # Get class name
                if class_id >= len(self.class_names):
                    continue
                class_name = self.class_names[class_id]
                
                # Filter by target objects
                if class_name not in self.target_objects:
                    continue
                
                # TRY DIFFERENT COORDINATE INTERPRETATIONS
                # Method 1: Assume normalized coordinates (0-1)
                center_x1 = int(detection[0] * width)
                center_y1 = int(detection[1] * height)
                w1 = int(detection[2] * width)
                h1 = int(detection[3] * height)
                
                # Method 2: Assume coordinates are already in pixels
                center_x2 = int(detection[0])
                center_y2 = int(detection[1])
                w2 = int(detection[2])
                h2 = int(detection[3])
                
                # Method 3: Assume coordinates need different scaling
                center_x3 = int(detection[0] * width / 640)  # Scale to input size
                center_y3 = int(detection[1] * height / 640)
                w3 = int(detection[2] * width / 640)
                h3 = int(detection[3] * height / 640)
                
                print(f"Coordinate debug for {class_name} (conf={confidence:.3f}):")
                print(f"  Raw: ({detection[0]:.3f},{detection[1]:.3f},{detection[2]:.3f},{detection[3]:.3f})")
                print(f"  Method 1 (norm*size): center=({center_x1},{center_y1}), size=({w1},{h1})")
                print(f"  Method 2 (direct): center=({center_x2},{center_y2}), size=({w2},{h2})")
                print(f"  Method 3 (scale/640): center=({center_x3},{center_y3}), size=({w3},{h3})")
                
                # Use method that gives reasonable coordinates (within frame bounds)
                if 0 <= center_x2 <= width and 0 <= center_y2 <= height:
                    # Method 2 looks good
                    center_x, center_y, w, h = center_x2, center_y2, w2, h2
                    print(f"  Using Method 2 (direct)")
                elif 0 <= center_x3 <= width and 0 <= center_y3 <= height:
                    # Method 3 looks good
                    center_x, center_y, w, h = center_x3, center_y3, w3, h3
                    print(f"  Using Method 3 (scale/640)")
                else:
                    # Default to method 1 but clamp
                    center_x = max(0, min(width, center_x1))
                    center_y = max(0, min(height, center_y1))
                    w = max(1, min(width, w1))
                    h = max(1, min(height, h1))
                    print(f"  Using Method 1 (clamped)")
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Clamp to frame bounds
                x = max(0, min(width - w, x))
                y = max(0, min(height - h, y))
                w = min(width - x, w)
                h = min(height - y, h)
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2)
                })
                
                print(f"  Final bbox: ({x},{y},{w},{h})")
                
                # Limit detections to prevent spam
                if len(detections) >= 5:
                    break
            
            if len(detections) >= 5:
                break
        
        print(f"Summary: {total_count} total, {high_conf_count} high confidence, {len(detections)} final")
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw detection results"""
        for detection in detections:
            x, y, w, h = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def run_debug(self):
        """Run debug vision system"""
        print("Starting debug vision system...")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("Press 'q' to quit")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            
            # Only process every 30th frame to reduce spam
            if frame_count % 30 == 0:
                print(f"\n=== FRAME {frame_count} ===")
                detection_start = time.time()
                detections = self.detect_objects_debug(frame)
                detection_time = time.time() - detection_start
                
                print(f"Detection time: {detection_time*1000:.1f}ms")
                print(f"Found {len(detections)} final detections")
            else:
                detections = []
            
            # Draw results
            frame = self.draw_detections(frame, detections)
            
            # Add status
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detections: {len(detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Debug Vision System', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Debug Vision System')
    parser.add_argument('--model', default='yolov5n_fp16.engine', help='Path to model file')
    parser.add_argument('--confidence', type=float, default=0.7, help='Confidence threshold')
    args = parser.parse_args()
    
    vision = None
    try:
        vision = DebugVisionSystem(args.model, args.confidence)
        vision.run_debug()
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if vision:
            vision.cleanup()

if __name__ == "__main__":
    main() 