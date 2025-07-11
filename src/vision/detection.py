#!/usr/bin/env python3
"""
Autonomous Vehicle Vision System - Simplified and Optimized for Jetson Nano
Using TensorRT for high-performance object detection
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path

# TensorRT and PyCUDA imports (no fallbacks since they work)
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

class SimplifiedVisionSystem:
    def __init__(self, model_path="yolov5n_fp16.engine", confidence_threshold=0.5):
        """Initialize simplified vision system"""
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.8  # Very aggressive NMS to merge overlapping boxes
        self.input_size = (640, 640)
        
        # Load model
        if model_path.endswith('.engine'):
            print("Loading TensorRT engine...")
            self.inference_engine = TensorRTInference(model_path)
            self.use_tensorrt = True
            print("TensorRT engine loaded successfully!")
        else:
            print("Loading OpenCV DNN model...")
            self.net = cv2.dnn.readNet(model_path)
            self._setup_opencv_backend()
            self.use_tensorrt = False
            print("OpenCV DNN model loaded successfully!")
        
        # COCO class names
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
        
        # Target objects for autonomous driving
        self.target_objects = {
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 
            'traffic light', 'stop sign', 'bottle', 'cup', 'laptop', 
            'mouse', 'keyboard', 'cell phone', 'book', 'chair'
        }
        
        # Initialize camera
        self.cap = None
        self.initialize_camera()
        
    def _setup_opencv_backend(self):
        """Setup OpenCV DNN backend for CUDA"""
        print("Using CUDA backend with FP16")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        
    def initialize_camera(self):
        """Initialize camera with proper resolution matching"""
        print("Initializing camera...")
        
        # FIXED: Use 640x640 to match model input size
        gst_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=(int)640, height=(int)640, format=(string)BGRx ! "  # FIXED: 640x640
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        )
        
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("CSI camera failed, trying USB camera...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open any camera")
            
            # FIXED: Set resolution to match model input
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Camera initialized successfully")
        
    def detect_objects(self, frame):
        """Simplified object detection with timing breakdown"""
        start_time = time.time()
        
        # Create blob
        preprocessing_start = time.time()
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1/255.0,
            size=self.input_size,
            swapRB=True,
            crop=False
        )
        preprocessing_time = (time.time() - preprocessing_start) * 1000
        
        # Run inference
        inference_start = time.time()
        if self.use_tensorrt:
            outputs = self.inference_engine.infer(blob)
        else:
            self.net.setInput(blob)
            outputs = self.net.forward()
        inference_time = (time.time() - inference_start) * 1000
        
        # Process detections
        postprocessing_start = time.time()
        detections = self._process_detections(outputs, frame.shape[:2])
        postprocessing_time = (time.time() - postprocessing_start) * 1000
        
        total_time = (time.time() - start_time) * 1000
        
        # Print timing breakdown occasionally
        if hasattr(self, 'timing_counter'):
            self.timing_counter += 1
        else:
            self.timing_counter = 1
            
        if self.timing_counter % 5 == 0:  # Print every 5th detection
            print(f"Timing: Preprocessing={preprocessing_time:.1f}ms, Inference={inference_time:.1f}ms, Postprocessing={postprocessing_time:.1f}ms, Total={total_time:.1f}ms")
        
        return detections
    

    
    def _process_detections(self, outputs, frame_shape):
        """Simplified detection processing"""
        height, width = frame_shape
        detections = []
        boxes = []
        confidences = []
        class_ids = []
        
        # Process outputs (simplified with early filtering)
        raw_count = 0
        filtered_count = 0
        
        for output in outputs:
            # Handle TensorRT output format
            if output.ndim == 1:
                output = output.reshape(-1, 85)
            elif output.ndim == 3:
                output = output[0]  # Remove batch dimension
            
            for detection in output:
                raw_count += 1
                if len(detection) < 85:
                    continue
                
                # Early exit if we already have enough good detections
                if len(boxes) >= 100:  # Stop processing after 100 candidate boxes
                    break
                    
                # Get confidence scores
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Early exit for very low confidence (performance optimization)
                if confidence <= self.confidence_threshold:
                    filtered_count += 1
                    continue
                
                # Additional filter: only keep higher confidence detections to reduce noise
                min_confidence = max(self.confidence_threshold, 0.4)  # At least 0.4 confidence
                if confidence < min_confidence:
                    filtered_count += 1
                    continue
                
                # Get class name
                if class_id >= len(self.class_names):
                    continue
                class_name = self.class_names[class_id]
                
                # Filter by target objects
                if class_name not in self.target_objects:
                    continue
                
                # Get bounding box (TensorRT outputs pixel coordinates directly)
                center_x = int(detection[0])
                center_y = int(detection[1])
                w = int(detection[2])
                h = int(detection[3])
                
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Filter out tiny boxes (likely noise/artifacts)
                min_box_size = 8  # Minimum 8x8 pixel box
                if w < min_box_size or h < min_box_size:
                    filtered_count += 1
                    continue
                
                # Filter out boxes that are too big (likely false detections)
                if w > 320 or h > 320:  # Max half the frame size
                    filtered_count += 1
                    continue
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
        
        # Apply NMS
        if boxes:
            # Debug NMS input
            if len(boxes) > 100:  # Only print when we have lots of boxes
                print(f"NMS Input: {len(boxes)} boxes, conf_thresh={self.confidence_threshold}, nms_thresh={self.nms_threshold}")
            
            # Use OpenCV vectorized NMS (much faster than custom)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, 0.3)
            
            if len(indices) > 0:
                # Sort by confidence since OpenCV NMS doesn't sort
                sorted_indices = sorted(indices.flatten(), key=lambda i: confidences[i], reverse=True)
                max_detections = 8  # Even lower limit since we have better NMS now
                
                for idx, i in enumerate(sorted_indices):
                    if idx >= max_detections:  # Stop after max detections
                        break
                        
                    x, y, w, h = boxes[i]
                    class_name = self.class_names[class_ids[i]]
                    confidence = confidences[i]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2)
                    })
                    
                    # Only print occasional debug info to reduce spam
                    if len(detections) <= 3:  # Only print first few detections
                        print(f"Final detection: {class_name} ({confidence:.3f}) at ({x},{y},{w},{h})")
        
        # Print performance summary instead of spam
        if raw_count > 0:
            pre_nms = len(boxes)
            print(f"Processed: {raw_count} raw → {raw_count-filtered_count} passed confidence → {pre_nms} pre-NMS → {len(detections)} final")
        
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
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(frame, (x, y - label_size[1] - 10), (x + label_size[0], y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def run(self, show_display=True):
        """Run the vision system"""
        print("Starting vision system...")
        print(f"Target objects: {self.target_objects}")
        print("Press 'q' to quit")
        
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        process_every_n_frames = 8  # Process every 8th frame (faster NMS allows more frequent detection)
        last_detections = []  # Keep last detections for smoother display
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            fps_counter += 1
            
            # Print frame info for debugging
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}: {frame.shape} - Processing: {'YES' if frame_count % process_every_n_frames == 0 else 'NO'}")
            
            # Process detection every N frames
            if frame_count % process_every_n_frames == 0:
                detection_start = time.time()
                detections = self.detect_objects(frame)
                detection_time = time.time() - detection_start
                last_detections = detections  # Store for next frames
                
                print(f"Detection time: {detection_time*1000:.1f}ms, Found: {len(detections)} objects")
            else:
                detections = last_detections  # Use previous detections for smooth display
            
            # Draw results
            frame = self.draw_detections(frame, detections)
            
            # Add status with more info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Shape: {frame.shape}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if detections:
                cv2.putText(frame, f"OBJECTS: {len(detections)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate FPS more frequently for slow framerates
            if fps_counter >= 10:  # Print every 10 frames instead of 30
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
                print(f"FPS: {fps:.1f}")
            
            if show_display:
                cv2.imshow('Simplified Autonomous Vehicle Vision', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Simplified Autonomous Vehicle Vision System')
    parser.add_argument('--model', default='yolov5n_fp16.engine', help='Path to model file')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Run without display')
    args = parser.parse_args()
    
    vision = None
    try:
        vision = SimplifiedVisionSystem(args.model, args.confidence)
        vision.run(show_display=not args.no_display)
        
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