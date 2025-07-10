#!/usr/bin/env python3
"""
Autonomous Vehicle Vision System - Optimized for Jetson Nano
Using TensorRT for high-performance object detection
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
import threading
from collections import deque

# Try to import TensorRT and PyCUDA - fallback to OpenCV DNN if not available
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT not available, will use OpenCV DNN")

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
    print("PyCUDA not available, will use OpenCV DNN for TensorRT engines")

class TensorRTInference:
    def __init__(self, engine_path):
        """Initialize TensorRT inference engine"""
        if not TENSORRT_AVAILABLE or not PYCUDA_AVAILABLE:
            raise ImportError("TensorRT and PyCUDA are required for TensorRTInference")
        
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

class OptimizedVisionSystem:
    def __init__(self, model_path="yolov5s.onnx", confidence_threshold=0.5, use_tensorrt=True):
        """
        Initialize optimized vision system
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = 0.4
        self.use_tensorrt = use_tensorrt
        
        # Reduce input size for better performance
        self.input_size = (640, 640)  # Reduced from 640x640
        
        # Frame processing settings
        self.process_every_n_frames = 3  # Process every 3rd frame
        self.frame_count = 0
        
        # Threading
        self.frame_queue = deque(maxlen=2)
        self.detection_results = None
        self.processing_lock = threading.Lock()
        
        # Load model
        if use_tensorrt and model_path.endswith('.engine') and TENSORRT_AVAILABLE and PYCUDA_AVAILABLE:
            try:
                print("Loading TensorRT engine with PyCUDA...")
                self.inference_engine = TensorRTInference(model_path)
                self.use_opencv_dnn = False
                print("TensorRT engine loaded successfully!")
            except Exception as e:
                print(f"TensorRT engine loading failed: {e}")
                print("Falling back to OpenCV DNN...")
                self.net = cv2.dnn.readNet(model_path)
                self._setup_opencv_backend()
                self.use_opencv_dnn = True
        else:
            if model_path.endswith('.engine'):
                print("Loading TensorRT engine with OpenCV DNN (no PyCUDA)...")
                print("Note: This will still be faster than ONNX!")
            else:
                print("Loading OpenCV DNN model...")
            
            try:
                self.net = cv2.dnn.readNet(model_path)
                self._setup_opencv_backend()
                self.use_opencv_dnn = True
                print("Model loaded successfully with OpenCV DNN!")
            except Exception as e:
                print(f"Model loading failed: {e}")
                raise
        
        # COCO class names (reduced list for better performance)
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
        
        # Focus on important objects for autonomous driving (+ bottle for testing)
        self.target_objects = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign', 'bottle']
        
        # Initialize camera
        self.cap = None
        self.initialize_camera()
        
    def _setup_opencv_backend(self):
        """Setup OpenCV DNN backend optimally for Jetson"""
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("Using CUDA backend with FP16")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        else:
            print("Using CPU backend")
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
    def initialize_camera(self, camera_id=0):
        """Initialize camera with optimized pipeline"""
        print(f"Initializing camera {camera_id}...")
        
        # Optimized GStreamer pipeline for better performance
        gst_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=(int)416, height=(int)416, format=(string)BGRx ! "  # Smaller resolution
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"  # Drop frames if processing is slow
        )
        
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not self.cap.isOpened():
            print("CSI camera failed, trying USB camera...")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception(f"Could not open any camera")
            
            # Set smaller resolution for USB camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size
        
        print("Camera initialized successfully")
        
    def preprocess_frame_gpu(self, frame):
        """GPU-accelerated preprocessing when possible"""
        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Resize on GPU
        gpu_resized = cv2.cuda.resize(gpu_frame, self.input_size)
        
        # Download back to CPU
        resized_frame = gpu_resized.download()
        
        # Create blob
        blob = cv2.dnn.blobFromImage(
            resized_frame,
            scalefactor=1/255.0,
            size=self.input_size,
            swapRB=True,
            crop=False
        )
        
        return blob
    
    def detect_objects_optimized(self, frame):
        """
        Optimized object detection with reduced processing
        """
        try:
            if self.use_opencv_dnn:
                # Use GPU preprocessing if available
                try:
                    blob = self.preprocess_frame_gpu(frame)
                except:
                    # Fallback to CPU preprocessing
                    blob = cv2.dnn.blobFromImage(
                        frame,
                        scalefactor=1/255.0,
                        size=self.input_size,
                        swapRB=True,
                        crop=False
                    )
                
                self.net.setInput(blob)
                outputs = self.net.forward()
            else:
                # TensorRT inference
                blob = cv2.dnn.blobFromImage(
                    frame,
                    scalefactor=1/255.0,
                    size=self.input_size,
                    swapRB=True,
                    crop=False
                )
                outputs = self.inference_engine.infer(blob)
            
            return self._process_detections(outputs, frame.shape[:2])
        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _process_detections(self, outputs, original_shape):
        """Optimized detection processing"""
        height, width = original_shape
        detections = []
        boxes = []
        confidences = []
        class_ids = []
        
        try:
            # Process only high-confidence detections early
            for i, output in enumerate(outputs):
                # Handle different output formats
                if hasattr(output, 'shape') and len(output.shape) == 3:
                    # Standard format: [batch, detections, 85]
                    output = output[0]  # Remove batch dimension
                elif hasattr(output, 'shape') and len(output.shape) == 1:
                    # Flattened format - reshape if needed
                    if len(output) % 85 == 0:
                        output = output.reshape(-1, 85)
                    else:
                        continue
                
                for detection in output:
                    if hasattr(detection, '__len__') and len(detection) >= 6:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                    else:
                        continue
                
                    # Early exit for low confidence
                    if confidence <= self.confidence_threshold:
                        continue
                    
                    # Only process important classes for autonomous driving
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else "unknown"
                    if class_name not in self.target_objects:
                        continue
                    
                    # Get bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
            
            # Apply NMS only if we have detections
            if boxes:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
                
                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        class_name = self.class_names[class_ids[i]]
                        confidence = confidences[i]
                        
                        detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': (x, y, w, h),
                            'center': (x + w//2, y + h//2)
                        })
            
            return detections
            
        except Exception as e:
            print(f"Error processing detections: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def async_detection_worker(self):
        """Asynchronous detection processing in separate thread"""
        while True:
            if len(self.frame_queue) > 0:
                with self.processing_lock:
                    frame = self.frame_queue.popleft()
                
                # Process detection
                detections = self.detect_objects_optimized(frame)
                
                with self.processing_lock:
                    self.detection_results = detections
            else:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
    
    def simplified_obstacle_detection(self, frame):
        """Simplified obstacle detection for better performance"""
        # Use smaller image for edge detection
        small_frame = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Simple edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Check center region for obstacles
        h, w = edges.shape
        center_region = edges[h//3:2*h//3, w//3:2*w//3]
        edge_density = np.sum(center_region > 0) / (center_region.shape[0] * center_region.shape[1])
        
        return 'stop' if edge_density > 0.3 else 'forward'
    
    def run_optimized_vision(self, show_display=True):
        """
        Run optimized vision system
        """
        print("Starting optimized vision system...")
        print(f"Target objects: {self.target_objects}")
        print("Press 'q' to quit")
        
        # Disable threading for TensorRT to avoid CUDA context issues
        use_threading = self.use_opencv_dnn
        
        if use_threading:
            print("Using threaded processing (OpenCV DNN)")
            # Start async detection thread
            detection_thread = threading.Thread(target=self.async_detection_worker, daemon=True)
            detection_thread.start()
        else:
            print("Using main thread processing (TensorRT)")
        
        frame_count = 0
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            frame_count += 1
            fps_counter += 1
            
            if use_threading:
                # Threading mode (OpenCV DNN)
                # Only process every N frames for detection
                if frame_count % self.process_every_n_frames == 0:
                    if len(self.frame_queue) < self.frame_queue.maxlen:
                        self.frame_queue.append(frame.copy())
                
                # Get latest detection results
                with self.processing_lock:
                    current_detections = self.detection_results if self.detection_results else []
            else:
                # Direct processing mode (TensorRT)
                if frame_count % self.process_every_n_frames == 0:
                    detection_start = time.time()
                    current_detections = self.detect_objects_optimized(frame)
                    detection_time = time.time() - detection_start
                    if frame_count % 30 == 0:  # Print timing every 30 frames
                        print(f"Detection time: {detection_time*1000:.1f}ms, Detections: {len(current_detections)}")
                else:
                    current_detections = []
            
            # Simple obstacle detection on every frame (fast)
            nav_hint = self.simplified_obstacle_detection(frame)
            
            # Draw results
            frame = self.draw_detections(frame, current_detections)
            
            # Add status text
            status_text = f"Frame: {frame_count}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Check for targets
            target_found = any(det['class'] in self.target_objects for det in current_detections)
            if target_found:
                cv2.putText(frame, "TARGET DETECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            nav_text = f"Nav: {nav_hint}"
            cv2.putText(frame, nav_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Calculate and display FPS
            if fps_counter >= 30:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
                print(f"FPS: {fps:.1f}, Detections: {len(current_detections)}")
            
            if show_display:
                cv2.imshow('Optimized Autonomous Vehicle Vision', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
    
    def draw_detections(self, frame, detections):
        """Optimized detection drawing"""
        for detection in detections:
            x, y, w, h = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Use simpler drawing for better performance
            color = (0, 255, 0)  # Green for all detections
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Simpler label
            label = f"{class_name}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Optimized Autonomous Vehicle Vision System')
    parser.add_argument('--model', default='yolov5s.onnx', help='Path to ONNX/TensorRT model')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Run without display (headless)')
    parser.add_argument('--tensorrt', action='store_true', help='Use TensorRT engine')
    args = parser.parse_args()
    
    try:
        # Initialize optimized vision system
        vision = OptimizedVisionSystem(args.model, args.confidence, args.tensorrt)
        
        # Run optimized vision
        vision.run_optimized_vision(show_display=not args.no_display)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'vision' in locals():
            vision.cleanup()

if __name__ == "__main__":
    main()