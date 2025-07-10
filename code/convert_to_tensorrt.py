#!/usr/bin/env python3
"""
Convert YOLOv5 ONNX model to TensorRT engine for optimized inference on Jetson Nano
"""

import tensorrt as trt
import argparse
import os

def check_tensorrt_version():
    """Check TensorRT version and compatibility"""
    try:
        version = trt.__version__
        print(f"TensorRT version: {version}")
        
        # Parse version
        major, minor = map(int, version.split('.')[:2])
        
        if major < 7:
            print("WARNING: TensorRT version is very old. Consider upgrading.")
            return False
        elif major == 7:
            print("Using TensorRT 7.x compatibility mode")
        elif major == 8 and minor < 4:
            print("Using TensorRT 8.0-8.3 compatibility mode")
        else:
            print("Using latest TensorRT API")
        
        return True
    except Exception as e:
        print(f"ERROR checking TensorRT version: {e}")
        return False

def build_engine(onnx_path, engine_path, precision='fp16', max_batch_size=1):
    """
    Build TensorRT engine from ONNX model
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        precision: 'fp16' or 'int8' for quantization
        max_batch_size: Maximum batch size
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    print(f"Loading ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print("Building TensorRT engine...")
    config = builder.create_builder_config()
    
    # Set workspace size (compatible with older TensorRT versions)
    try:
        # TensorRT 8.4+ method
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)  # 256MB
    except AttributeError:
        # TensorRT 8.0-8.3 method (common on Jetson Nano)
        config.max_workspace_size = 1 << 28  # 256MB
    
    # Set precision
    if precision == 'fp16':
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    elif precision == 'int8':
        config.set_flag(trt.BuilderFlag.INT8)
        print("Using INT8 precision")
        # Note: INT8 requires calibration dataset - implement if needed
    
    # Enable optimization profiles for dynamic shapes
    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape
    
    # Set input shape profiles
    profile.set_shape(input_tensor.name, 
                     (1, input_shape[1], input_shape[2], input_shape[3]),  # min
                     (1, input_shape[1], input_shape[2], input_shape[3]),  # opt
                     (max_batch_size, input_shape[1], input_shape[2], input_shape[3]))  # max
    config.add_optimization_profile(profile)
    
    # Build engine (compatible with different TensorRT versions)
    try:
        # TensorRT 8.0+ method
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("ERROR: Failed to build TensorRT engine")
            return False
        engine_data = serialized_engine
    except AttributeError:
        # TensorRT 7.x method
        engine = builder.build_engine(network, config)
        if engine is None:
            print("ERROR: Failed to build TensorRT engine")
            return False
        engine_data = engine.serialize()
        del engine
    
    print(f"Saving TensorRT engine: {engine_path}")
    with open(engine_path, 'wb') as f:
        f.write(engine_data)
    
    print("TensorRT engine created successfully!")
    return True

def download_optimized_model():
    """Download a smaller YOLOv5 model optimized for Jetson Nano"""
    import urllib.request
    
    # YOLOv5n (nano) is much faster than YOLOv5s
    models = {
        'yolov5n.onnx': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx',
        'yolov5s.onnx': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx'
    }
    
    print("Available optimized models for Jetson Nano:")
    print("1. YOLOv5n (nano) - Fastest, lower accuracy")
    print("2. YOLOv5s (small) - Current model")
    
    choice = input("Download YOLOv5n for better performance? (y/n): ")
    
    if choice.lower() == 'y':
        model_url = models['yolov5n.onnx']
        model_path = 'yolov5n.onnx'
        
        print(f"Downloading {model_path}...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Downloaded {model_path}")
        return model_path
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Convert YOLOv5 to TensorRT')
    parser.add_argument('--onnx', default='yolov5s.onnx', help='Input ONNX model path')
    parser.add_argument('--engine', help='Output TensorRT engine path')
    parser.add_argument('--precision', choices=['fp16', 'int8'], default='fp16', 
                       help='Precision mode (fp16 recommended for Jetson Nano)')
    parser.add_argument('--download', action='store_true', help='Download optimized model')
    args = parser.parse_args()
    
    # Check TensorRT compatibility
    if not check_tensorrt_version():
        print("TensorRT compatibility check failed!")
        return
    
    # Download optimized model if requested
    if args.download:
        downloaded_model = download_optimized_model()
        if downloaded_model:
            args.onnx = downloaded_model
    
    # Set default engine path
    if not args.engine:
        base_name = os.path.splitext(args.onnx)[0]
        args.engine = f"{base_name}_{args.precision}.engine"
    
    # Check if ONNX file exists
    if not os.path.exists(args.onnx):
        print(f"ERROR: ONNX model not found: {args.onnx}")
        print("Run with --download to get an optimized model")
        return
    
    # Convert to TensorRT
    success = build_engine(args.onnx, args.engine, args.precision)
    
    if success:
        print(f"\nConversion complete!")
        print(f"Use the TensorRT engine with: --model {args.engine} --tensorrt")
        print(f"Expected performance improvement: 3-5x faster inference")
    else:
        print("Conversion failed!")

if __name__ == "__main__":
    main() 