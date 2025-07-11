#!/usr/bin/env python3
"""
Simple model downloader for YOLOv5 models - No TensorRT required
Use this if TensorRT conversion fails
"""

import urllib.request
import os
import argparse

def download_model(model_name):
    """Download YOLOv5 model from official releases"""
    models = {
        'yolov5n': {
            'url': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.onnx',
            'filename': 'yolov5n.onnx',
            'description': 'Nano - Fastest, good for Jetson Nano'
        },
        'yolov5s': {
            'url': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx',
            'filename': 'yolov5s.onnx', 
            'description': 'Small - Current model'
        },
        'yolov5m': {
            'url': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.onnx',
            'filename': 'yolov5m.onnx',
            'description': 'Medium - Higher accuracy, slower'
        }
    }
    
    if model_name not in models:
        print(f"Model '{model_name}' not available.")
        print("Available models:", list(models.keys()))
        return None
    
    model_info = models[model_name]
    filename = model_info['filename']
    
    if os.path.exists(filename):
        print(f"{filename} already exists!")
        overwrite = input("Overwrite? (y/n): ")
        if overwrite.lower() != 'y':
            return filename
    
    print(f"Downloading {model_info['description']}...")
    print(f"URL: {model_info['url']}")
    
    try:
        urllib.request.urlretrieve(model_info['url'], filename)
        print(f"Downloaded: {filename}")
        
        # Check file size
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")
        
        return filename
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Download YOLOv5 models')
    parser.add_argument('--model', choices=['yolov5n', 'yolov5s', 'yolov5m'], 
                       default='yolov5n', help='Model to download')
    parser.add_argument('--list', action='store_true', help='List available models')
    args = parser.parse_args()
    
    if args.list:
        print("Available YOLOv5 models:")
        print("  yolov5n - Nano (14MB) - Fastest, recommended for Jetson Nano")
        print("  yolov5s - Small (28MB) - Balanced speed/accuracy")
        print("  yolov5m - Medium (84MB) - Higher accuracy, slower")
        print("\nRecommended for Jetson Nano: yolov5n")
        return
    
    print(f"Downloading {args.model}...")
    downloaded_file = download_model(args.model)
    
    if downloaded_file:
        print(f"\nSuccess! Use with:")
        print(f"   python3 vision.py --model {downloaded_file}")
        print(f"\nFor best performance on Jetson Nano:")
        print(f"   python3 vision.py --model {downloaded_file} --confidence 0.6")
    else:
        print("\nDownload failed!")

if __name__ == "__main__":
    main() 