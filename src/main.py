#!/usr/bin/env python3
"""
Main entry point for the Autonomous Vehicle Vision System
"""

import argparse
from vision.detection import SimplifiedVisionSystem
from utils.config import DEFAULT_MODEL_PATH, CONFIDENCE_THRESHOLD

def main():
    parser = argparse.ArgumentParser(description='Autonomous Vehicle Vision System')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, 
                       help='Path to TensorRT model engine')
    parser.add_argument('--confidence', type=float, default=CONFIDENCE_THRESHOLD, 
                       help='Confidence threshold for detections')
    # parser.add_argument('--no-display', action='store_true', 
    #                    help='Run without display (headless mode)')
    args = parser.parse_args()
    
    try:
        # Initialize vision system
        vision = SimplifiedVisionSystem(args.model, args.confidence)
        
        # Run vision system
        vision.run(show_display=True)  # Always show display
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'vision' in locals():
            vision.cleanup()

if __name__ == "__main__":
    main() 