import utils.torch_patch  # Workaround for PyTorch 2.6 weights_only=True default
import cv2
import argparse
import os
from inference.yolo_pipeline import SafetyPipeline
from visualization.annotator import Annotator

def run_video_demo(input_video_path, output_video_path):
    print(f"Starting video demo on: {input_video_path}")
    
    pipeline = SafetyPipeline()
    annotator = Annotator()
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}")
        return
        
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Track persons
        person_results = pipeline.track_persons(frame)
        
        # 2. Detect PPE
        ppe_results = pipeline.detect_ppe(frame)
        
        # 3. Annotate
        annotated_frame = annotator.draw_detections(frame, person_results)
        annotated_frame = annotator.draw_detections(annotated_frame, ppe_results)
        
        # 4. Write
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
    cap.release()
    out.release()
    print(f"Demo complete. Output saved to {output_video_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Video Demo for Safety Monitoring System")
    parser.add_argument("--input", type=str, default="videos/test.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/demo_output.mp4", help="Path to output video")
    
    args = parser.parse_args()
    run_video_demo(args.input, args.output)
