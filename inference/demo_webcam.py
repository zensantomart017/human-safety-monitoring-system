import utils.torch_patch  # Workaround for PyTorch 2.6 weights_only=True default
import cv2
import argparse
from inference.yolo_pipeline import SafetyPipeline
from visualization.annotator import Annotator

def run_webcam_demo(camera_id=0):
    print(f"Starting webcam demo on camera ID: {camera_id}")
    print("Press 'q' to quit.")
    
    pipeline = SafetyPipeline()
    annotator = Annotator()
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {camera_id}")
        return
        
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
        
        # 4. Show
        cv2.imshow("Human Safety Monitoring System - Realtime", annotated_frame)
        
        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam demo terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Webcam Demo for Safety Monitoring System")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID to use (default 0)")
    
    args = parser.parse_args()
    run_webcam_demo(args.camera)
