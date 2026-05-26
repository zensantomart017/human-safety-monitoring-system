from ultralytics import YOLO
import time

def evaluate_model(model_path: str, data_config: str):
    """
    Evaluates a YOLOv8 model using standard metrics (mAP, Precision, Recall).
    """
    model = YOLO(model_path)
    print(f"Evaluating {model_path} on {data_config}...")
    metrics = model.val(data=data_config)
    
    print("\n--- Evaluation Results ---")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"Precision: {metrics.box.mp}")
    print(f"Recall: {metrics.box.mr}")
    return metrics

def calculate_fps(pipeline, source_video, num_frames=100):
    """
    Calculates the FPS of the inference pipeline.
    """
    import cv2
    cap = cv2.VideoCapture(source_video)
    
    start_time = time.time()
    frames_processed = 0
    
    while cap.isOpened() and frames_processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run detection
        _ = pipeline.detect_person(frame)
        frames_processed += 1
        
    end_time = time.time()
    cap.release()
    
    elapsed = end_time - start_time
    fps = frames_processed / elapsed if elapsed > 0 else 0
    
    print(f"Processed {frames_processed} frames in {elapsed:.2f} seconds.")
    print(f"Estimated FPS: {fps:.2f}")
    return fps

if __name__ == "__main__":
    # Example usage
    # evaluate_model("models/yolov8s.pt", "coco128.yaml")
    pass
