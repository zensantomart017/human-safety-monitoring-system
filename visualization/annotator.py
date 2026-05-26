import cv2
import numpy as np

class Annotator:
    def __init__(self):
        # Define some colors for different classes
        self.colors = {
            "person": (255, 0, 0),     # Blue
            "helmet": (0, 255, 0),     # Green
            "vest": (0, 165, 255),     # Orange
            "glasses": (255, 255, 0),  # Cyan
            "default": (255, 255, 255) # White
        }

    def draw_detections(self, image: np.ndarray, results: list) -> np.ndarray:
        """
        Draws bounding boxes, labels, and tracking IDs on the image.
        """
        annotated_image = image.copy()

        for result in results:
            boxes = result.boxes
            names = result.names
            
            for i, box in enumerate(boxes):
                # Bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Class name and confidence
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label_name = names[cls_id]
                
                # Tracking ID if available
                track_id = int(box.id[0]) if box.id is not None else None
                
                # Select color
                color = self.colors.get(label_name, self.colors["default"])
                
                # Draw Box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                
                # Prepare Label Text
                label_text = f"{label_name} {conf:.2f}"
                if track_id is not None:
                    label_text = f"ID:{track_id} " + label_text
                    
                # Draw Label Background
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
                
                # Draw Label Text
                cv2.putText(annotated_image, label_text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                            
        return annotated_image
