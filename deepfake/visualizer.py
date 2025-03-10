import cv2
import numpy as np
from typing import List, Tuple

class Visualizer:
    """Handles visualization of detected faces and progress"""
    
    @staticmethod
    def draw_faces(image: np.ndarray, faces: List[Tuple[float, float, float, float]], face_predictions=None, padding: float = 0.0, conf: float = 0.5) -> np.ndarray:
        """Draw bounding boxes around detected faces (hide text if no predictions are provided)"""
        output = image.copy()
    
        # If no predictions are given, disable text labels
        hide_text = face_predictions is None or len(face_predictions) == 0
    
        if hide_text:
            face_predictions = [None] * len(faces)  # Create dummy list to match faces

        for (x, y, w, h), prediction in zip(faces, face_predictions):
            x_pad = int(w * padding)
            y_pad = int(h * padding)

            x1 = max(0, x - x_pad)
            y1 = max(0, y - y_pad)
            x2 = min(output.shape[1], x + w + x_pad)
            y2 = min(output.shape[0], y + h + y_pad)
        
            # Draw bounding box around face
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Skip text if hide_text is True
            if not hide_text:
                cv2.putText(output, f"{prediction:.2f}", (x2 - 15, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                prediction_label = "Real" if prediction > conf else "Fake"
                cv2.putText(output, prediction_label, (x1, max(10, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return output

        
    @staticmethod
    def draw_progress(image: np.ndarray, progress: float) -> np.ndarray:
        """Draw progress bar on the image"""
        height, width = image.shape[:2]
        
        # Draw progress bar background
        cv2.rectangle(image, (10, height - 30), (width - 10, height - 20),
                     (0, 0, 0), cv2.FILLED)
                     
        # Draw progress
        progress_width = int((width - 20) * (progress / 100))
        cv2.rectangle(image, (10, height - 30), (10 + progress_width, height - 20),
                     (0, 255, 0), cv2.FILLED)
                     
        # Draw progress text
        cv2.putText(image, f'{progress:.1f}%', (10, height - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
        return image
