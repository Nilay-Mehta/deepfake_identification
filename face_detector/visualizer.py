import cv2
import numpy as np
from typing import List, Tuple

class Visualizer:
    """Handles visualization of detected faces and progress"""
    
    @staticmethod
    def draw_faces(image: np.ndarray, faces: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """Draw bounding boxes around detected faces"""
        output = image.copy()
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add confidence label
            cv2.putText(output, 'Face', (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                       
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
