import cv2
import os
import numpy as np
from typing import Tuple

class FaceExtractor:
    """Handles extraction and saving of detected faces"""

    def __init__(self, output_dir: str = "extracted_faces"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        self.frame_count = 0

    def extract_face(self, frame: np.ndarray, face_coords: Tuple[float, float, float, float], padding: float) -> np.ndarray:
        """Extract face region from frame using coordinates"""
        x, y, w, h = face_coords

        # Add padding around face 
        x_pad = int(w * padding)
        y_pad = int(h * padding)

        # Calculate padded coordinates
        x1 = max(0, x - x_pad)
        y1 = max(0, y - y_pad)
        x2 = min(frame.shape[1], x + w + x_pad)
        y2 = min(frame.shape[0], y + h + y_pad)

        # Extract and return face region
        return frame[y1:y2, x1:x2]

    def save_face(self, face_img: np.ndarray, is_video: bool = False) -> str:
        """Save extracted face to file"""
        self.frame_count += 1

        if is_video:
            filename = f"face_frame_{self.frame_count}.jpg"
        else:
            filename = f"face_{self.frame_count}.jpg"

        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, face_img)
        print(f"Saved face: {filepath}")
        return filepath
    
    def process_face(self, face_img: np.ndarray) -> np.ndarray:
        face_img = cv2.resize(face_img, (150, 150))  
        face_img = face_img.astype("float32") / 255.0  # Normalize
        face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
        return face_img