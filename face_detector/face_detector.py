import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple

class FaceDetector:
    """Handles face detection using MediaPipe"""

    def __init__(self, min_detection_confidence: float = 0.5):
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                min_detection_confidence=min_detection_confidence
            )
        except Exception as e:
            print(f"Error initializing MediaPipe: {str(e)}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """
        Detect faces in the image and return their bounding boxes
        Returns list of tuples (x, y, width, height)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect faces
        results = self.face_detection.process(image_rgb)

        faces = []
        if results.detections:
            image_height, image_width, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * image_width)
                y = int(bbox.ymin * image_height)
                w = int(bbox.width * image_width)
                h = int(bbox.height * image_height)
                faces.append((x, y, w, h))

        return faces

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()