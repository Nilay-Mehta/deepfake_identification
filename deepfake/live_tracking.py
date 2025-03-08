import cv2
import numpy as np
from face_detector import FaceDetector
from visualizer import Visualizer

class LiveTracking:
    """Handles live face tracking using webcam"""
    
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        self.face_detector = FaceDetector(min_detection_confidence=0.5)
        self.visualizer = Visualizer()
    
    def start_tracking(self):
        """Start live face tracking"""
        if not self.cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to exit")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            faces = self.face_detector.detect_faces(frame)
            output_frame = self.visualizer.draw_faces(frame, faces, [1.0] * len(faces), padding=0.0, conf=0.5)
            
            cv2.imshow("Live Face Tracking", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = LiveTracking()
    tracker.start_tracking()
