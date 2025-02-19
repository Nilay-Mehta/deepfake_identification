import cv2
import numpy as np
from typing import Generator, Union, Tuple

class MediaProcessor:
    """Handles video and image processing operations"""

    def __init__(self):
        self.cap = None
        self.is_video = True

    def load_media(self, path: str) -> bool:
        """
        Load video or image file
        Returns True if successful, False otherwise
        """
        try:
            # Try to load as video first
            self.cap = cv2.VideoCapture(path)
            if self.cap.isOpened():
                self.is_video = True
                return True

            # If not a video, try to load as image
            self.media = cv2.imread(path)
            if self.media is not None:
                self.is_video = False
                return True

            return False
        except Exception as e:
            print(f"Error loading media: {str(e)}")
            return False

    def get_frames(self) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator that yields frames and progress percentage
        """
        if not self.is_video:
            yield self.media, 100.0
            return

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            current_frame += 1
            progress = (current_frame / total_frames) * 100

            yield frame, progress

        self.cap.release()

    def release(self):
        """Release video capture resources"""
        if self.cap is not None:
            self.cap.release()