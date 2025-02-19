import os
import cv2
import numpy as np
from media_processor import MediaProcessor
from face_detector import FaceDetector
from visualizer import Visualizer
from face_extractor import FaceExtractor

def get_media_path() -> str:
    """Get media path from user input"""
    while True:
        path = input("Enter path to media file: ").strip()
        if os.path.exists(path):
            return path
        print(f"Error: File not found: {path}")

def process_media(file_path: str):
    """Main function to process media files and detect faces"""
    try:
        # Initialize components
        media_proc = MediaProcessor()
        face_detector = FaceDetector(min_detection_confidence=0.5)
        visualizer = Visualizer()
        face_extractor = FaceExtractor()

        print(f"\nProcessing {file_path}")
        print(f"Extracted faces will be saved in: {face_extractor.output_dir}")
        print("Press 'q' to quit")

        # Load media file
        if not media_proc.load_media(file_path):
            print(f"Failed to load media file: {file_path}")
            return

        # Process frames
        for frame, progress in media_proc.get_frames():
            try:
                # Detect faces
                faces = face_detector.detect_faces(frame)
                if faces:
                    print(f"\rDetected {len(faces)} faces in current frame", end="")

                # Extract and save face regions
                for face_coords in faces:
                    face_img = face_extractor.extract_face(frame, face_coords)
                    face_extractor.save_face(face_img, media_proc.is_video)

                # Draw faces and progress
                output_frame = visualizer.draw_faces(frame, faces)
                output_frame = visualizer.draw_progress(output_frame, progress)

                # Display result
                cv2.imshow('Deepfake Detection', output_frame)

                # Break if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nExiting...")
                    break

            except Exception as e:
                print(f"\nError processing frame: {str(e)}")
                break

    except Exception as e:
        print(f"Error processing media: {str(e)}")

    finally:
        # Cleanup
        media_proc.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        media_path = get_media_path()
        process_media(media_path)

        retry = input("\nProcess another file? (y/n): ").lower().strip()
        if retry != 'y':
            print("Goodbye!")
            break