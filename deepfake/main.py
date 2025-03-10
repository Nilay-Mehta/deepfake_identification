import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from media_processor import MediaProcessor
from face_detector import FaceDetector
from visualizer import Visualizer
from face_extractor import FaceExtractor
from live_tracking import LiveTracking  # Added live tracking import

model_path = "C:\\Users\\nilay\\OneDrive\\Desktop\\New folder (3)\\facial_recognition-main\\deepfake\\last_model.h5"   

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
        media_proc = MediaProcessor()
        face_detector = FaceDetector(min_detection_confidence=0.5)
        visualizer = Visualizer()
        face_extractor = FaceExtractor()
        model = load_model(model_path)

        print(f"\nProcessing {file_path}")
        print(f"Extracted faces will be saved in: {face_extractor.output_dir}")
        print("Press 'q' to quit")

        if not media_proc.load_media(file_path):
            print(f"Failed to load media file: {file_path}")
            return

        for frame, progress in media_proc.get_frames():
            try:
                faces = face_detector.detect_faces(frame)
                if faces:
                    print(f"\rDetected {len(faces)} faces in current frame", end="")

                face_predictions = []
                for face_coords in faces:
                    face_img = face_extractor.extract_face(frame, face_coords, padding=0.4)
                    face_img = face_extractor.process_face(face_img)
                    face_predictions.append(model.predict(face_img, verbose=0)[0][0])
                    
                output_frame = visualizer.draw_faces(frame, faces, face_predictions, padding=0.0, conf=0.5)
                output_frame = visualizer.draw_progress(output_frame, progress)

                cv2.imshow('Deepfake Detection', output_frame)
                out.write(output_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nExiting...")
                    break

            except Exception as e:
                print(f"\nError processing frame: {str(e)}")
                break

    except Exception as e:
        print(f"Error processing media: {str(e)}")

    finally:
        media_proc.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        print("Select an option:")
        print("1. Process media file")
        print("2. Start live tracking")
        print("3. Exit")
        choice = input("Enter choice: ").strip()

        if choice == "1":
            media_path = get_media_path()
            output_video_path = f"{media_path.split('\\')[-1].split('.')[0]}_output.mp4"
            cap = cv2.VideoCapture(media_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
            process_media(media_path)
        
        elif choice == "2":
            tracker = LiveTracking()
            tracker.start_tracking()

        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Try again.")
            continue

        retry = input("\nProcess another file or restart live tracking? (y/n): ").lower().strip()
        if retry != 'y':
            print("Goodbye!")
            break
