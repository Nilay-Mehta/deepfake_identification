# **Deepfake Identifier**

## **Overview**
The Deepfake Identifier is a Python-based system that detects faces in images and videos, extracts them, and determines whether they are real or fake using a deep learning model.

---

## **Features**
- 📸 **Detects faces** in images and videos  
- ✂️ **Extracts face regions** for analysis  
- 🤖 **Uses deep learning** to classify faces as real or fake  
- 🖍️ **Displays results visually** with bounding boxes  
- ⚙️ **Supports real-time and live camera processing** to give user real time experience

---

## **Installation**
### **Prerequisites**
Ensure you have Python 3 installed along with the following dependencies:
```bash
pip install numpy opencv-python tensorflow mediapipe
```

## **Usage**
- Run the main program:
```bash
python main.py
```
- Enter the path to an image or video file when prompted.
- The system will process the file and display the results.
- Press q to exit the program.
- Choose whether to analyze another file.

---

## **Project Structure**
```base
.
├── face_detector.py       # Detects faces using MediaPipe
├── face_extractor.py      # Extracts and processes face regions
├── live_detector.py       # Handles real-time face detection from webcam
├── main.py                # Manages the entire workflow
├── media_processor.py     # Loads and processes images/videos
├── model.h5               # Pre-trained deep learning model
├── visualizer.py          # Draws results on images/videos
```

---

## **How It Works**
- The system loads an image or video.
- It detects faces using MediaPipe.
- Each detected face is extracted and resized.
- The face is passed to a deep learning model for classification.
- The system labels faces as real or fake and displays results.

---

## **Contibuting**
Feel free to contribute by improving detection accuracy, adding more features, or optimizing the code!

---

