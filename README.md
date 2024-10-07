# Helmet and Safety Gear Detection
This project aims to detect various safety-related objects such as helmets, vests, heads, and people using YOLOv8. The model is trained to identify these objects, which makes it suitable for applications in environments like construction sites or schools, where safety is a priority.

## Table of Contents
- Project Overview
- Dataset
- Model Training
- Installation
- How to Use
- Results
- Future Improvements

## Project Overview
The main objective of this project is to create an object detection model to identify the following four objects:

- Helmet
- Vest
- Head
- Person

This model can be applied to monitor safety compliance in various environments such as:
- Construction Sites: Ensuring workers wear helmets and vests.
- Schools: Monitoring students and ensuring safe environments.

## Dataset 
The dataset consists of:
- Training images: 17,248 images
- Validation images: 2,438 images

Each image is annotated with bounding boxes for the four target objects: helmet, vest, head, and person.

## Model Training 
The model is trained using YOLO11s. Below are the training details:

- Number of Epochs: 5 (initial training) + additional epochs based on requirement
- Image Size: 640x640
- GPU Used: T4 GPU (on Google Colab)
Losses: Box loss, Class loss, and DFL loss were monitored during training.

## Training Command
```
!yolo train model=/content/ultralytics/runs/detect/train/weights/best.pt \
data=/content/ultralytics/Helmet_detect_dataset/data.yaml \
epochs=5 imgsz=640
```

## Installation
1. Clone the repository:
```
git clone https://github.com/your-username/helmet-safety-detection.git
cd helmet-safety-detection
```
2. Install dependencies:
```
pip install ultralytics
```


## How to use 
### Predict with an Image
To perform object detection on an image:
```
!yolo predict model=/path/to/best.pt source='https://your-image-url.com'
```
### Predict with a Video: 
You can also perform predictions on a video file or stream:
```
!yolo predict model=/path/to/best.pt source='/path/to/video.mp4'
```
### Predict with Camera (Webcam) (Only on local !!!):
```
!yolo predict model=/path/to/best.pt source=0
```

## Classes and Labels 

### The detected objects will be labeled as:

- helmet
- vest
- head
- person


## Results
The model has shown promising results, achieving high accuracy in detecting the target objects in real-time on video and image streams.

Example Detection:
- F1-score, Precision, and Recall graphs are available in the All_graphs/.
- Sample images showing detection results are stored in predict/.

## Future Improvements

- Model fine-tuning: Continue training with more epochs for better accuracy.
- Multi-class detection: Expand the model to detect more objects related to safety in different environments.
- Real-time deployment: Integrate the model into a real-time surveillance system for construction sites and schools.
