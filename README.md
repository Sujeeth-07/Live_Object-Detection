# Child Safety Object Detection

Overview:

This project uses Flask to implement a real-time object detection system for child safety. It captures video from the client's camera, processes the frames using a trained machine learning model, and then detects objects in the video stream (such as people, toys, or other potential safety risks). The processed video is streamed back to the client with the detected objects highlighted.

# Technologies Used

Flask: Web framework to serve the application.
HTML-CSS-JS: Serves UI and Frontend


# Installation:

pip install -r requirements.txt

requirements.txt

-flask

-opencv-python

-numpy

-Pillow

-gunicorn

# Dataset

The YOLOV3 weights and configuration files are required for this project.You can download the yoloV3 dataset from kaggle (https://www.kaggle.com/datasets/shivam316/yolov3-weights).

#Define Paths:

You need to define the path to your class names file,config file,weights file within yoour python scripts.
For Ex:
 # Define your YOLO model paths
    config_path = r"C:\Users\SUJITH\OneDrive\Desktop\Live_Object_Detection_Children\yolov3.cfg"
    weights_path = r"C:\Users\SUJITH\OneDrive\Desktop\Live_Object_Detection_Children\yolov3.weights"
    classes_path = r"C:\Users\SUJITH\OneDrive\Desktop\Live_Object_Detection_Children\coco.names"

