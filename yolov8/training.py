from ultralytics import YOLO
import cv2


def load_model():
    model = YOLO("yolov8n.pt")
    return model 