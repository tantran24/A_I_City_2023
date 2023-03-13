import cv2

import sys
sys.path.append(".")
from yolov8.detect import *

def crop_img(img, name_image, x1, y1, x2, y2):
    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite("data/input_for_convnext/"+name_image+".jpg", crop_img)

def create_input_convnext(video_path):
    results = detector(video_path)
    boxes = results[0].boxes.numpy()

    cap = cv2.VideoCapture(video_path)    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in range(0, total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, image = cap.read()
        x1 = int(boxes[frame].xyxy[0][0])
        y1 = int(boxes[frame].xyxy[0][1])
        x2 = int(boxes[frame].xyxy[0][2])
        y2 = int(boxes[frame].xyxy[0][3])
        crop_img(image, "s"+str(frame), x1, y1, x2, y2)
        
        
if __name__ == "__main__":
    create_input_convnext("data/video/stock-footage-street-camera-view-of-people-walking-in-pedestrian-zone-high-angle-view.mp4")
