import cv2

import sys
sys.path.append(".")
from yolov8.detect import *

def crop_img(img, name_image, x1, y1, x2, y2):
    crop_img = img[y1:y2, x1:x2]
    cv2.imwrite("data/input_for_convnext/"+name_image+".jpg", crop_img)

def create_input_convnext(video_path):
    results = detector(video_path)
    cap = cv2.VideoCapture(video_path)    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in range(0, total_frames):
        boxes = results[frame].boxes.numpy()
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        _, image = cap.read()
        S_max_box = 0
        x_1, y_1, x_2, y_2 = 0, 0, 0, 0

        for box in boxes.xyxy:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            S_box = (x2-x1)*(y2-y1)

            if S_max_box < S_box:
                S_max_box = S_box
                x_1, y_1, x_2, y_2 = x1, y1, x2, y2

        crop_img(image, "frame_" + str(frame), x_1, y_1, x_2, y_2)
        
        
if __name__ == "__main__":
    create_input_convnext("data/video/stock-footage-street-camera-view-of-people-walking-in-pedestrian-zone-high-angle-view.mp4")
    
    #create_input_convnext("data/video/many-car-street-15244647.jpg")
