from yolov8.training import load_model
import cv2


def detector(data_path):
    model = load_model()
    results = model.predict(source = data_path, show = True)
    return results


