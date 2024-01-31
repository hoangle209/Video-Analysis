from .yolo5 import YOLO5
from .yolo8 import YOLO8

detector_factory = {
    'yolov5': YOLO5,
    'yolov8': YOLO8
}

__all__ = [
    'YOLO5', 'YOLO8', 'detector_factory'
]