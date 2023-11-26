from config.default import _C
from lib.detection.yolo8 import YOLO8

if __name__ == '__main__':
    v8 = YOLO8(_C.clone())
    v8()
