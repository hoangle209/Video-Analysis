from config.default import _C
from detection.yolo8 import YOlO8

if __name__ == '__main__':
    v8 = YOlO8(_C.clone())
    v8()
