"""
This module evaluates detection results between Engine and the origin model
"""
from collections import defaultdict
import glob
import os
import cv2 as cv
from tqdm import tqdm
import torch

from .tools import CustomEval
from ...yolo8 import YOLO8


def _run_eval(path, cfg_base, cfg_engine):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = YOLO8(cfg_base)
    engine_model = YOLO8(cfg_engine)
    _gts = defaultdict(list)
    _dts = []

    src = glob.glob(os.path.join(path, '*.jpg'))
    for i in tqdm(src):
        name = os.path.split(i)[-1]
        img = cv.imread(i)
        h, w = img.shape[:2]

        # store detection from base model as grouth truth
        rbase = base_model([img])
        for result in rbase[0]:
            x1, y1, x2, y2, *_ = result
            _gts['annotations'].append({
                'area': (x2-x1)*(y2-y1),
                'bbox': list(map(float, result[:4])),
                'category_id': 1,
                'image_id': name[:-4],
                'iscrowd': int(0),
                'segmentation': [],
                'person_id': -1
            })
        _gts['images'].append({
            'file_name': name,
            'id': name[:-4], 
            'width': int(w),
            'height': int(h)
        })

        # store detection from engine
        rengine = engine_model([img], device=device)
        for result in rengine[0]:
            x1, y1, x2, y2, conf, _ = result
            detection = {
                "image_id": name[:-4],
                "category_id": 1,
                "bbox": [x1, y1, x2, y2],
                "score": float("{:.2f}".format(conf)),
                "segmentation": []
            }
            _dts.append(detection)


    _gts['categories'] = [{
            "id": 1,
            "name": "person",
            "supercategory": "person"
        }]
    
    _eval_tool = CustomEval(_gts, _dts)
    _eval_tool.evaluate()
    _eval_tool.accumulate()
    _eval_tool.summarize()

    