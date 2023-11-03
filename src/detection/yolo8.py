from ultralytics import YOLO
import torch

from .base import BaseInfrence
from .engines.yolo8_detect.models import TRTModule
from .engines.yolo8_detect.run import yolo8engine

class YOLO8(BaseInfrence):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.load_model(self.cfg, **kwargs)
    

    def load_model(self, cfg, **kwargs):
        model = kwargs.get('model', None)
        model = cfg.DETECT.MODEL if model is None else model

        if isinstance(model, str):
            suffix = model.split('.')[-1]

            if suffix in ['pt', 'pth']:
                self.model = YOLO(model)
                self.engine = False
            elif suffix in ['engine']:
                self.engine = True
                self.model = TRTModule(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            self.model = model
        # self.names = self.model.names 


    def detect(self, batch, **kwargs):
        if self.engine: # Run engine model
            results = yolo8engine(batch, self.model)
        else: # Run model
            imgsz = self.cfg.DETECT.SZ
            conf = self.cfg.DETECT.CONF
            classes = kwargs.get('classes', 0)
            verbose = kwargs.get('verbose', False)
            results = self.model(batch,
                                 imgsz = imgsz, 
                                 conf=conf,
                                 classes=classes,
                                 verbose=verbose,
                                )
            results = [pred.boxes.data.cpu().numpy() for  pred in results]
        return results 
    

    def __detect(self, batch, **kwargs):
        method = self.cfg.DETECT.METHOD
        if method=='sliding_window':
            results = self.sliding_window_infer(batch, self.cfg, **kwargs)
        elif method=='grid':
            results = self.grid_infer(batch, self.cfg, **kwargs)
        else:
            results = self.normal_infer(batch, **kwargs)
        return results


    def __call__(self, batch, **kwargs):
        results = self.__detect(batch, **kwargs)
        return results
