from ultralytics import YOLO
from .base import BaseInfrence

class YOlO8(BaseInfrence):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg.clone()
        self.load_model(self.cfg, **kwargs)
    

    def load_model(self, cfg, **kwargs):
        model = kwargs.get('model', None)
        model = cfg.DETECT.MODEL if model is None else model
        self.model = YOLO(model)
        self.names = self.model.names 


    def detect(self, batch, **kwargs):
        imgsz = self.cfg.DETECT.SZ
        results = self.model(batch,
                             imgsz = imgsz, 
                             classes=kwargs['classes'],
                             verbose=kwargs['verbose'],
                             conf=kwargs['conf'])
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
