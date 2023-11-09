from ultralytics import YOLO
import torch

from .base import BaseInfrence
from .engines.yolo8_detect.models import TRTModule
from .engines.yolo8_detect.run import yolo8engine

class YOLO8(BaseInfrence):
    def __init__(self, cfg):
        super().__init__()
        self.load_model(cfg)
    

    def load_model(self, cfg):
        """Load Detection model

        Parameters:
        -----------
        cfg, CfgNode instance
            params config
        """
        self.cfg = cfg
        model = self.cfg.DETECT.MODEL # model path

        if isinstance(model, str):
            suffix = model.split('.')[-1]

            if suffix in ['pt', 'pth']:
                self.model = YOLO(model)
                self.engine = False
            elif suffix in ['engine']:
                self.model = TRTModule(model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                self.engine = True   
        else:
            self.model = model # TODO change the arguments of load model function
        # self.names = self.model.names 


    def __detect(self, batch, **kwargs): # inference phase for spceify model 
        """YOLO8 inference

        Parameters:
        -----------
        batch, list[np.ndarray]
            list of input images
        """
        classes = self.cfg.DETECT.CLASSES
        device = kwargs.get('device', torch.device('cuda:0'))
        if self.engine: # Run engine 
            results = yolo8engine(batch, 
                                  self.model, 
                                  classes=classes,
                                  device=device
                                  )
        else: # Run model
            imgsz = self.cfg.DETECT.SZ
            conf = self.cfg.DETECT.CONF
            verbose = kwargs.get('verbose', False)
            results = self.model(batch,
                                 imgsz = imgsz, 
                                 conf=conf,
                                 classes=classes,
                                 verbose=verbose,
                                )
            results = [pred.boxes.data.cpu().numpy() for  pred in results]
        return results 
    

    def detect(self, batch, **kwargs):
        method = self.cfg.DETECT.METHOD
        if method=='sliding_window':
            window_sz = self.cfg.DETECT.WINDOW_SIZE
            overlap_ratio = self.cfg.DETECT.OVERLAP_WINSIZE
            overlap_area_thresh = self.cfg.DETECT.AREA_NMS_THRESH
            results = self.sliding_window_infer(batch, 
                                                window_sz, 
                                                overlap_ratio, 
                                                overlap_area_thresh,
                                                self.__detect, 
                                                **kwargs)
        elif method=='grid':
            grid_sz = self.cfg.DETECT.GRID_SIZE
            results = self.grid_infer(batch, 
                                      grid_sz,
                                      self.__detect, 
                                      **kwargs)
        else:
            results = self.normal_infer(batch, self.__detect, **kwargs)
        return results


    def __call__(self, batch, **kwargs):
        results = self.detect(batch, **kwargs)
        return results
