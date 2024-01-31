from ultralytics import YOLO
import torch

from ..utils import get_pylogger
from .base import BaseInference

LOGGER = get_pylogger(__name__)

try:
    from .engines.yolo8_detect.models import TRTModule
    from .engines.yolo8_detect.run import yolo8engine
except:
    LOGGER.warning('Cannot import trt model')


class YOLO8(BaseInference):
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
        model = self.cfg.DETECT.WEIGHT # model path

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


    def __detect(self, batch): # inference phase for spceify model 
        """YOLO8 inference

        Parameters:
        -----------
        batch, list[np.ndarray]
            list of input images
        """
        classes = self.cfg.DETECT.CLASSES
        device = torch.device(self.cfg.DETECT.DEVICES)
        if self.engine: # Run engine 
            results = yolo8engine(batch, 
                                  self.model, 
                                  classes=classes,
                                  device=device, )
        else: # Run model
            imgsz = self.cfg.DETECT.SZ
            conf = self.cfg.DETECT.CONF
            verbose = self.cfg.DETECT.VERBOSE
            results = self.model(batch,
                                 imgsz=imgsz, 
                                 conf=conf,
                                 classes=classes,
                                 verbose=verbose,
                                 device=device, )
            results = [pred.boxes.data.cpu().numpy() for pred in results]
        return results 
    

    def normal_infer(self, batch):
        return super().normal_infer(batch,
                                    det_function=self.__detect, )


    def grid_infer(self, batch, ):
        grid_sz = self.cfg.DETECT.GRID.GRID_SIZE
        return super().grid_infer(batch, 
                                  grid_sz=grid_sz,
                                  det_function=self.__detect, )


    def sliding_window_infer(self, batch, ):
        window_sz = self.cfg.DETECT.SLIDING_WINDOW.WINDOW_SIZE
        overlap_ratio = self.cfg.DETECT.SLIDING_WINDOW.OVERLAP_WINDOW_SIZE_RATIO
        overlap_area_thresh = self.cfg.DETECT.SLIDING_WINDOW.NMS_THRESH
        return super().sliding_window_infer(batch, 
                                            window_sz=window_sz, 
                                            overlap_ratio=overlap_ratio, 
                                            overlap_area_thresh=overlap_area_thresh,
                                            det_function=self.__detect, )  


    def detect(self, batch):
        method = self.cfg.DETECT.METHOD
        if method=='sliding_window':
            results = self.sliding_window_infer(batch, )
        elif method=='grid':
            results = self.grid_infer(batch, )
        else:
            results = self.normal_infer(batch, )
        return results


    def __call__(self, batch):
        results = self.detect(batch)
        return results
