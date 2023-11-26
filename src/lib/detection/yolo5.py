import torch
from .base import BaseInference


class YOLO5(BaseInference):
    def __init__(self, cfg):
        super().__init__()
        self.load_model(cfg)
    

    def load_model(self, cfg):
        """Load Detection model

        Parameters:
        -----------
        cfg, CfgNode instance
            param config
        """
        self.cfg = cfg
        model = self.cfg.DETECT.MODEL 

        if isinstance(model, str):
            suffix = model.split('.')[-1]

            if suffix in ['pt', 'pth']:
                self.model =  torch.hub.load('ultralytics/yolov5', 'custom', path=model)
                self.engine = False
        else:
            self.model = model # TODO change the arguments of load model function
        # self.names = self.model.names 
    

    def __detect(self, batch): # inference phase for spceify model 
        """YOLO5 inference

        Parameters:
        -----------
        batch, list[np.ndarray]
            list of input images
        """
        device = torch.device(self.cfg.DETECT.DEVICES)
        imgsz = self.cfg.DETECT.SZ
        self.model.conf = self.cfg.DETECT.CONF
        self.model.classes = self.cfg.DETECT.CLASSES
        self.model.verbose = self.cfg.DETECT.VERBOSE
        self.model = self.model.to(device)

        results = self.model(batch, 
                             size=imgsz, )
        results = [pred.cpu().numpy() for pred in results.xyxy]
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
