from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# DETECTION TASK CONFIG
# -----------------------------------------------------------------------------
_C.DETECT = CN()

_C.DETECT.MODEL = '' # model path
_C.DETECT.SZ = 640 # input size
_C.DETECT.CONF = 0.5
_C.DETECT.METHOD = 'sliding_window' # inference mode: normal
                                  #                   grid
                                  #                   silding_window   

# used for grid inference mode
_C.DETECT.GRID_SIZE = 640 # int or tuple(W, H)

# used for sliding window mode 
_C.DETECT.WINDOW_SIZE = 640 # int or tuple
_C.DETECT.OVERLAP_WINSIZE = 0.3 # float or tuple, must be in-range (0, 1)
_C.DETECT.AREA_NMS_THRESH = 0.9 # float, must be in-range (0, 1)

