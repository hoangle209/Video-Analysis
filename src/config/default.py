from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# DETECTION TASK CONFIG
# -----------------------------------------------------------------------------
_C.DETECT = CN()

_C.DETECT.MODEL = '' # model path
_C.DETECT.SZ = 640 # input size
_C.DETECT.CONF = 0.5
_C.DETECT.CLASSES = (0, ) # None for all classes
_C.DETECT.METHOD = 'sliding_window' # inference mode: normal
                                    #                 grid
                                    #                 silding_window   
# some other configs
_C.DETECT.DEVICES = 'cpu' # device
_C.DETECT.VERBOSE = False # print results or not (for yolo family only)


# used for grid inference mode
_C.DETECT.GRID = CN()
_C.DETECT.GRID.GRID_SIZE = 640 # int or tuple (W, H)

# used for sliding window mode 
_C.DETECT.SLIDING_WINDOW = CN()
_C.DETECT.SLIDING_WINDOW.WINDOW_SIZE = 640 # int or tuple
_C.DETECT.SLIDING_WINDOW.OVERLAP_WINDOW_SIZE_RATIO = 0.3 # float or tuple, must be in-range (0, 1)
_C.DETECT.SLIDING_WINDOW.NMS_THRESH = 0.65 # float, must be in-range (0, 1)


# -----------------------------------------------------------------------------
# SYNOPSIS TASK CONFIG
# -----------------------------------------------------------------------------
_C.SYNOPSIS = CN()
_C.SYNOPSIS.TOTAL_LENGTH = 450 # frames
_C.SYNOPSIS.FPS = 25

_C.SYNOPSIS.TUBE = CN() # a tube is used to manage an object 
_C.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH = 48
