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
_C.DETECT.VERBOSE = False # whether to print the results


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

_C.SYNOPSIS.MARK_FOLDER = 'src\\marker\\video4'

_C.SYNOPSIS.START_FRAME = 0 # synopsis video starting frame index
_C.SYNOPSIS.MAX_LENGTH = 200 # maximum synopsis video length (in frames)
_C.SYNOPSIS.FPS = 30 # using to render synopsis video
                     # -1 if using source video FPS

_C.SYNOPSIS.OMEGA_A = 1 # activity weight
_C.SYNOPSIS.OMEGA_C = 100 # collisions weight
_C.SYNOPSIS.OMEGA_T = 1e-8 # chronological weight
_C.SYNOPSIS.OMEGA_SD = 1e-3 # smooth and deviation weight


# TUBE
_C.SYNOPSIS.TUBE = CN() # a tube is used to manage an object 
_C.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH = 48
_C.SYNOPSIS.TUBE.RATIO_SIZE_RANGE = [0.9, 1.]
_C.SYNOPSIS.TUBE.RATIO_SPEED_RANGE = [0.3, 2.]

_C.SYNOPSIS.TUBE.DEVIATION_SIGMA = 2. # emprically set as 2 
_C.SYNOPSIS.TUBE.DEVIATION_WEIGHT_V = 10000 # velocity weight in deviation term
_C.SYNOPSIS.TUBE.DEVIATION_WEIGHT_S = 10000 # size weight in deviation term
_C.SYNOPSIS.TUBE.SMOOTH_WEIGHT_V = 0.01 # velocity weight in smooth term
_C.SYNOPSIS.TUBE.SMOOTH_WEIGHT_S = 0.01 # size weight in smooth term


# Markov-chain Monte-Carlo
_C.SYNOPSIS.MCMC = CN()
_C.SYNOPSIS.MCMC.NUM_ITERATIONS = 300000 # maximum num iterations to compute MCMC


