from dataclasses import dataclass
import numpy as np
import math

import sys

@dataclass
class Tube:
    v_translate: np.ndarray # 2D matrix
    ID: int

    # begin and end frame in synopsis / source video
    time_start: int 
    time_end: int 
    src_time_start: int 
    src_time_end: int
    
    segment_length: np.ndarray # store length for each tube's segment in synopsis video
    num_segments: int 
    ratio_v: np.ndarray # velocity ratio for each segment
    ratio_size: np.ndarray # size ratio for each segment
    min_size: np.ndarray # minimum size of an object (for each segment ???) 



    def __init__(self, cfg):
        self.cfg = cfg

        self.ID:int = -1

        self.time_start:int = 0
        self.time_end:int = 0 
        self.src_time_start:int = 0
        self.src_time_end:int = 0
        
        self.segment_length:np.ndarray = None 
        self.num_segments:int = 0
        self.ratio_v:np.ndarray = None 
        self.ratio_size:np.ndarray = None 
        self.min_size:np.ndarray = None 


    def __v_translating(self, v_tranlong):
        v_translate = np.empty(shape=(v_tranlong, v_tranlong), dtype=np.int64)
        for i in range(1, v_tranlong):
            for j in range(i):
                v_translate[i, j] = j / i * self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH # ??
        return v_translate


    def initialization(self, ID, src_time_start, src_time_end):
        unit_segment_length = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH
        self.v_translate = self.__v_translating(4 * unit_segment_length + 1) # ???
        self.ID = ID
        self.src_time_start = src_time_start
        self.src_time_end = src_time_end
        self.src_length = src_time_end - src_time_start + 1 # in frame
        self.num_segments = math.ceil(self.src_length / unit_segment_length)

        self.segment_length = np.full(shape=(self.num_segments, ), fill_value=unit_segment_length)
        self.ratio_v = np.full(shape=(self.num_segments, ), fill_value=1)
        self.ratio_size = np.full(shape=(self.num_segments, ), fill_value=1)
        self.min_size = np.full(shape=(self.num_segments, ), fill_value=sys.maxint)
    

    def __compute_activity_enery(self, synopsis_video_length):
        activity_enery = 0
        for i, seg_len in enumerate(self.segment_length):
            for j in range(seg_len):
                pass
        return activity_enery
        

    
    

    
