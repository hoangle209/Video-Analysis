from dataclasses import dataclass
from typing import Any
import numpy as np
import math
import os
from sys import maxsize

@dataclass
class Tube:
    v_translate: np.ndarray # 2d # row_idx is synopsis tube's segment length, 
                                        # col_idx is frame index (in a synopsis tube's segment)
                                        # store src_frame_idx (of a segment) corresponding to current_frame_idx (of a tube's segment) 
    ID: int # object ID

    # begin and end frame in synopsis / source video of an object
    time_start: int 
    time_end: int 
    src_time_start: int 
    src_time_end: int
    
    src_length: int # tube's length (in frame) in source video
    segments_length: np.ndarray # store length for each tube's segment in synopsis video
    num_segments: int # the number of segments of a tube
    ratio_v: np.ndarray # speed ratio for each segment
    ratio_size: np.ndarray # size ratio for each segment
    min_size: np.ndarray # minimum size of an object (for each segment ???) 

    frame_fg_bg_diff: np.ndarray # total pixel values different between frames and background ???
    frame_bounding_box: np.ndarray # objects's bounding box in frame


    def __init__(self, cfg):
        self.cfg = cfg

        self.ID = -1

        self.time_start = 0
        self.time_end = 0 
        self.src_time_start = 0
        self.src_time_end = 0
        
        self.src_length = 0
        self.segments_length = None 
        self.num_segments = 0
        self.ratio_v = None 
        self.ratio_size = None 
        self.min_size = None 

        self.frame_fg_bg_diff = None
        self.frame_bounding_box = None


    def __v_translating(self, v_tranlong):
        """Calculate source frame idx (of a segment)
        from a current synopsis segment length and synopsis frame idx (of a tube's segment)
        
                                unit_segment_length               
        src_frame_idx =   --------------------------------   *   synopsis_frame_idx_in_segment
                            synopsis_tube_segment_length 

        Paramters:
        ----------
            v_tranlong, int:
                maximum length of a synopsis tube's segment
        """
        unit_segment_length = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH
        v_translate = np.empty(shape=(v_tranlong, v_tranlong), dtype=np.int64)
        for synopsis_seg_length in range(1, v_tranlong):
            for frame_idx_in_seg in range(synopsis_seg_length):
                v_translate[synopsis_seg_length, frame_idx_in_seg] = frame_idx_in_seg * unit_segment_length / synopsis_seg_length 
        return v_translate


    def initialization(self, ID, src_time_start, src_time_end):
        unit_segment_length = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH
        max_speed_ratio = self.cfg.SYNOPSIS.TUBE.RATIO_SPEED_RANGE[0]
        v_tranlong =  math.ceil(1 / max_speed_ratio) * unit_segment_length + 1
        self.v_translate = self.__v_translating(int(v_tranlong)) 
        
        self.ID = ID
        self.src_time_start = src_time_start
        self.src_time_end = src_time_end
        self.src_length = src_time_end - src_time_start + 1 # in frame
        self.num_segments = math.ceil(self.src_length / unit_segment_length)

        self.segments_length = np.full(shape=(self.num_segments, ), fill_value=unit_segment_length, dtype=np.int64)
        self.ratio_v = np.ones(shape=(self.num_segments, ))
        self.ratio_size = np.ones(shape=(self.num_segments, ))
        self.min_size = np.full(shape=(self.num_segments, ), fill_value=maxsize)

        self.time_start = -1
        self.time_end = -1

        self.frame_fg_bg_diff = None
        self.frame_bounding_box = None
    

    def update_ratio_v(self):
        def get_segment_ratio_v(self, segment_length):
            """ This function return the speed ratio of one segment 

            Parameters:
            -----------
                seg_length: int
                    current segment's length in synopsis video
            """
            return self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH / segment_length
        
        for i in range(self.num_segments):
            self.ratio_v[i] = get_segment_ratio_v(self, self.segments_length[i])
    
    
    def read_frame_fg_bg_diff_value(self, mark_folder): # TODO convert to store in json file
        """Read tube background-foreground value in each individual frame
        
        Parameters:
        -----------
            mark_folder: str,
                path to folder that store video's metadata
        """
        self.frame_fg_bg_diff = np.zeros(shape=(self.src_length+self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH*3))
        with open(os.path.join(mark_folder, f'{self.ID}', 'value.txt'), 'r') as f:
            # source frame index is count from 0 to source_length for each tube
            for idx, _ in enumerate(range(self.src_time_start, self.src_time_end + 1)):
                _, value = f.readline().split() # frame index, value
                self.frame_fg_bg_diff[idx] = int(value)


    def read_frame_bounding_box(self, mark_folder): # TODO convert to store in json file
        """Read tube's bounding box in each individual frame

        Parameters:
        -----------
            mark_folder: str,
                path to folder that store video's metadata
        """
        self.frame_bounding_box = np.zeros(shape=(self.src_length, 4)) 
        with open(os.path.join(mark_folder, f'{self.ID}', 'node.txt'), 'r') as f:
            for idx, _ in enumerate(range(self.src_time_start, self.src_time_end + 1)):
                _, *bb = list(map(int, f.readline().split()))
                self.frame_bounding_box[idx] = np.array(bb) # xmin, ymin, w, h 


    def __compute_activity_energy(self, synopsis_begin_frame, synopsis_video_length):
        """Compute activity energy of tube
        Used to preserve the objects into the synopsis video. 
        """
        activity_energy = 0
        ts = self.time_start

        for seg_idx in range(self.num_segments):
            segment_len = self.segments_length[seg_idx]
            mark = seg_idx * self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH

            for k in range(segment_len):
                if ts < synopsis_begin_frame or ts > (synopsis_begin_frame + synopsis_video_length):
                    activity_energy += self.frame_fg_bg_diff[mark + self.v_translate[segment_len, k]] * self.ratio_v[seg_idx]
                ts += 1 
        return activity_energy
                    

    def __compute_smooth_and_deviation_energy(self):
        """Compute smooth and deviation energy of tube
        Smooth term, the speed and size variables of the same tube to vary smoothly over the segments
        of the tube. 
        Deviation term, the speed and size should not be changed too much.
        """
        smooth_energy = 0
        deviation_energy = 0

        #                                   sigma                             sigma  
        #  deviation term:  exp{ ---------------------------- }  +  exp{ --------------- }           
        #                         min{ ratio_v, 1./ratio_v }                ratio_size
        sigma = self.cfg.SYNOPSIS.TUBE.DEVIATION_SIGMA
        deviation_weight_v = self.cfg.SYNOPSIS.TUBE.DEVIATION_WEIGHT_V
        deviation_weight_s = self.cfg.SYNOPSIS.TUBE.DEVIATION_WEIGHT_S

        for seg_idx in range(self.num_segments):
            if self.ratio_v[seg_idx] > 1:
                deviation_energy += (deviation_weight_v * math.exp(sigma * self.ratio_v[seg_idx])) 
            else:
                deviation_energy += (deviation_weight_v * math.exp(sigma / self.ratio_v[seg_idx]))

            deviation_energy += (deviation_weight_s * math.exp(sigma / self.ratio_size[seg_idx]))

        # smooth term: F[(2*v[i] - v[i-1] - v[i+1])] + F[(2*s[i] - s[i-1] - s[i+1])]
        # with F is activate function, e.g. to the power of 2 || exp
        smooth_weight_v = self.cfg.SYNOPSIS.TUBE.SMOOTH_WEIGHT_V
        smooth_weight_s = self.cfg.SYNOPSIS.TUBE.SMOOTH_WEIGHT_S
        F = lambda x: math.exp(x) # TODO flexibility setup smooth active function

        for seg_idx in range(1, self.num_segments - 1):
            smooth_energy += smooth_weight_v * F((2*self.ratio_v[seg_idx] - self.ratio_v[seg_idx-1] - self.ratio_v[seg_idx+1]))
            smooth_energy += smooth_weight_s * F((2*self.ratio_size[seg_idx] - self.ratio_size[seg_idx-1] - self.ratio_size[seg_idx+1]))
        
        return (smooth_energy + deviation_energy) / self.num_segments


    def __compute_tube_length(self):
        remainder = self.src_length % self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH
        if  remainder == 0:
            tube_length = self.segments_length.sum()
        else:
            tube_length = self.segments_length[:-1].sum()
            tube_length += int(remainder*(self.segments_length[-1] / self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH))
        return tube_length


    def get_activity_energy(self, 
                            synopsis_begin_frame, 
                            synopsis_video_length,
                            temporary_seg_idx=None,
                            temporary_time_start=None,
                            temporary_seg_length=None
                            ):
        """
        Parameters:
        -----------
        """
        if (temporary_seg_idx is not None and 
            temporary_time_start is not None and 
            temporary_seg_length is not None):
            
            tmp_seg_len = self.segments_length[temporary_seg_idx]
            self.segments_length[temporary_seg_idx] = temporary_seg_length
            tmp_ratio_v = self.ratio_v[temporary_seg_idx]
            self.ratio_v[temporary_seg_idx] = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH / temporary_seg_length
            tmp_time_start, self.time_start = self.time_start, temporary_time_start

            _activity_energy = self.__compute_activity_energy(synopsis_begin_frame, synopsis_video_length)

            self.segments_length[temporary_seg_idx] = tmp_seg_len
            self.ratio_v[temporary_seg_idx] = tmp_ratio_v
            self.time_start = tmp_time_start
            return _activity_energy
        
        else:
            return self.__compute_activity_energy(synopsis_begin_frame, synopsis_video_length)
        

    def get_smooth_and_deviation_energy(self,
                                        temporary_seg_idx=None,
                                        temporary_seg_length=None,
                                        temporary_size_ratio=None):
        """
        Parameters:
        -----------
        """
        if (
            temporary_seg_idx is not None and 
            temporary_seg_length is not None and 
            temporary_size_ratio is not None
            ):
            
            tmp_seg_len = self.segments_length[temporary_seg_idx]
            self.segments_length[temporary_seg_idx] = temporary_seg_length
            tmp_ratio_v = self.ratio_v[temporary_seg_idx]
            self.ratio_v[temporary_seg_idx] = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH / temporary_seg_length
            tmp_ratio_size = self.ratio_size[temporary_seg_idx]
            self.ratio_size[temporary_seg_idx] = temporary_size_ratio

            _smooth_and_deviation_energy = self.__compute_smooth_and_deviation_energy()

            self.segments_length[temporary_seg_idx] = tmp_seg_len
            self.ratio_v[temporary_seg_idx] = tmp_ratio_v
            self.ratio_size[temporary_seg_idx] = tmp_ratio_size
            return _smooth_and_deviation_energy

        else:
            return self.__compute_smooth_and_deviation_energy()


    def get_tube_length(self, 
                        temporary_seg_idx=None,
                        temporary_seg_length=None):
        """
        Parameters:
        -----------
        """
        if temporary_seg_idx is not None and temporary_seg_length is not None:
            tmp_seg_len = self.segments_length[temporary_seg_idx]
            self.segments_length[temporary_seg_idx] = temporary_seg_length
            
            _tube_length = self.__compute_tube_length()

            self.segments_length[temporary_seg_idx] = tmp_seg_len
            return _tube_length
        
        else:
            return self.__compute_tube_length()