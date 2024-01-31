from dataclasses import dataclass
import numpy as np
import os
from sys import maxsize
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Any, List

from .tube import Tube


@dataclass
class TubeManager:
    mark_folder: str # path that store video metadata
    src_fps: int # source video FPS
    num_tubes: int # the number of objects
    tubes: List[Tube]

    synopsis_begin_frame: int 
    synopsis_video_length: int # maximum length of synopsis video

    activity_energy: np.ndarray
    smooth_deviation_energy: np.ndarray # 2D
    collision_energy: np.ndarray 
    time_energy: np.ndarray # 2d # chronological energy

    M: int # ξ value, using when calcualting time (chronological) energy
           # maximum number of frames that can be tolerated to be inverted
    
    # before energy is used to store energy of a specify tube
    before_activity_energy: float
    before_smooth_deviation_energy: float
    before_collision_energy: np.ndarray # 2d
    before_time_energy: np.ndarray # 2d

    energy: float # total energy
    best_result: List[np.ndarray] # time start, segments length and size ratio


    def __init__(self, cfg):
        self.cfg = cfg

        self.synopsis_begin_frame = cfg.SYNOPSIS.START_FRAME
        self.synopsis_video_length = cfg.SYNOPSIS.MAX_LENGTH

        self.mark_folder = cfg.COMMON.MARKER_PATH # TODO make mark folder link to tracking result save path
        tube_data_file = open(os.path.join(self.mark_folder, 'tubeframe.txt'), 'r')
        self.src_fps = int(tube_data_file.readline())
        self.num_tubes = int(tube_data_file.readline())
        self.tubes = [Tube(self.cfg) for _ in range(self.num_tubes)]
        self.best_result = [None for _ in range(self.num_tubes)]

        for tube_idx in range(self.num_tubes):
            tube_data = tube_data_file.readline()
            tube_id, src_time_start, src_time_end = list(map(int, tube_data.split())) # ID, source frame start, source frame end 
                                                                                      # tube id is counted from 1, tube index is stored from 0
            self.tubes[tube_idx].initialization(tube_id, src_time_start, src_time_end)
            self.tubes[tube_idx].read_frame_bounding_box(self.mark_folder)
            self.tubes[tube_idx].read_frame_fg_bg_diff_value(self.mark_folder)

            # init starting frame for each tube
            self.tubes[tube_idx].time_start = int(tube_idx / self.num_tubes * self.synopsis_video_length + self.synopsis_begin_frame) 

            self.best_result[tube_idx] = np.zeros(shape=(1 + 2*self.tubes[tube_idx].num_segments)) # the first idx is to store time_start, 
                                                                                                   # next num_segments idxes are to store seg_length,
                                                                                                   # last num_segments idxes are to store ratio_size 
        tube_data_file.close()

        self.M = self.__compute_M()
        self.energy = 0

        # use last position to store total energy
        self.time_energy = np.zeros(shape=(self.num_tubes+1, self.num_tubes+1))
        self.collision_energy = np.zeros(shape=(self.num_tubes+1, self.num_tubes+1))
        self.activity_energy = np.zeros(shape=(self.num_tubes+1))
        self.smooth_deviation_energy = np.zeros(shape=(self.num_tubes+1))

        self.before_time_energy = np.zeros(shape=(2, self.num_tubes+1))
        self.before_collision_energy = np.zeros(shape=(self.num_tubes+1))
        self.before_activity_energy = 0
        self.before_smooth_deviation_energy = 0


    def __compute_M(self):
        """
        ξ is automatically determined as follows. 
        Assuming f1 , f2 , . . . , fN are ascending numbers, with fi is source time start of tube with id i,
        compute f3 - f1 , f4 - f2 , . . . , fN - fN-2 first. 
        The median of these numbers is selected as ξ.
        """
        per_2 = np.zeros(shape=(self.num_tubes-2), dtype=np.int64)
        for i in range(self.num_tubes - 2):
            per_2[i] = abs(self.tubes[i+2].src_time_start - self.tubes[i].src_time_start)

        _M = np.median(per_2)
        # print('Calculating maximum frames that can be tolerated')
        return _M
    

    def __compute_boxes_overlap(self, box1, box2, ratio_size1, ratio_size2):
        """Compute overlapping area of the two boxes
        each box format is (xmin, ymin, w, h, ratio_size), new box value can be computed as follow:
            cx, cy = xmin + w/2, ymin + h/2
            xmin_new = cx - w/2*ratio_size = xmin + w/2 - w/2*ratio_size = xmin + (1-ratio_size)*w/2
            ymin_new = cy - h/2*ratio_size = ymin + h/2 = h/2*ratio_size = ymin + (1-ratio_size)*h/2
            xmax_new, ymax_new = xmin_new + w*ratio_size, ymin_new + h*ratio_size
        """
        x1min, y1min, w1, h1 = box1
        x2min, y2min, w2, h2 = box2

        x1min = x1min + w1/2 * (1-ratio_size1)
        y1min = y1min + h1/2 * (1-ratio_size1)
        x1max, y1max = x1min + w1*ratio_size1, y1min + h1*ratio_size1

        x2min = x2min + w2/2 * (1-ratio_size2)
        y2min = y2min + h2/2 * (1-ratio_size2)
        x2max, y2max = x2min + w2*ratio_size2, y2min + h2*ratio_size2
        
        xmin, ymin = max(x1min, x2min), max(y1min, y2min)
        xmax, ymax = min(x1max, x2max), min(y1max, y2max)

        # return overlap area
        if xmin >= xmax or ymin >= ymax:
            return 0
        return (xmax - xmin) * (ymax - ymin) 


    def __compute_collision_lock_in_time(self, 
                                         sooner_id,
                                         sooner_time_start,
                                         later_time_start, ):
        """Determines the frame index that two tubes start having collision

        Parameters:
        -----------
            sooner_id, int:

            sooner_time_start, int:

            later_time_start, int:

        Returns:
        -----------
            0: have collision, return:        
            - synopsis segment index 
            - synopsis frame index in that segment

            1: have no collision, return None
        """
        for i in range(self.tubes[sooner_id].num_segments): # i: segment index
            if sooner_time_start + self.tubes[sooner_id].segments_length[i] >= later_time_start:
                j = int(later_time_start - sooner_time_start) 
                sooner_time_start += j
                if sooner_time_start == later_time_start:
                    unit_segment_length = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH
                    sooner_src_frame_idx_in_segment = self.tubes[sooner_id].v_translate[self.tubes[sooner_id].segments_length[i], j-1]
                    if i * unit_segment_length + sooner_src_frame_idx_in_segment < self.tubes[sooner_id].src_length:
                        sooner_segment_idx = i
                        sooner_frame_idx_in_segment = j-1
                        return 0, (sooner_time_start, 
                                    sooner_segment_idx, 
                                    sooner_frame_idx_in_segment)
                    else:
                        return 1, None
            else:
                sooner_time_start += self.tubes[sooner_id].segments_length[i] 
        return 1, None


    def __compute_collision_energy(self, 
                                   id1, id2, 
                                   reduce='sum',
                                   use_ratio_size=False):
        """Finds the time interval during which both the two tubes occur, 
        and then sum up the intersection area of the occurrences during the time interval.

        Parameters:
        -----------
            reduce, str | None: 
                default None, ['sum', 'none', None]
                if two tubes do not have any collision, always return 0
                else:
                    'None': return collsion energy for id1 each tube's segment 
                    'sum': return the total energy of tube 
            use_ratio_size, bool:
                default False
        """
        if id1 == id2:
            return 0
        if (self.tubes[id1].time_start > (self.tubes[id2].time_start + self.tubes[id2].get_tube_length()) or
            self.tubes[id2].time_start > (self.tubes[id1].time_start + self.tubes[id1].get_tube_length())
            ): return 0
        
        collision_energy = np.zeros(shape=self.tubes[id1].num_segments)
        unit_tube_segment = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH
        id1_segment_idx = 0
        id2_segment_idx = 0
        id1_frame_idx_in_segment = 0
        id2_frame_idx_in_segment = 0
        id1_time_start = self.tubes[id1].time_start
        id2_time_start = self.tubes[id2].time_start

        # output of __compute_collision_lock_in_time get id1_time_start = id2_time_Start
        if self.tubes[id1].time_start > self.tubes[id2].time_start:
            flag, id2_collision_infor = self.__compute_collision_lock_in_time(id2, 
                                                                              id2_time_start, 
                                                                              id1_time_start, )
            if flag: # flag that the two tubes do not collide
                return 0
            else: 
                id2_time_start, id2_segment_idx, id2_frame_idx_in_segment = id2_collision_infor 
        
        if self.tubes[id2].time_start > self.tubes[id1].time_start:
            flag, id1_collision_infor = self.__compute_collision_lock_in_time(id1, 
                                                                              id1_time_start, 
                                                                              id2_time_start, )
            if flag: # flag that the two tubes do not collide
                return 0
            else:
                id1_time_start, id1_segment_idx, id1_frame_idx_in_segment = id1_collision_infor

        # source frame index is counted from 0 to source_length for each tube
        # src_idx of frame that start having collision :
        # src_idx = unit_segment_length * segment_idx + frame_idx_in_segment
        id1_src_frame_idx = id1_segment_idx * unit_tube_segment + \
                            self.tubes[id1].v_translate[self.tubes[id1].segments_length[id1_segment_idx], id1_frame_idx_in_segment]
        id2_src_frame_idx = id2_segment_idx * unit_tube_segment + \
                            self.tubes[id2].v_translate[self.tubes[id2].segments_length[id2_segment_idx], id2_frame_idx_in_segment]

        while True:
            # check if source frame index is out of tube's source length 
            if not (
                self.tubes[id1].src_length > id1_src_frame_idx and
                self.tubes[id2].src_length > id2_src_frame_idx
                ): break
            
            if id1_time_start >= self.synopsis_begin_frame and id1_time_start <= (self.synopsis_begin_frame + self.synopsis_video_length):
                q = self.__compute_boxes_overlap(self.tubes[id1].frame_bounding_box[id1_src_frame_idx],
                                                 self.tubes[id2].frame_bounding_box[id2_src_frame_idx],
                                                 self.tubes[id1].ratio_size[id1_segment_idx],
                                                 self.tubes[id2].ratio_size[id2_segment_idx])
                if q > 0 and not use_ratio_size:
                    q = self.__compute_boxes_overlap(self.tubes[id1].frame_bounding_box[id1_src_frame_idx],
                                                     self.tubes[id2].frame_bounding_box[id2_src_frame_idx],
                                                     1, 1)
                collision_energy[id1_segment_idx] += q
                    
            id1_frame_idx_in_segment += 1
            if id1_frame_idx_in_segment == self.tubes[id1].segments_length[id1_segment_idx]:
                id1_segment_idx += 1
                if id1_segment_idx == self.tubes[id1].num_segments:
                    break
                id1_frame_idx_in_segment = 0
            
            id2_frame_idx_in_segment += 1
            if id2_frame_idx_in_segment == self.tubes[id2].segments_length[id2_segment_idx]:
                id2_segment_idx += 1
                if id2_segment_idx == self.tubes[id2].num_segments:
                    break
                id2_frame_idx_in_segment = 0
            
            # update next source frames idx that have collision of two tubes
            id1_src_frame_idx = id1_segment_idx * unit_tube_segment + \
                                self.tubes[id1].v_translate[self.tubes[id1].segments_length[id1_segment_idx], id1_frame_idx_in_segment]
            id2_src_frame_idx = id2_segment_idx * unit_tube_segment + \
                                self.tubes[id2].v_translate[self.tubes[id2].segments_length[id2_segment_idx], id2_frame_idx_in_segment]
        
        if reduce is None or reduce == 'none':
            return collision_energy
        elif reduce == 'sum':
            return collision_energy.sum().item()
    

    def __compute_time_energy(self, id1, id2): 
        """A tube appears earlier in the original video should be earlier in the synopsis video too.
        The chronological term is used to penalize the invert of the chronological order of any pair of object tubes.
        """
        if id1 == id2:
            return 0
        
        time_energy = 0
        if (self.tubes[id1].src_time_start > self.tubes[id2].src_time_start and 
            self.tubes[id1].time_start < self.tubes[id2].time_start):     

            id1_time_start = self.tubes[id1].time_start
            for id1_seg_idx in range(self.tubes[id1].num_segments):
                mark = id1_seg_idx * self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH

                for id1_frame_idx_in_seg in range(self.tubes[id1].segments_length[id1_seg_idx]):
                    if id1_time_start < self.tubes[id2].time_start:
                        id1_synopsis_segment_length = self.tubes[id1].segments_length[id1_seg_idx]
                        id1_src_frame_idx = mark + self.tubes[id1].v_translate[id1_synopsis_segment_length, id1_frame_idx_in_seg]
                        time_energy += self.tubes[id1].frame_fg_bg_diff[id1_src_frame_idx]
                    else:
                        # The chronological term should be proportional to the temporal distance between two shifted tubes. 
                        # Therefore define the following weight:
                        weight = 1. / ((1 - min(abs(self.tubes[id1].time_start - self.tubes[id2].time_start), 
                                                self.M) / self.M) **5 + 1e-5) - 1
                        time_energy = weight * time_energy
                        return time_energy

                    id1_time_start += 1
        
        return time_energy


    def get_collision_energy(self,
                             id1, id2,
                             id1_temporary_time_start=None,
                             temporary_seg_idx=None,
                             id1_temporary_seg_length=None,
                             temporary_ratio_size=None,
                             reduce='sum',
                             use_ratio_size=False):
        """

        Parameters:
        -----------
        """
        if (id1_temporary_time_start is not None and
            temporary_seg_idx is not None and 
            id1_temporary_seg_length is not None and 
            temporary_ratio_size is not None):

            id1_tmp_time_start = self.tubes[id1].time_start
            id1_tmp_seg_length = self.tubes[id1].segments_length[temporary_seg_idx]
            id1_tmp_ratio_size = self.tubes[id1].ratio_size[temporary_seg_idx]
            id1_tmp_ratio_v = self.tubes[id1].ratio_v[temporary_seg_idx]

            self.tubes[id1].time_start = id1_temporary_time_start
            self.tubes[id1].segments_length[temporary_seg_idx] = id1_temporary_seg_length
            self.tubes[id1].ratio_size[temporary_seg_idx] = temporary_ratio_size
            self.tubes[id1].ratio_v[temporary_seg_idx] = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH / id1_temporary_seg_length

            _collision_energy = self.__compute_collision_energy(id1, id2, reduce, use_ratio_size)

            self.tubes[id1].time_start = id1_tmp_time_start
            self.tubes[id1].segments_length[temporary_seg_idx] = id1_tmp_seg_length
            self.tubes[id1].ratio_size[temporary_seg_idx] = id1_tmp_ratio_size
            self.tubes[id1].ratio_v[temporary_seg_idx] = id1_tmp_ratio_v
            return _collision_energy
        
        else:
            return self.__compute_collision_energy(id1, id2, reduce, use_ratio_size)


    def get_time_energy(self,
                        id1, id2,
                        temporary_seg_idx=None,
                        id1_temporary_time_start=None,
                        id1_temporary_seg_length=None,
                        id2_temporary_time_start=None,
                        id2_temporary_seg_length=None,
                        ):
        """

        Parameters:
        -----------
        """
        if (id1_temporary_time_start is not None and
            temporary_seg_idx is not None and 
            id1_temporary_seg_length is not None):

            id1_tmp_time_start = self.tubes[id1].time_start
            id1_tmp_seg_length = self.tubes[id1].segments_length[temporary_seg_idx]
            id1_tmp_ratio_v = self.tubes[id1].ratio_v[temporary_seg_idx]

            self.tubes[id1].time_start = id1_temporary_time_start
            self.tubes[id1].segments_length[temporary_seg_idx] = id1_temporary_seg_length
            self.tubes[id1].ratio_v[temporary_seg_idx] = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH / id1_temporary_seg_length

            _time_energy = self.__compute_time_energy(id1, id2)

            self.tubes[id1].time_start = id1_tmp_time_start
            self.tubes[id1].segments_length[temporary_seg_idx] = id1_tmp_seg_length
            self.tubes[id1].ratio_v[temporary_seg_idx] = id1_tmp_ratio_v
            return _time_energy
        
        elif (id2_temporary_time_start is not None and
              temporary_seg_idx is not None and 
              id2_temporary_seg_length is not None):
            
            id2_tmp_time_start = self.tubes[id2].time_start
            id2_tmp_seg_length = self.tubes[id2].segments_length[temporary_seg_idx]
            id2_tmp_ratio_v = self.tubes[id2].ratio_v[temporary_seg_idx]

            self.tubes[id2].time_start = id2_temporary_time_start
            self.tubes[id2].segments_length[temporary_seg_idx] = id2_temporary_seg_length
            self.tubes[id2].ratio_v[temporary_seg_idx] = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH / id2_temporary_seg_length

            _time_energy = self.__compute_time_energy(id1, id2)

            self.tubes[id2].time_start = id2_tmp_time_start
            self.tubes[id2].segments_length[temporary_seg_idx] = id2_tmp_seg_length
            self.tubes[id2].ratio_v[temporary_seg_idx] = id2_tmp_ratio_v
            return _time_energy
        
        else:
            return self.__compute_time_energy(id1, id2)
    
    
    def __compute_energy(self):
        """Compute total energy
        """
        self.activity_energy[-1] = 0
        self.smooth_deviation_energy[-1] = 0
        self.collision_energy[-1, -1] = 0
        self.time_energy[-1, -1] = 0

        for tube_idx in range(self.num_tubes):
            self.activity_energy[tube_idx] = self.tubes[tube_idx].get_activity_energy(self.synopsis_begin_frame, self.synopsis_video_length)
            self.smooth_deviation_energy[tube_idx] = self.tubes[tube_idx].get_smooth_and_deviation_energy()
            
            self.activity_energy[-1] += self.activity_energy[tube_idx]
            self.smooth_deviation_energy[-1] += self.smooth_deviation_energy[tube_idx]

            self.collision_energy[tube_idx, -1] = 0

            for tube_idx_2 in range(tube_idx, self.num_tubes):
                if tube_idx == tube_idx_2:
                    continue
                self.collision_energy[tube_idx, tube_idx_2] = self.get_collision_energy(tube_idx, tube_idx_2)
                self.collision_energy[tube_idx_2, tube_idx] = self.collision_energy[tube_idx, tube_idx_2] # collsion_energy is symmetric 
                
                self.collision_energy[-1, -1] += self.collision_energy[tube_idx, tube_idx_2]
                self.collision_energy[tube_idx, -1] += self.collision_energy[tube_idx, tube_idx_2]

                self.time_energy[tube_idx, tube_idx_2] = self.get_time_energy(tube_idx, tube_idx_2)
                self.time_energy[tube_idx_2, tube_idx] = self.get_time_energy(tube_idx_2, tube_idx)
                self.time_energy[-1, -1] += (self.time_energy[tube_idx, tube_idx_2] + self.time_energy[tube_idx_2, tube_idx])
        
        wa  = self.cfg.SYNOPSIS.OMEGA_A  # activity weight
        wc  = self.cfg.SYNOPSIS.OMEGA_C  # collisions weight
        wt  = self.cfg.SYNOPSIS.OMEGA_T  # chronological weight
        wsd = self.cfg.SYNOPSIS.OMEGA_SD # smooth and deviation weight
        _energy = wa  * self.activity_energy[-1] + \
                  wc  * self.collision_energy[-1, -1] + \
                  wt  * self.time_energy[-1, -1] + \
                  wsd * self.smooth_deviation_energy[-1]
        
        return _energy
    

    def get_energy(self,
                   ID=None,
                   temporary_time_start=None,
                   temporary_seg_idx=0,
                   temporary_seg_length=None,
                   temporary_ratio_size=None):
        """Get total energy function 

        Parameters:
        -----------
            ID, int:
                default: None,
                if specified return energy after exchange the arugments value of a IDth tube
            temporary_time_start, int:

            temporary_seg_idx, int:

            temporary_seg_length, int:

            temporary_ratio_size, int:
        """
        if ID is not None:
            self.before_activity_energy = 0
            self.before_smooth_deviation_energy = 0
            self.before_collision_energy = np.zeros_like(self.before_collision_energy)
            self.before_time_energy = np.zeros_like(self.before_time_energy)

            if temporary_time_start == maxsize:
                temporary_time_start = self.tubes[ID].time_start
            if temporary_seg_length == maxsize:
                temporary_seg_length = self.tubes[ID].segments_length[temporary_seg_idx]
            if temporary_ratio_size == 0:
                temporary_ratio_size = self.tubes[ID].ratio_size[temporary_seg_idx]
            
            self.before_activity_energy = self.tubes[ID].get_activity_energy(self.synopsis_begin_frame,
                                                                             self.synopsis_video_length,
                                                                             temporary_seg_idx,
                                                                             temporary_time_start, 
                                                                             temporary_seg_length, )
            self.before_smooth_deviation_energy = self.tubes[ID].get_smooth_and_deviation_energy(temporary_seg_idx,
                                                                                                 temporary_seg_length,
                                                                                                 temporary_ratio_size, )

            for i in range(self.num_tubes):
                if i == ID:
                    continue

                self.before_collision_energy[i] = self.get_collision_energy(ID, i,
                                                                            temporary_time_start,
                                                                            temporary_seg_idx,
                                                                            temporary_seg_length,
                                                                            temporary_ratio_size)
                self.before_collision_energy[-1] += self.before_collision_energy[i]

                self.before_time_energy[0, i] = self.get_time_energy(ID, i,
                                                                     temporary_seg_idx=temporary_seg_idx,
                                                                     id1_temporary_time_start=temporary_time_start,
                                                                     id1_temporary_seg_length=temporary_seg_length, )
                self.before_time_energy[1, i] = self.get_time_energy(i, ID,
                                                                     temporary_seg_idx=temporary_seg_idx,
                                                                     id2_temporary_time_start=temporary_time_start,
                                                                     id2_temporary_seg_length=temporary_seg_length, )
                self.before_time_energy[0, -1] += self.before_time_energy[0, i]
                self.before_time_energy[1, -1] += self.before_time_energy[1, i]
                
            self.time_energy[ID, -1] = (self.time_energy[ID, :-1].sum() + self.time_energy[:-1, ID].sum())
            self.collision_energy[:-1, -1] = self.collision_energy[:-1, :-1].sum(axis=1)

            self.before_time_energy[0, -1] += self.before_time_energy[1, -1]   
            
            wa  = self.cfg.SYNOPSIS.OMEGA_A  # activity weight
            wc  = self.cfg.SYNOPSIS.OMEGA_C  # collisions weight
            wt  = self.cfg.SYNOPSIS.OMEGA_T  # chronological weight
            wsd = self.cfg.SYNOPSIS.OMEGA_SD # smooth and deviation weight
            _energy = wa  * (self.activity_energy[-1] + self.before_activity_energy - self.activity_energy[ID]) + \
                      wc  * (self.collision_energy[-1, -1] + self.before_collision_energy[-1] - self.collision_energy[ID, -1]) + \
                      wt  * (self.time_energy[-1, -1] + self.before_time_energy[0, -1] - self.time_energy[ID, -1]) + \
                      wsd * (self.smooth_deviation_energy[-1] + self.before_smooth_deviation_energy - self.smooth_deviation_energy[ID])
            return _energy
        
        else:
            self.energy = self.__compute_energy()
            return self.energy


    def set(self, 
            ID, 
            time_start,
            seg_idx,
            seg_length,
            ratio_size):
        """set value to IDth tube
        
        Parameters:
        -----------
        """
        if seg_length != maxsize:
            self.tubes[ID].segments_length[seg_idx] = seg_length
            self.tubes[ID].ratio_v[seg_idx] = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH / seg_length
        
        if time_start != maxsize:
            self.tubes[ID].time_start = time_start
        
        if ratio_size != 0:
            self.tubes[ID].ratio_size[seg_idx] = ratio_size

        self.activity_energy[-1] -= self.activity_energy[ID]
        self.activity_energy[ID] = self.before_activity_energy 
        self.activity_energy[-1] += self.activity_energy[ID]

        if time_start == maxsize:
            self.smooth_deviation_energy[-1] -= self.smooth_deviation_energy[ID]
            self.smooth_deviation_energy[ID] = self.before_smooth_deviation_energy
            self.smooth_deviation_energy[-1] += self.smooth_deviation_energy[ID]

        self.collision_energy[-1, -1] -= self.collision_energy[ID, -1]
        self.collision_energy[ID, -1] = self.before_collision_energy[-1]
        self.collision_energy[-1, -1] += self.collision_energy[ID, -1] 

        self.time_energy[-1, -1] -= self.time_energy[ID, -1]
        self.time_energy[ID, -1] = self.before_time_energy[0, -1]
        self.time_energy[-1, -1] += self.time_energy[ID, -1]

        for i in range(self.num_tubes):
            self.collision_energy[ID, i] = self.before_collision_energy[i]
            self.collision_energy[i, ID] = self.before_collision_energy[i]
            self.time_energy[ID, i] = self.before_time_energy[0, i]
            self.time_energy[i, ID] = self.before_time_energy[1, i]      

        self.collision_energy[:-1, -1] = self.collision_energy[:-1, :-1].sum(axis=1)   

        wa  = self.cfg.SYNOPSIS.OMEGA_A  # activity weight
        wc  = self.cfg.SYNOPSIS.OMEGA_C  # collisions weight
        wt  = self.cfg.SYNOPSIS.OMEGA_T  # chronological weight
        wsd = self.cfg.SYNOPSIS.OMEGA_SD # smooth and deviation weight

        self.energy =  wa  * self.activity_energy[-1]      + \
                       wc  * self.collision_energy[-1, -1] + \
                       wt  * self.time_energy[-1, -1]      + \
                       wsd * self.smooth_deviation_energy[-1]


    def size_reset(self):
        """Reset size ratio of all tubes to 1. 
        if there is no collision, keep the size ratio to 1
        """        
        for i in range(self.num_tubes):
            self.tubes[i].time_start = self.best_result[i][0]
            self.tubes[i].segments_length = np.int64(self.best_result[i][1: 1 + self.tubes[i].num_segments])
            self.tubes[i].ratio_size = np.ones_like(self.tubes[i].ratio_size)

            self.tubes[i].update_ratio_v()
        
        for id1 in range(self.num_tubes):
            collision_energy = np.zeros(shape=self.tubes[id1].num_segments) # store collision energy for each segment of a tube

            for id2 in range(self.num_tubes):
                _collsion_e = self.__compute_collision_energy(id1, id2, reduce='none')
                collision_energy += _collsion_e
            
            for seg_idx in range(self.tubes[id1].num_segments):
                if collision_energy[seg_idx] > 10: # TODO why only 10 ???
                    self.tubes[id1].ratio_size[seg_idx] = self.best_result[id1][1 + seg_idx + self.tubes[id1].num_segments]
                else:
                    self.best_result[id1][1 + seg_idx + self.tubes[id1].num_segments] = 1
    

    def v_reset(self):
        """Reset velocity ratio of all tubes to 1. 
        if there is no collision, keep the velocity ratio to 1
        """
        old_tubes_time_end = np.zeros(shape=(self.num_tubes))
        for tube_idx in range(self.num_tubes):
            self.tubes[tube_idx].time_start = self.best_result[tube_idx][0]
            
            for seg_idx in range(self.tubes[tube_idx].num_segments):
                self.tubes[tube_idx].segments_length[seg_idx] = self.best_result[tube_idx][1 + seg_idx]
                self.tubes[tube_idx].ratio_size[seg_idx] = self.best_result[tube_idx][1 + seg_idx + self.tubes[tube_idx].num_segments]
            
            self.tubes[tube_idx].update_ratio_v()
            old_tubes_time_end[tube_idx] += (self.tubes[tube_idx].time_start + self.tubes[tube_idx].get_tube_length())
        
        old_collision_energy = np.zeros(shape=(self.num_tubes+1))
        for id1 in range(self.num_tubes):
            for id2 in range(self.num_tubes):
                _collision_energy = self.__compute_collision_energy(id1, id2, use_ratio_size=True)
                old_collision_energy[id1] += _collision_energy
        
        for id1 in range(self.num_tubes):
            for id1_seg_idx in range(self.tubes[id1].num_segments):
                __collision_energy = 0
                self.tubes[id1].segments_length[id1_seg_idx] = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH
                self.tubes[id1].ratio_v[id1_seg_idx] = 1

                for id2 in range(self.num_tubes):
                    q = self.__compute_collision_energy(id1, id2, use_ratio_size=True) 
                    __collision_energy += q 

                if ((__collision_energy > old_collision_energy[id1]) | 
                    (self.tubes[id1].time_start + self.tubes[id1].get_tube_length() > old_tubes_time_end[id1] and
                     self.tubes[id1].time_start + self.tubes[id1].get_tube_length() > (self.synopsis_begin_frame + self.synopsis_video_length))
                    ):
                    self.tubes[id1].segments_length[id1_seg_idx] = self.best_result[id1][1 + id1_seg_idx]
                    self.tubes[id1].ratio_v[id1_seg_idx] = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH / self.tubes[id1].segments_length[id1_seg_idx]
                else:
                    self.best_result[id1][1 + id1_seg_idx] = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH


    def size_smoothing(self):
        """
        """
        for tube_idx in range(self.num_tubes):
            if self.tubes[tube_idx].num_segments == 1:
                continue

            A = np.zeros(shape=(2*self.tubes[tube_idx].num_segments - 2, 
                                self.tubes[tube_idx].num_segments))
            b = self.best_result[tube_idx][1 + self.tubes[tube_idx].num_segments: ]

            A[:self.tubes[tube_idx].num_segments, 
              :self.tubes[tube_idx].num_segments] = np.eye(self.tubes[tube_idx].num_segments, self.tubes[tube_idx].num_segments) 
            
            for j in range(self.tubes[tube_idx].num_segments, 
                           2 * self.tubes[tube_idx].num_segments - 2):
                A[j, j - self.tubes[tube_idx].num_segments] = -1
                A[j, j - self.tubes[tube_idx].num_segments + 1] = 2
                A[j, j - self.tubes[tube_idx].num_segments + 2] = -1
            
            AT_A = csr_matrix(A.T@A)
            x = spsolve(AT_A, b) # shape (self.tubes[tube_idx].num_segments, )
            self.best_result[tube_idx][1+self.tubes[tube_idx].num_segments: ] = \
                            np.minimum(x, self.best_result[tube_idx][1+self.tubes[tube_idx].num_segments: ])
            
            min_ratio = self.cfg.SYNOPSIS.TUBE.RATIO_SIZE_RANGE[0]
            _best_ratio_result = self.best_result[tube_idx][1+self.tubes[tube_idx].num_segments: ]
            self.best_result[tube_idx][1+self.tubes[tube_idx].num_segments: ][_best_ratio_result<min_ratio] = min_ratio

