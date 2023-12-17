"""Proposal distribution chooses among the following four possible proposals with equal probability:
1. Make a modification to some shift variable ˜fi by randomly choosing an integer in the range [0, Ls), 
where Ls is the length of the synopsis video.
2. Randomly choose an integer e in the range [round(l/a), round(l/b)]. 
then choose a speed variable vij, and set it as l/e.
3. Perturb some resizing variable rij by normal distribution N(0, 0.01^2), 
if the modified variable is out of the user defined range [c, d], sample again.
4. Randomly choose two tubes, and swap the values of their shift variables.
"""
from dataclasses import dataclass
from math import exp
import numpy as np
import random
from sys import maxsize
from tqdm import tqdm

from .tubes.tubemanager import TubeManager

@dataclass
class MCMC:
    qsd: float # start energy value
    best_q: float # best energy value
    beta: float # beta value in Boltzmann function
    q_Boltzmann: float # Boltzmann value from energy 

    def __init__(self, cfg):
        self.cfg = cfg
        self.num_iters = cfg.SYNOPSIS.MCMC.NUM_ITERATIONS

        self.tubes_manager = TubeManager(cfg)
        self.qsd = self.tubes_manager.get_energy() # initital energy
        self.best_q = self.qsd
        
        # MCMC compute energy in form of Boltzmann-like density function
        self.beta = (1 / self.qsd) * 600 
        self.q_Boltzmann = self.__to_Boltzmann_like_value(self.qsd, self.beta) # convert energy to Boltzmann form
        
    
    def __to_Boltzmann_like_value(self, q, beta=None):
        """MCMC compute energy in form of Boltzmann-like density function:
                                1
                p(F, V, R)  =  --- * exp {-beta * E(F, V, R)}
                                Z
        
        Parameters:
        -----------
        beta, float:
        q, float:
        """
        # TODO, value = exp{-1 / 600} ????
        if beta is None:
            beta = (1 / q) * 600 
        return exp(-1 * beta * q) 


    def __compute(self):
        """Run Morkov-chain Monte-Carlo sampling to find optimal solution 
        """
        for i in tqdm(range(self.num_iters)):
            ID = random.randint(0, self.tubes_manager.num_tubes - 1)
            flag = random.random() < 0.75

            if flag:
                temporary_time_start = random.randint(0, self.tubes_manager.synopsis_begin_frame + self.tubes_manager.synopsis_video_length)
                temporary_seg_length = maxsize
                temporary_ratio_size = 0
                temporary_seg_idx = 0

            else:
                temporary_time_start = maxsize
                temporary_seg_idx = random.randint(0, self.tubes_manager.tubes[ID].num_segments - 1)

                if i % 4 == 0:
                    low_ratio_size, up_ratio_size = self.cfg.SYNOPSIS.TUBE.RATIO_SIZE_RANGE
                    temporary_ratio_size = random.uniform(low_ratio_size, up_ratio_size)
                    temporary_seg_length = maxsize

                else:
                    low_v_ratio, up_v_ratio = self.cfg.SYNOPSIS.TUBE.RATIO_SPEED_RANGE
                    unit_segment_length = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH
                    temporary_seg_length = random.randint(int(round(unit_segment_length/up_v_ratio)),
                                                          int(round(unit_segment_length/low_v_ratio)))
                    temporary_ratio_size = 0

            tmp_q = self.tubes_manager.get_energy(ID, 
                                                  temporary_time_start,
                                                  temporary_seg_idx,
                                                  temporary_seg_length,
                                                  temporary_ratio_size)
            # print('q value: ', tmp_q)
            tmp_q_Boltzmann = self.__to_Boltzmann_like_value(tmp_q, beta=self.beta)
            
            # TODO why need this if statement ???
            if self.best_q > tmp_q:
                self.best_q = tmp_q

                for tube_idx in range(self.tubes_manager.num_tubes):
                    self.tubes_manager.best_result[tube_idx][0] = self.tubes_manager.tubes[tube_idx].time_start
                    
                    tube_numseg = self.tubes_manager.tubes[tube_idx].num_segments
                    self.tubes_manager.best_result[tube_idx][1: tube_numseg + 1] = self.tubes_manager.tubes[tube_idx].segments_length
                    self.tubes_manager.best_result[tube_idx][1 + tube_numseg: ] = self.tubes_manager.tubes[tube_idx].ratio_size
                
                if temporary_time_start != maxsize:
                    self.tubes_manager.best_result[ID][0] = temporary_time_start 
                if temporary_seg_length != maxsize:
                    self.tubes_manager.best_result[ID][1 + temporary_seg_idx] = temporary_seg_length
                if temporary_ratio_size != 0:
                    self.tubes_manager.best_result[ID][1 + temporary_seg_idx + self.tubes_manager.tubes[tube_idx].num_segments] = temporary_ratio_size

            alpha = np.minimum(1.0, tmp_q_Boltzmann / self.q_Boltzmann)
            accept_prob = random.random()
            if alpha.item() > accept_prob:
                if self.q_Boltzmann > 0.05: # TODO Why 0.05 ???
                    self.beta = (1 / tmp_q) * 600
                    tmp_q_Boltzmann = self.__to_Boltzmann_like_value(tmp_q)
                
                self.q_Boltzmann = tmp_q_Boltzmann 
                self.tubes_manager.set(ID,
                                       temporary_time_start,
                                       temporary_seg_idx,
                                       temporary_seg_length,
                                       temporary_ratio_size)
        
        for tube_idx in range(self.tubes_manager.num_tubes):
            self.tubes_manager.best_result[tube_idx][0] = self.tubes_manager.tubes[tube_idx].time_start
            
            tube_numseg = self.tubes_manager.tubes[tube_idx].num_segments
            self.tubes_manager.best_result[tube_idx][1: 1 + tube_numseg] = self.tubes_manager.tubes[tube_idx].segments_length
            self.tubes_manager.best_result[tube_idx][1 + tube_numseg: ] = self.tubes_manager.tubes[tube_idx].ratio_size

            self.tubes_manager.tubes[tube_idx].update_ratio_v()

        self.tubes_manager.v_reset()
        self.tubes_manager.size_reset()
        self.tubes_manager.size_smoothing()
    

    def run(self):
        self.__compute()