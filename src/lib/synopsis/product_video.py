import cv2 as cv
import os

from .mcmc import MCMC

class ProductVideo:
    def __init__(self, cfg):
        self.cfg = cfg
        self.__mcmc = MCMC(cfg)
        self.video_frames = [None for _ in range(70*cfg.SYNOPSIS.MAX_LENGTH)]
    

    def __xywhr_to_xywh(self, rect, ratio):
        """
        Parameters:
        -----------
        rect, ndarray | list:
            xmin, ymin, w, h
        ratio, float:
            scaling ratio
        """
        xmin, ymin, w, h = rect
        
        xmin = xmin + w/2 - w/2*ratio
        ymin = ymin + h/2 - h/2*ratio
        w = w*ratio
        h = h*ratio

        return xmin, ymin, w, h
    
    
    def __add_box(self, 
                  ID, 
                  src_frame_idx,
                  box, 
                  ratio,
                  synopsis_frame_idx):
        """Add object box to frame

        Parameters:
        -----------
        ID, int:
            object ID
        src_frame_idx, int:
            frame index in source video, counted from 0
        box, list | ndarray:
            object's bounding box in source frame, xmin ymin w h
        ratio, float:
            ratio size
        synopsis_frame_idx, float:
            frame index of synopsis video that object will appear
        """
        xmin, ymin, w, h = box # object's bounding box in source frame

        src_ts = self.__mcmc.tubes_manager.tubes[ID].src_time_start
        src_idx = src_ts + src_frame_idx

        # TODO reading frame
        path = 'src\\marker\\video4'
        mask_path = os.path.join(path, str(ID), f'{src_idx}.png')
        src_frame_path = os.path.join(path, 'outputs', 'input', f'{src_idx}.png') # source frame

        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        src_frame = cv.imread(src_frame_path)
        H, W = src_frame.shape[:2]

        mask = cv.resize(mask, (int(W*ratio), int(H*ratio)), interpolation=cv.INTER_AREA)
        src_frame = cv.resize(src_frame, (int(W*ratio), int(H*ratio)), interpolation=cv.INTER_AREA)

        frame_with_object = cv.bitwise_and(src_frame, src_frame, mask=mask) # remove background

        obj = frame_with_object[int(ymin*ratio) : int(ymin*ratio) + int(h*ratio), 
                                int(xmin*ratio) : int(xmin*ratio) + int(w*ratio), :]

        # xmin, ymin of obj in synopsis frame
        xmin_syn, ymin_syn, w, h = list(map(int, self.__xywhr_to_xywh(box, ratio))) 
        self.video_frames[synopsis_frame_idx][ymin_syn : ymin_syn + h, 
                                              xmin_syn : xmin_syn + w, :][obj > 0] = obj[obj > 0]


    def __add_tube(self, ID):
        """Add tube to synopsis video

        Parameters:
        -----------
        ID, int:
            object ID
        """
        unit_segment_length = self.cfg.SYNOPSIS.TUBE.UNIT_SEGMENT_LENGTH
        synopsis_frame_idx = int(self.__mcmc.tubes_manager.best_result[ID][0]) # tube synopsis starting frame

        for seg_idx in range(self.__mcmc.tubes_manager.tubes[ID].num_segments):
            seg_length = int(self.__mcmc.tubes_manager.best_result[ID][1 + seg_idx])
            ratio = self.__mcmc.tubes_manager.best_result[ID][1 + self.__mcmc.tubes_manager.tubes[ID].num_segments + seg_idx]

            for frame_idx_in_seg in range(seg_length):
                if self.video_frames[synopsis_frame_idx] is None: # TODO, why list index is out of range
                    self.video_frames[synopsis_frame_idx] = cv.imread('src\\marker\\video4\\0.png')

                # source frame index of tube with ID (always counted from 0)
                src_frame_idx = seg_idx*unit_segment_length + \
                                self.__mcmc.tubes_manager.tubes[ID].v_translate[seg_length, frame_idx_in_seg]
                if src_frame_idx < 0:
                    src_frame_idx = 0
                if src_frame_idx >= self.__mcmc.tubes_manager.tubes[ID].src_length: # TODO 
                    return
                
                box = self.__mcmc.tubes_manager.tubes[ID].frame_bounding_box[src_frame_idx] # bouding box of object in source frame
                self.__add_box(ID, src_frame_idx, box, ratio, synopsis_frame_idx)
                
                synopsis_frame_idx += 1 # synopsis frame that 
                                        # tube with ID will appear


    def __render(self):
        """
        """
        for ID in range(self.__mcmc.tubes_manager.num_tubes):
            self.__add_tube(ID)

            fps = self.__mcmc.tubes_manager.src_fps if self.cfg.SYNOPSIS.FPS == -1 \
                                                    else self.cfg.SYNOPSIS.FPS
            
        background = cv.imread('src\\marker\\video4\\0.png')
        (H, W) = background.shape[:2] # TODO read background image shape(W, H)

        writer = cv.VideoWriter('src\\marker\\video4\\filename.avi',
                                cv.VideoWriter_fourcc(*'MJPG'),
                                fps,
                                (W, H))
        
        for frame in self.video_frames[:self.cfg.SYNOPSIS.MAX_LENGTH]:
            writer.write(frame)
        writer.release()
        

    def run(self):
        self.__mcmc.run()
        print("---------------- Done optimizing process !!!")
        self.__render()
        print("---------------- Finish processing video !!!")