import numpy as np
import math
from multiprocessing import Pool


class BaseInference:
    def normal_infer(self, batch, det_function, **kwargs):
        """Normal model inference 

        Parameters:
        -----------
        batch, b x [(H, W, 3)]:
            list of input image
        
        det_function, 
            a callable detection function 
            parameters:
            -----------
            batch, b x [(H, W, 3)]:
                list of input image
            **kwargs
            returns: list of detection for each batch
        
        **kwargs:
            arguments option for detection function 
        """
        return det_function(batch, **kwargs)
    

    def grid_infer(self, 
                   batch, 
                   grid_sz,
                   det_function, 
                   **kwargs):
        """Grid inference 
        Dividing input image into patches -> making detection on each patch -> remap 

        Parameters:
        -----------
        batch, b x [(H, W, 3)]:
            list of input image
        
        grid_sz, int or tuple
            grid size 

        det_function, 
            a callable detection function  
            batch, b x [(H, W, 3)]:
                list of input image
            **kwargs
            returns: list of detection for each batch
        
        **kwargs:
            arguments option for detection function
            
        Returns:
        -----------
        The list containing the detection result of each image in batch
        """
        batch_shape = [img.shape[:2] for img in batch]
        batch_shape = np.array(batch_shape).reshape(-1, 2) # shape (N, 2)

        if isinstance(grid_sz, int):
            grid_sz = (grid_sz, grid_sz)
        grid_sz = np.array(grid_sz).astype(np.int32) # shape (1, 2)

        num_grids = np.ceil(batch_shape/grid_sz).astype(np.int32) # number of grids in h- and w- dims 
                                                                  # shape (N, 2)
        total_num_grid_each_img = num_grids[:, 0]*num_grids[:, 1] # total number of girds of each image
        cumsum_num_grid = np.cumsum(total_num_grid_each_img) # cummulative number of grids from first image in batch
        padded_shapes = [(int(num_grid[0]*grid_sz[0]), 
                          int(num_grid[1]*grid_sz[1]), 
                          3) for num_grid in num_grids] # padded shape for each image in batch

        patches=[]
        for idx, img in enumerate(batch):
            padded_frame = np.zeros(shape=padded_shapes[idx])
            h, w = img.shape[:2]
            padded_frame[:h, :w] = img
            num_grid = num_grids[idx] # the number of grids of image at index idx
            for i in range(num_grid[0]):
                for j in range(num_grid[1]):
                    patches.append(padded_frame[int(i*grid_sz[0]): int((i+1)*grid_sz[0]),
                                                int(j*grid_sz[1]): int((j+1)*grid_sz[1]), ])
        
        remap = []
        results = det_function(patches, **kwargs)
        for idx, cs_num_grid in enumerate(cumsum_num_grid):
            # the detection result of one image
            if idx == 0:
                result = results[:cs_num_grid]
            else:
                result = results[cumsum_num_grid[idx-1]: cumsum_num_grid[idx]]

            # re-map to the original coor
            r = []
            for idx_result, pred in enumerate(result):
                num_grid = num_grids[idx]
                j, i = idx_result % num_grid[1], idx_result // num_grid[1]
                pred[:, [0, 2]] += j*grid_sz[1]
                pred[:, [1, 3]] += i*grid_sz[0]
                r += pred.tolist()
            remap.append(r)
        return remap
    

    def sliding_window_infer(self, 
                             batch, 
                             window_sz, 
                             overlap_ratio, 
                             overlap_area_thresh,
                             det_function,
                             **kwargs):
        """Sliding window model inference 
        Sliding window through image -> detect in each window -> remap

        Parameters:
        -----------
        batch, b x [(H, W, 3)]:
            list of input image

        window_sz, int or tuple:
            window size
        
        overlap_ratio, float:
            must be in-range (0, 1). Window overlap ratio
        
        overlap_area_thresh, float:
            must be in-range (0, 1). Overlap area thresh to do NMS

        det_function, 
            a callable detection function 
            parameters:
            -----------
            batch, b x [(H, W, 3)]:
                list of input image
            **kwargs
            returns: list of detection for each batch 
        
        **kwargs:
            arguments option for detection function 

        Returns:
        -----------
        The list containing the detection result of each image in batch
        """
        batch_shape = [img.shape[:2] for img in batch]
        batch_shape = np.array(batch_shape).reshape(-1, 2) # shape (N, 2)
        
        if isinstance(window_sz, int):
            window_sz = (window_sz, window_sz)
        window_sz = np.array(window_sz).astype(np.int32)

        step_size = [int((1-overlap_ratio) * size) for size in window_sz] # non-overlap size in h- and w- dim
                                                                          # shape (2,)
        num_grid_h = (batch_shape[:, 0] - window_sz[0]) / step_size[0] + 1 # shape (N, )
        num_grid_w = (batch_shape[:, 1] - window_sz[1]) / step_size[1] + 1
        left_out_h = batch_shape[:, 0] - window_sz[0] - np.int32(num_grid_h-1)*step_size[0]
        left_out_w = batch_shape[:, 1] - window_sz[1] - np.int32(num_grid_w-1)*step_size[1]

        total_num_grid_each_batch = np.int32(np.ceil(num_grid_h)*np.ceil(num_grid_w)) # shape (N, )
        cumsum_num_grid = np.cumsum(total_num_grid_each_batch) # cummulative number of grids from first image
        padded_shapes = [(int(bz[0] + step_size[0] - left_out_h[i]), 
                          int(bz[1] + step_size[1] - left_out_w[i]), 
                          3) for i, bz in enumerate(batch_shape)]
        patches=[]
        for idx, img in enumerate(batch):
            padded_frame = np.zeros(shape=padded_shapes[idx])
            h, w = img.shape[:2]
            padded_frame[:h, :w] = img
            for i in range(int(math.ceil(num_grid_h[idx]))):
                for j in range(int(math.ceil(num_grid_w[idx]))):
                    patches.append(padded_frame[i*step_size[0]: i*step_size[0] + window_sz[0],
                                                j*step_size[1]: j*step_size[1] + window_sz[1], ])
        
        remap = []
        results = det_function(patches, **kwargs)
        for idx, cs_num_grid in enumerate(cumsum_num_grid):
            if idx == 0:
                result = results[:cs_num_grid]
            else:
                result = results[cumsum_num_grid[idx-1]: cumsum_num_grid[idx]]
            
            # re-map to the original coor
            r=[]
            for idx_result, pred in enumerate(result):
                j, i = idx_result % (int(num_grid_w[idx]+1)), idx_result // (int(num_grid_w[idx]+1))
                pred[:, [0, 2]] += j*step_size[1]
                pred[:, [1, 3]] += i*step_size[0]
                r += pred.tolist()
            remap.append(r)

        assert len(remap) == len(batch)

        NMS_remap = []
        for _r in remap:
            NMS_remap.append(area_NMS(_r, overlap_area_thresh))

        # TODO check why using Pool take longer runtime
        # with Pool() as p: 
        #     NMS_remap = p.starmap(area_NMS, zip(remap, 
        #                                         [overlap_area_thresh for _ in range(len(remap))]))
        return NMS_remap


def area_NMS(preds, overlap_area_thresh=0.7):
    '''Non-max supression redundant bounding-boxes 
    A box that lies inside or nearly inside the other boxes  
    (e.g. area overlapping > t%) is removed  
    '''
    preds = list(sorted(preds, key=lambda x: (x[2]-x[0])*(x[3]-x[1]), reverse=True))
    preds = np.array(preds).reshape(-1, 1, 6)

    keep = preds[0:1]
    for idx, bb in enumerate(preds[1:], start=1):
        bb = np.array(bb).reshape(-1, 1, 6).repeat(keep.shape[0], axis=0)
        bbes = np.concatenate([keep, bb], axis=1) # shape N x 2 x 4
        
        # calculate IOU of current box with all boxes in keep 
        xmin = np.max(bbes[..., 0], axis=1) # shape N,
        ymin = np.max(bbes[..., 1], axis=1)
        xmax = np.min(bbes[..., 2], axis=1)
        ymax = np.min(bbes[..., 3], axis=1)
        request_bb = np.concatenate([xmin[:, None],
                                     ymin[:, None],
                                     xmax[:, None],
                                     ymax[:, None]], axis=1) #shape N x 4
        check = (xmin < xmax) & (ymin < ymax)
        request_bb = request_bb[check] # Keep valid IOU values
        if request_bb.shape[0]==0:
            keep = np.concatenate([keep, preds[idx:idx+1]], axis=0)
            continue

        request_side = request_bb[..., [2, 3]] - request_bb[..., [0, 1]]
        request_area = request_side[:, 0]*request_side[:, 1]
        bb_area = (bb[0, 0, 2]-bb[0, 0, 0])*(bb[0, 0, 3]-bb[0, 0, 1])
        ratio = request_area / bb_area
        check = ratio > overlap_area_thresh
        if np.any(check): 
            continue
        keep = np.concatenate([keep, preds[idx:idx+1]], axis=0)
    
    return keep[:, 0]

def iou_NMS(preds, iou_thresh=0.7):
    '''Non-max supression redundant bounding-boxes  
    '''
    preds = list(sorted(preds, key=lambda x: x[4], reverse=True)) # sorted by confidence
    preds = np.array(preds).reshape(-1, 1, 6)

    keep = preds[0:1]
    for idx, bb in enumerate(preds[1:], start=1):
        bb = np.array(bb).reshape(-1, 1, 6).repeat(keep.shape[0], axis=0)
        bbes = np.concatenate([keep, bb], axis=1) # shape N x 2 x 4
        
        # calculate IOU of current box with all boxes in keep 
        xmin = np.max(bbes[..., 0], axis=1) # shape N,
        ymin = np.max(bbes[..., 1], axis=1)
        xmax = np.min(bbes[..., 2], axis=1)
        ymax = np.min(bbes[..., 3], axis=1)
        request_bb = np.concatenate([xmin[:, None],
                                     ymin[:, None],
                                     xmax[:, None],
                                     ymax[:, None]], axis=1) #shape N x 4
        check = (xmin < xmax) & (ymin < ymax)
        request_bb = request_bb[check] # Keep valid IOU values
        if request_bb.shape[0]==0:
            keep = np.concatenate([keep, preds[idx:idx+1]], axis=0)
            continue
        
        check_keep = keep[check]
        keep_side = check_keep[..., [2, 3]] - check_keep[..., [0, 1]]
        keep_area = keep_side[:, 0, 0]*keep_side[:, 0, 1]

        intersect_side = request_bb[..., [2, 3]] - request_bb[..., [0, 1]]
        intersect_area = intersect_side[:, 0]*intersect_side[:, 1]
        bb_area = (bb[0, 0, 2]-bb[0, 0, 0])*(bb[0, 0, 3]-bb[0, 0, 1])
        ratio = intersect_area / (bb_area + keep_area - intersect_area)
        check = ratio > iou_thresh
        if np.any(check): 
            continue
        keep = np.concatenate([keep, preds[idx:idx+1]], axis=0)
    
    return keep[:, 0]