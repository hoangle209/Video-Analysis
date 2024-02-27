# TODO write setup file for MOTRv2
import sys
sys.path.insert(1, "../Video-Analysis")

import cv2
from copy import deepcopy
import numpy as np
import yaml
import dacite

import torch
import torchvision.transforms.functional as F

from VideoAnalysis.track.MOTRv2.models import build_model
from VideoAnalysis.track.MOTRv2.models.structures import Instances
from VideoAnalysis.track.MOTRv2.util.tool import load_model
from .motrv2_config import MOTRv2Config


class RuntimeTrackerBase:
    def __init__(self, score_thresh=0.6, filter_score_thresh=0.5, miss_tolerance=10):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0
    
    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores >= self.score_thresh] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores >= self.score_thresh)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_thresh)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


class MOTRv2:
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        with open(f"{self.cfg.TRACK.MODEL_CONFIG}", "r") as stream:
            try:
                motrv2_cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)     

        MOTRv2Cfg = dacite.from_dict(data_class=MOTRv2Config, data=motrv2_cfg)

        self.detr, _, _ = build_model(MOTRv2Cfg) # detr
        self.detr.track_embed.score_thr = MOTRv2Cfg.update_score_threshold
        self.detr.track_base = RuntimeTrackerBase(MOTRv2Cfg.score_threshold, MOTRv2Cfg.score_threshold, MOTRv2Cfg.miss_tolerance)
        self.detr = load_model(self.detr, MOTRv2Cfg.resume)
        self.detr = self.detr.to(torch.device(cfg.TRACK.DEVICES))

        self.track_instances = None
        self.total_dts = 0
        self.total_occlusion_dts = 0

        # TODO flexible
        self.input_img_height = 800
        self.input_img_width = 1536
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    
    def reset(self):
        self.track_instances = None
        self.total_dts = 0
        self.total_occlusion_dts = 0


    @staticmethod
    def filter_dt_by_score(dt_instances: Instances, prob_threshold: float) -> Instances:
        keep = dt_instances.scores > prob_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]


    @staticmethod
    def filter_dt_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        wh = dt_instances.boxes[:, 2:4] - dt_instances.boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > area_threshold
        return dt_instances[keep]
    

    def preprocess(self, image, proposals): # input is bgr image
        """
        Parameters:
        -----------
            image, ndarray
                image in bgr color mode
            proposals, ndarray, (-1, 5)
                xyxys format
        """
        cur_img = image[..., ::-1]
        im_h, im_w = cur_img.shape[:2]

        proposals_ = []
        for proposal in proposals:
            x1, y1, x2, y2, conf = proposal
            proposals_.append([(x1 + x2) / 2 / im_h,
                               (y1 + y2) / 2 / im_w,
                               (x2 - x1) / im_w, 
                               (y2 - y1) / im_h, 
                                conf])
        
        scale = self.input_img_height / min(im_h, im_w)
        if max(im_h, im_w) * scale > self.input_img_width:
            scale = self.img_width / max(im_h, im_w)
        
        target_h = int(im_h * scale)
        target_w = int(im_w * scale)
        img = cv2.resize(cur_img, (target_w, target_h), interpolation = cv2.INTER_AREA)
        img = F.normalize(F.to_tensor(img), self.mean, self.std)
        img = img.unsqueeze(0)

        return img, image, torch.as_tensor(proposals_).reshape(-1, 5)
    

    def detect(self, image, proposals, prob_threshold=0.6, area_threshold=100):
        cur_img, ori_img , proposals_ = self.preprocess(image, proposals)
        cur_img, proposals_ = cur_img.to(torch.device(self.cfg.TRACK.DEVICES)), proposals_.to(torch.device(self.cfg.TRACK.DEVICES))

        if self.track_instances is not None:
            self.track_instances.remove("boxes")
            self.track_instances.remove("labels")
        
        seq_h, seq_w, _ = ori_img.shape
        res = self.detr.inference_single_image(cur_img, (seq_h, seq_w), self.track_instances, proposals_)
        self.track_instances = res['track_instances']

        dt_instances = deepcopy(self.track_instances)

        # filter det instances by score.
        dt_instances = self.filter_dt_by_score(dt_instances, prob_threshold)
        dt_instances = self.filter_dt_by_area(dt_instances, area_threshold)

        total_dts += len(dt_instances)
        bbox_xyxy = dt_instances.boxes.tolist()
        confs = conf = dt_instances.scores.tolist()
        identities = dt_instances.obj_idxes.tolist()

        tracklets = []
        for xyxy, conf, identity in zip(bbox_xyxy, confs, identities):
            x1, y1, x2, y2 = xyxy
            tracklet = [x1, y1, x2, y2, conf, identity]
            tracklets.append(tracklet)
        
        return np.array(tracklets)
        




