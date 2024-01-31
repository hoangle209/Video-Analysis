"""Partial of this code is modified from:
        https://github.com/brjathu/PHALP/tree/master/phalp
"""
from ..detection import detector_factory
from ..action.models import HMR2018Predictor
from ..action.models.hmr.utils_dataset import process_image
from ..synopsis import SynopsisVideoProducer

from ..utils import get_pylogger
LOGGER = get_pylogger(__name__)

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn


class VideoAnalyzer(nn.Module):
    def __init__(self, cfg, task=None):
        super().__init__()
        self.cfg = cfg
        
        # self.setup_HMR()
        # self.setup_pose_predictor()
        self.setup_synopsis_video_producer()


    ################# SETUP MODEL #################
    def setup_detector(self):
        LOGGER.info(f'Setting up {self.cfg.DETECT.MODEL} detection model. Using {self.cfg.DETECT.WEIGHT}...')
        self.detector = detector_factory[self.cfg.DETECT.MODEL](self.cfg)


    def setup_HMR(self):
        LOGGER.info(f"Setting HMR model. Loading {self.cfg.hmr.hmar_path}...")
        self.HMR = HMR2018Predictor(self.cfg)
        self.HMR.load_weights(self.cfg.hmr.hmar_path)


    def setup_pose_predictor(self):
        if self.cfg.POSE.MDOEL == 'mmpose':
            LOGGER.info(f'Setting {self.cfg.POSE.MODEL} - {self.cfg.POSE.MODEL_CONFIG} pose predictor. \
                        USing {self.cfg.POSE.WEIGHT}...')
            from mmpose.apis import init_model, inference_bottomup
            self.pose_predictor = init_model(self.cfg.POSE.MODEL_CONFIG,
                                             device=self.cfg.POSE.DEVICES,
                                             checkpoint=self.cfg.POSE.WEIGHT)
    

    def setup_tracker(self):
        LOGGER.info('Setting tracker')
        self.tracker = None

    def setup_synopsis_video_producer(self):
        LOGGER.info('Setting up Synopsis Video Producer')
        self.synopsis_video_producer = SynopsisVideoProducer(self.cfg)

    ################# GET RESULTS ################# 
    def get_bbox(self, image):
        bboxes = self.detector(image)
        return bboxes

    def get_pose(self, image):
        if self.cfg.POSE.MODEL == 'mmpose':
            pose_result = inference_bottomup(self.pose_predictor, image)
            keypoints_2D = pose_result[0].pred_instances.keypoints
            bboxes = pose_result[0].bboxes
            bbox_scores = pose_result[0].bbox_scores

            det_thresh = self.cfg.DETECT.CONF
            bboxes = bboxes[bbox_scores>det_thresh]
            bbox_scores = bbox_scores[bbox_scores>det_thresh]

            bboxes = np.concatenate([bboxes, bbox_scores[..., None]], axis=1)

        return bboxes, keypoints_2D


    def get_human_features(self, image, frame_name, cls_id, t_, gt=1, ann=None, extra_data=None):
        bboxes, keypoints_2D = self.get_pose(image)
        
        NPEOPLE = bboxes.shape[0]
        if NPEOPLE == 0:
            return []

        # TODO: Track to reID and compute ground-point here

        img_height, img_width, _  = image.shape
        new_image_size            = max(img_height, img_width)
        top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
        
        ratio = 1.0/int(new_image_size)*self.cfg.render.res
        image_list = []
        center_list = []
        scale_list = []
        selected_ids = []

        for p_ in range(NPEOPLE):
            if bboxes[p_][2]-bboxes[p_][0]<self.cfg.phalp.small_w or bboxes[p_][3]-bboxes[p_][1]<self.cfg.phalp.small_h:
                continue
            p_image, center_, scale_, center_pad, scale_pad = self.get_croped_image(image, bboxes[p_], bboxes[p_]) # PHALP return both bboxes and bboxes_pad 
                                                                                                                   # as pred instance from MaskRCNN
            image_list.append(p_image)
            center_list.append(center_pad)
            scale_list.append(scale_pad)
            selected_ids.append(p_)
        
        BS = len(image_list)
        if BS == 0: return []
        
        with torch.no_grad():
            extra_args      = {}
            hmar_out        = self.HMAR(image_list.cuda(), **extra_args)

            pred_smpl_params, pred_joints_2d, pred_joints, pred_cam  = self.HMAR.get_3d_parameters(hmar_out['pose_smpl'], hmar_out['pred_cam'],
                                                                                                   center=(np.array(center_list) + np.array([left, top]))*ratio,
                                                                                                   img_size=self.cfg.render.res,
                                                                                                   scale=np.max(np.array(scale_list), axis=1, keepdims=True)*ratio)
            pred_smpl_params = [{k:v[i].cpu().numpy() for k,v in pred_smpl_params.items()} for i in range(BS)]
            pred_cam_ = pred_cam.view(BS, -1)
            pred_cam_.contiguous()
        
        detection_data_list = []
        for i, p_ in enumerate(selected_ids):
            detection_data = {
                    "bbox"            : np.array([bboxes[p_][0], bboxes[p_][1], (bboxes[p_][2] - bboxes[p_][0]), (bboxes[p_][3] - bboxes[p_][1])]), # xmin, ymin, w, h
                    "conf"            : bboxes[p_][-1],
                    "id"              : None,
                    
                    "center"          : center_list[i],
                    "scale"           : scale_list[i],
                    "smpl"            : pred_smpl_params[i],
                    "camera"          : pred_cam_[i].cpu().numpy(),
                    "camera_bbox"     : hmar_out['pred_cam'][i].cpu().numpy(),
                    "2d_joints"       : keypoints_2D[p_],
                    
                    "size"            : [img_height, img_width],
                    "img_path"        : frame_name,
                    "img_name"        : frame_name.split('/')[-1] if isinstance(frame_name, str) else None,
                    "class_name"      : cls_id[p_],
                    "time"            : t_,

                    "ground_truth"    : gt[p_],
                    "annotations"     : ann[p_],
                    "extra_data"      : extra_data[p_] if extra_data is not None else None
            }
            detection_data_list.append(detection_data)
        
        return detection_data_list


    def get_synopsis_video(self):
        self.synopsis_video_producer.run()


    def get_croped_image(image, bbox, bbox_pad):
        center_      = np.array([(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2])
        scale_       = np.array([(bbox[2] - bbox[0]), (bbox[3] - bbox[1])])

        center_pad   = np.array([(bbox_pad[2] + bbox_pad[0])/2, (bbox_pad[3] + bbox_pad[1])/2])
        scale_pad    = np.array([(bbox_pad[2] - bbox_pad[0]), (bbox_pad[3] - bbox_pad[1])])
        image_tmp    = process_image(image, center_pad, 1.0*np.max(scale_pad))

        return image_tmp, center_, scale_, center_pad, scale_pad




