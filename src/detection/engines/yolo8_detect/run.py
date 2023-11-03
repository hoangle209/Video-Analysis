import numpy as np
import torch

from .models.torch_utils import det_postprocess
from .models.utils import blob, letterbox

def yolo8engine(batch, engine):
    H, W = engine.inp_info[0].shape[-2:]

    data = [letterbox(i, (H, W)) for i in batch] # bgr, ratio, dwdh 
    frames = [im for im, *_ in data]
    ratios = [ratio for _, ratio, _ in data]
    dwdhs = [np.array(dwdh * 2) for *_, dwdh in data]
    
    batch = [blob(frame[..., ::-1]) for frame in frames]
    batch = np.concatenate(batch, axis=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = torch.asarray(batch, device=device)

    raw_data = engine(batch)
    postprocess = det_postprocess(raw_data) # bboxes, scores, labels
    bboxes = [(bb.cpu().numpy() - dwdh)/ratio 
                        if bb.cpu().numpy().numel() > 0 else bb.cpu().numpy()
                        for (bb, *_), ratio, dwdh in zip(postprocess, ratios, dwdhs)]
    scores = [score.cpu().numpy() for _, score, _ in postprocess]
    labels = [label.cpu().numpy() for *_, label in postprocess]

    results = [np.concatenate([bbox, score, label], axis=-1) 
                            for (bbox, score, label) in zip(bboxes, scores, labels)]
    return results