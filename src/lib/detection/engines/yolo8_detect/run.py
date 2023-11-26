import numpy as np
import torch

from .models.torch_utils import det_postprocess
from .models.utils import blob, letterbox

def yolo8engine(batch, engine, device, classes=(0,)):
    """Run engine model
    
    Parameters:
    -----------
    batch, list[np.ndarray]
        list of input image
    engine,
        engine model
    device,
        system device
    """
    n = 8 # TODO: Modify to dynamic inference

    H, W = engine.inp_info[0].shape[-2:]

    data = [letterbox(i, (H, W)) for i in batch] # bgr, ratio, dwdh 
    frames = [bgr for bgr, _, _ in data]
    ratios = [ratio for _, ratio, _ in data]
    dwdhs = [np.array(dwdh * 2) for _, _, dwdh in data]
    
    batch = [blob(frame[..., ::-1]) for frame in frames] # shape (1, 3, H, W)
    batch = np.concatenate(batch, axis=0)
    batch = torch.asarray(batch, device=device)

    raw_data = engine(batch) # num_dets, bboxes, scores, labels, shape (n, ...)
    postprocess = [det_postprocess([[raw_data[0][i]],
                                    [raw_data[1][i]],
                                    [raw_data[2][i]],
                                    [raw_data[3][i]]
                                    ], classes) for i in range(n)] # bboxes, scores, labels

    bboxes = [(bb.cpu().numpy() - dwdh)/ratio 
                    if bb.numel() > 0 else bb.cpu().numpy()
                    for (bb, _, _), ratio, dwdh in zip(postprocess, ratios, dwdhs)]
    scores = [score.cpu().numpy() for _, score, _ in postprocess]
    labels = [label.cpu().numpy() for _, _, label in postprocess]

    results = [np.concatenate([bbox, score.reshape(-1,1), label.reshape(-1, 1)], axis=-1) 
                                 for (bbox, score, label) in zip(bboxes, scores, labels)]
    return results