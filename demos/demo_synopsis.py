import sys
sys.path.insert(1, "..\Video-Analysis")

from VideoAnalysis.apis.video_analyzer import VideoAnalyzer
from VideoAnalysis.configs.default import _C

if __name__ == '__main__':
    cfg = _C.clone()
    va = VideoAnalyzer(cfg)
    va.get_synopsis_video()
    
    