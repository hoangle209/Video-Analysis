from ..config import default
from ..lib.synopsis.product_video import ProductVideo

def run():
    cfg = default._C.clone()
    producer = ProductVideo(cfg)
    producer.run()
