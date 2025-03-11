from ultralytics import YOLOE
import numpy as np
from ultralytics.models.yolo.yoloe.predict_vp import YOLOEVPSegPredictor

model = YOLOE("pretrain/yoloe-v8l-seg.pt")

# Handcrafted shape can also be passed, please refer to app.py
visuals = dict(
    bboxes=np.array(
        [
            [221.52, 405.8, 344.98, 857.54]
        ]
    )
)

source_image = 'ultralytics/assets/bus.jpg'

model.predict(source_image, save=True, prompts=visuals, predictor=YOLOEVPSegPredictor)

# Prompts in different images can be passed
# Please set a smaller conf for cross-image prompts
# model.predictor = None  # remove VPPredictor
target_image = 'ultralytics/assets/zidane.jpg'
model.predict(source_image, prompts=visuals, predictor=YOLOEVPSegPredictor, return_vpe=True)
model.set_classes(["object0"], model.predictor.vpe)
model.predictor = None  # remove VPPredictor
model.predict(target_image, save=True)