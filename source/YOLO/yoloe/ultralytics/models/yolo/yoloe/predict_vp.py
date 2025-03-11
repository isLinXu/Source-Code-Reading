from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.models.yolo.segment import SegmentationPredictor
from ultralytics.data.augment import LetterBox, LoadVisualPrompt
from ultralytics.utils.instance import Instances
import numpy as np
import torch
from copy import deepcopy

from ultralytics.utils.torch_utils import select_device

class YOLOEVPPredictorMixin:
    def setup_model(self, model, verbose=True):
        """Initialize YOLO model with given parameters and set it to evaluation mode."""
        device = select_device(self.args.device, verbose=verbose)
        self.model = model.to(device)

        self.device = device  # update device
        self.model.fp16 = False
        self.args.half = False
        self.model.eval()
        
        self.done_warmup = True
        self.return_vpe = False
        
    def set_return_vpe(self, return_vpe):
        self.return_vpe = return_vpe
    
    def set_prompts(self, prompts):
        self.prompts = deepcopy(prompts)
        
    def pre_transform(self, im):
        letterbox = LetterBox(
            self.imgsz,
            auto=False,
            stride=int(self.model.stride[-1].item()),
        )
        assert(len(im) == 1)
        
        if "bboxes" in self.prompts and len(self.prompts["bboxes"]) > 0:
            labels = dict(
                img=im[0],
                instances=Instances(bboxes=self.prompts["bboxes"], 
                                    segments=np.zeros((0, 1000, 2), dtype=np.float32), 
                                    bbox_format="xyxy", normalized=False),
                cls=torch.zeros((len(self.prompts["bboxes"]), 1))
            )
            
            labels = letterbox(labels)
            
            instances = labels.pop("instances")
            h, w = labels["img"].shape[:2]
            instances.normalize(w, h)
            instances.convert_bbox(format="xywh")
            labels["bboxes"] = torch.from_numpy(instances.bboxes)
        else:
            masks = self.prompts["masks"]

            img = letterbox(image=im[0])
            resized_masks = []
            for i in range(len(masks)):
                resized_masks.append(letterbox(image=masks[i]))
            masks = np.stack(resized_masks)
            masks[masks == 114] = 0

            labels = dict(
                img=img,
                masks=masks,
                cls=torch.zeros((len(masks), 1))
            )

        labels["img"] = labels["img"].transpose(2, 0, 1)
        
        load_vp = LoadVisualPrompt(nc=1, augment=False)
        labels = load_vp(labels)
        
        self.prompts = labels["visuals"].unsqueeze(0).to(self.device)
        self.model.model[-1].nc = self.prompts.shape[1]
        self.model.names = [f"object{i}" for i in range(self.prompts.shape[1])]
        
        return [labels["img"].transpose(1, 2, 0)]

    def inference(self, im, *args, **kwargs):
        if self.return_vpe:
            self.vpe = self.model.get_visual_pe(im, visual=self.prompts)
        return super().inference(im, vpe=self.prompts, *args, **kwargs)


class YOLOEVPDetectPredictor(YOLOEVPPredictorMixin, DetectionPredictor):
    pass

class YOLOEVPSegPredictor(YOLOEVPPredictorMixin, SegmentationPredictor):
    pass