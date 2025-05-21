"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4

Download https://github.com/RUCAIBox/POPE.git -> /fsx/hugo/POPE
"""

import json
import os
from io import BytesIO

import datasets
from datasets import DatasetDict
from PIL import Image
from tqdm import tqdm


FEATURES = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "context": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=["No", "Yes"]),
    }
)

PATHS_JSON_POPE = [
    "/fsx/hugo/POPE/output/coco/coco_pope_random.json",
    "/fsx/hugo/POPE/output/coco/coco_pope_popular.json",
    "/fsx/hugo/POPE/output/coco/coco_pope_adversarial.json",
]

COMMON_PATH_IMAGES = "/fsx/hugo/coco/val2014"

REPO_ID = "HuggingFaceM4/POPE_modif"


def convert_img_to_bytes(img_path, format):
    img = Image.open(img_path)
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img.close()
    return img_bytes


dict_pope = {"image": [], "context": [], "label": []}

for path_json_pope in tqdm(PATHS_JSON_POPE):
    with open(path_json_pope, "r") as f:
        for line in tqdm(f):
            example = json.loads(line)
            image = {
                "bytes": convert_img_to_bytes(
                    img_path=os.path.join(COMMON_PATH_IMAGES, example["image"]), format="JPEG"
                ),
                "path": None,
            }
            dict_pope["image"].append(image)
            dict_pope["context"].append(example["text"])
            if example["label"] == "yes":
                dict_pope["label"].append("Yes")
            elif example["label"] == "no":
                dict_pope["label"].append("No")
            else:
                raise ValueError("Unexpected label")

ds_pope = datasets.Dataset.from_dict(dict_pope, features=FEATURES)
ds_pope = DatasetDict({"test": ds_pope})

ds_pope.push_to_hub(REPO_ID)
