"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4

Download https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip -> /fsx/hugo/MMVET
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
        "question": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "question_id": datasets.Value("string"),
    }
)

PATH_JSON_MMVET = "/fsx/hugo/MMVET/mm-vet/mm-vet.json"

COMMON_PATH_IMAGES = "/fsx/hugo/MMVET/mm-vet/images"

REPO_ID = "HuggingFaceM4/MM_VET_modif"


def convert_img_to_bytes(img_path, format):
    img = Image.open(img_path)
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img.close()
    return img_bytes


dict_mmvet = {"image": [], "question": [], "answer": [], "question_id": []}

with open(PATH_JSON_MMVET, "r") as f:
    data_mmvet = json.load(f)
    for question_id, example in tqdm(data_mmvet.items()):
        image = {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join(COMMON_PATH_IMAGES, example["imagename"]), format="png"
            ),
            "path": None,
        }
        dict_mmvet["image"].append(image)
        dict_mmvet["question"].append(example["question"])
        dict_mmvet["answer"].append(example["answer"])
        dict_mmvet["question_id"].append(question_id)

ds_mmvet = datasets.Dataset.from_dict(dict_mmvet, features=FEATURES)
ds_mmvet = DatasetDict({"test": ds_mmvet})

ds_mmvet.push_to_hub(REPO_ID)
