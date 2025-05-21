"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4

Download https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild -> /fsx/hugo/llava_wild
"""

import json
import os
from io import BytesIO

import datasets
from datasets import DatasetDict
from PIL import Image


FEATURES = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "question": datasets.Value("string"),
        "answer": datasets.Value("string"),
        "question_id": datasets.Value("string"),
    }
)

PATH_JSON_QUESTIONS = "/fsx/hugo/llava_wild/llava-bench-in-the-wild/questions.jsonl"
PATH_JSON_ANSWERS = "/fsx/hugo/llava_wild/llava-bench-in-the-wild/answers_gpt4.jsonl"

COMMON_PATH_IMAGES = "/fsx/hugo/llava_wild/llava-bench-in-the-wild/images"

REPO_ID = "HuggingFaceM4/LLaVA_Wild_Modif"


def convert_img_to_bytes(img_path, format):
    img = Image.open(img_path)
    img = img.convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img.close()
    return img_bytes


dict_llava_wild = {"image": [], "question": [], "answer": [], "question_id": []}

with open(PATH_JSON_QUESTIONS, "r") as f:
    for line in f:
        example = json.loads(line)
        image = {
            "bytes": convert_img_to_bytes(img_path=os.path.join(COMMON_PATH_IMAGES, example["image"]), format="JPEG"),
            "path": None,
        }
        dict_llava_wild["image"].append(image)
        dict_llava_wild["question"].append(example["text"])
        dict_llava_wild["question_id"].append(str(example["question_id"]))


with open(PATH_JSON_ANSWERS, "r") as f:
    for line in f:
        example = json.loads(line)
        dict_llava_wild["answer"].append(example["text"])

assert len(dict_llava_wild["question"]) == len(dict_llava_wild["answer"]) == len(dict_llava_wild["question_id"])

ds_llava_wild = datasets.Dataset.from_dict(dict_llava_wild, features=FEATURES)
ds_llava_wild = DatasetDict({"test": ds_llava_wild})

ds_llava_wild.push_to_hub(REPO_ID)
