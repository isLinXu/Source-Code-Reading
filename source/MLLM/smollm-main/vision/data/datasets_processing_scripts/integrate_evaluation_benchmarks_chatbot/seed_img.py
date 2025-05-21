"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4

Download https://huggingface.co/datasets/AILab-CVC/SEED-Bench/tree/main -> /fsx/hugo/SEED-Bench
"""


import json
import os
from io import BytesIO

import datasets
from datasets import Dataset, DatasetDict
from PIL import Image
from tqdm import tqdm


PATH_JSON_SEED = "/fsx/hugo/SEED-Bench/SEED-Bench.json"
COMMON_PATH_IMAGES = "/fsx/hugo/SEED-Bench/SEED-Bench-image"

FEATURES = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "question": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=["A", "B", "C", "D"]),
        "question_id": datasets.Value("string"),
    }
)

REPO_ID = "HuggingFaceM4/SEED_Img_Modif"


def convert_img_to_bytes(img_path, format):
    img = Image.open(img_path)
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img.close()
    return img_bytes


with open(PATH_JSON_SEED, "r") as f:
    data_seed = json.load(f)["questions"]


dict_seed = {"image": [], "question": [], "label": [], "question_id": []}

for example in tqdm(data_seed):
    if example["data_type"] != "image":
        continue
    question_id = example["question_id"]
    question = example["question"]
    all_choices = [example["choice_a"], example["choice_b"], example["choice_c"], example["choice_d"]]
    answer = example["answer"]
    prompt = ""
    prompt += f"Question: {question}\n"
    prompt += "Choices:\n"
    letters_cap = ["A", "B", "C", "D"]
    for idx, choice in enumerate(all_choices):
        letter = letters_cap[idx]
        prompt += f"{letter}. {choice}\n"
    prompt += "Answer with the letter."
    dict_seed["question"].append(prompt)
    dict_seed["label"].append(answer)
    dict_seed["question_id"].append(str(question_id))
    image = {
        "bytes": convert_img_to_bytes(img_path=os.path.join(COMMON_PATH_IMAGES, example["data_id"]), format="JPEG"),
        "path": None,
    }
    dict_seed["image"].append(image)


ds_seed = Dataset.from_dict(dict_seed, features=FEATURES)
ds_seed = DatasetDict({"test": ds_seed})

ds_seed.push_to_hub(REPO_ID)
