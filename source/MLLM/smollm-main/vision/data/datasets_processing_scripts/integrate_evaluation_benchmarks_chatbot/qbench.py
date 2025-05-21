"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4

Download https://huggingface.co/datasets/teowu/LLVisionQA-QBench -> /fsx/hugo/LLVisionQA-QBench
"""


import json
import os
from io import BytesIO

import datasets
from datasets import Dataset, DatasetDict
from PIL import Image
from tqdm import tqdm


PATHS_JSON_QBENCH = {
    "validation": "/fsx/hugo/LLVisionQA-QBench/llvisionqa_dev.json",
    "test": "/fsx/hugo/LLVisionQA-QBench/llvisionqa_test.json",
}
COMMON_PATH_IMAGES = "/fsx/hugo/LLVisionQA-QBench/images"

WRONG_LABEL = "cfghjvkljhgjffxghckjvlbknlmjghfdfgfghggdfxsdgfg"

REPO_ID = "HuggingFaceM4/QBench_Modif"


def convert_img_to_bytes(img_path, format):
    img = Image.open(img_path)
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img.close()
    return img_bytes


def create_dict_qbench(split):
    """`split` is "train", "validation" or "test"."""
    dict_qbench = {"image": [], "question": [], "label": [], "tested_labels": []}
    #
    with open(PATHS_JSON_QBENCH[split], "r") as f:
        data_qbench = json.load(f)
    #
    for example in tqdm(data_qbench):
        dict_qbench["question"].append(example["question"])
        if split == "validation":
            dict_qbench["label"].append(example["correct_ans"])
        elif split == "test":
            dict_qbench["label"].append(None)
        else:
            raise ValueError("Incorrect argument `split`")
        dict_qbench["tested_labels"].append(example["candidates"])
        image = {
            "bytes": convert_img_to_bytes(
                img_path=os.path.join(COMMON_PATH_IMAGES, example["img_path"]), format="JPEG"
            ),
            "path": None,
        }
        dict_qbench["image"].append(image)
    return dict_qbench


dict_qbench_validation = create_dict_qbench(split="validation")
dict_qbench_test = create_dict_qbench(split="test")

all_classes = [
    tested_labels for tested_labels in dict_qbench_validation["tested_labels"] + dict_qbench_test["tested_labels"]
]
all_classes = [sub_el for el in all_classes for sub_el in el]
all_classes = list(set(all_classes))
all_classes.append(WRONG_LABEL)

num_candidates_per_question = [
    len(tested_labels) for tested_labels in dict_qbench_validation["tested_labels"] + dict_qbench_test["tested_labels"]
]
max_num_candidates_per_question = max(num_candidates_per_question)

# Super hacky: we can have different tested labels for each example, but the number of tested labels has to be fix
# We can add a super noisy tested label (that will have a really low probability) to complete the list of candidates
# answers to the maximum number of candidate answer in an exemple

for idx in range(len(dict_qbench_validation["tested_labels"])):
    num_candidates_current_example = len(dict_qbench_validation["tested_labels"][idx])
    dict_qbench_validation["tested_labels"][idx] = dict_qbench_validation["tested_labels"][idx] + [WRONG_LABEL] * (
        max_num_candidates_per_question - num_candidates_current_example
    )

for idx in range(len(dict_qbench_test["tested_labels"])):
    num_candidates_current_example = len(dict_qbench_test["tested_labels"][idx])
    dict_qbench_test["tested_labels"][idx] = dict_qbench_test["tested_labels"][idx] + [WRONG_LABEL] * (
        max_num_candidates_per_question - num_candidates_current_example
    )

features = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "question": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=all_classes),
        "tested_labels": datasets.Sequence(datasets.Value("string")),
    }
)

ds_qbench_validation = Dataset.from_dict(dict_qbench_validation, features=features)
ds_qbench_test = Dataset.from_dict(dict_qbench_test, features=features)

ds_qbench = DatasetDict({"validation": ds_qbench_validation, "test": ds_qbench_test})

ds_qbench.push_to_hub(REPO_ID, private=True)
