import json
import os
from io import BytesIO

import datasets
from datasets import DatasetDict
from PIL import Image


LOCAL_PATH = "/fsx/m4/victor/raw_datasets/aokvqa/"

FEATURES = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "question_id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "choices": [datasets.Value("string")],
        "correct_choice_idx": datasets.Value("int8"),
        "direct_answers": datasets.Value("string"),
        "difficult_direct_answer": datasets.Value("bool"),
        "rationales": [datasets.Value("string")],
    }
)


def convert_img_to_bytes(img_path, format):
    img = Image.open(img_path)
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img.close()
    return img_bytes


def find_image_path(image_id):
    for folder in ["/fsx/hugo/coco/train2017/", "/fsx/hugo/coco/val2017/", "/fsx/hugo/coco/test2017/"]:
        path = folder + str(image_id).zfill(12) + ".jpg"
        if os.path.exists(path):
            return path
    raise ValueError(f"Image not found for `{image_id}`")


def init_dict():
    dict_to_fill = {
        "image": [],
        "question_id": [],
        "question": [],
        "choices": [],
        "correct_choice_idx": [],
        "direct_answers": [],
        "difficult_direct_answer": [],
        "rationales": [],
    }
    return dict_to_fill


# Train

train_dict = init_dict()

with open(f"{LOCAL_PATH}/aokvqa_v1p0_train.json") as f:
    train_data = json.load(f)
for example in train_data:
    image = {
        "bytes": convert_img_to_bytes(img_path=find_image_path(example["image_id"]), format="JPEG"),
        "path": None,
    }
    train_dict["image"].append(image)
    train_dict["question_id"].append(example["question_id"])
    train_dict["question"].append(example["question"])
    train_dict["choices"].append(example["choices"])
    train_dict["correct_choice_idx"].append(example["correct_choice_idx"])
    train_dict["direct_answers"].append(example["direct_answers"])
    train_dict["difficult_direct_answer"].append(example["difficult_direct_answer"])
    train_dict["rationales"].append(example["rationales"])

train_dataset = datasets.Dataset.from_dict(train_dict, features=FEATURES, split="train")

print("Finished building train")

# Validation
val_dict = init_dict()

with open(f"{LOCAL_PATH}/aokvqa_v1p0_val.json") as f:
    val_data = json.load(f)
for example in val_data:
    image = {
        "bytes": convert_img_to_bytes(img_path=find_image_path(example["image_id"]), format="JPEG"),
        "path": None,
    }
    val_dict["image"].append(image)
    val_dict["question_id"].append(example["question_id"])
    val_dict["question"].append(example["question"])
    val_dict["choices"].append(example["choices"])
    val_dict["correct_choice_idx"].append(example["correct_choice_idx"])
    val_dict["direct_answers"].append(example["direct_answers"])
    val_dict["difficult_direct_answer"].append(example["difficult_direct_answer"])
    val_dict["rationales"].append(example["rationales"])

val_dataset = datasets.Dataset.from_dict(val_dict, features=FEATURES, split="validation")

print("Finished building validation")

# Test

test_dict = init_dict()

with open(f"{LOCAL_PATH}/aokvqa_v1p0_test.json") as f:
    test_data = json.load(f)
for example in test_data:
    image = {
        "bytes": convert_img_to_bytes(img_path=find_image_path(example["image_id"]), format="JPEG"),
        "path": None,
    }
    test_dict["image"].append(image)
    test_dict["question_id"].append(example["question_id"])
    test_dict["question"].append(example["question"])
    test_dict["choices"].append(example["choices"])
    test_dict["correct_choice_idx"].append(None)
    test_dict["direct_answers"].append(None)
    test_dict["difficult_direct_answer"].append(example["difficult_direct_answer"])
    test_dict["rationales"].append(None)

test_dataset = datasets.Dataset.from_dict(test_dict, features=FEATURES, split="test")

print("Finished building test")

dataset = DatasetDict(
    {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    }
)
dataset.push_to_hub("HuggingFaceM4/A-OKVQA")
