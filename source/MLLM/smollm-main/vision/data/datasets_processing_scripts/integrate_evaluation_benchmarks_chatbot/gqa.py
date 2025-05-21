import os
import random

import datasets
from datasets import DatasetDict, load_dataset

from datasets_processing_scripts.build_concatenation_datasets_sft.build_ds_sft import (
    PROMPTS_ANSWER_SHORTLY,
    convert_img_to_bytes,
)


NUM_PROC = 96

FEATURES = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "question": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "question_id": datasets.Value("int64"),
    }
)


def map_transform_gqa(example):
    path_image = os.path.join("/fsx/hugo/gqa/images", os.path.basename(example["image_id"]))
    image_bytes = convert_img_to_bytes(img_path=path_image, format="JPEG")
    question = example["question"] + random.choice(PROMPTS_ANSWER_SHORTLY)
    example["image"] = {"path": None, "bytes": image_bytes}
    example["question"] = question
    example["answers"] = [example["label"]]
    return example


def load_gqa(split):
    ds_gqa = load_dataset("Graphcore/gqa", split=split)
    columns_to_keep = ["image", "question", "answers", "question_id"]
    columns_to_remove = [c_n for c_n in ds_gqa.column_names if c_n not in columns_to_keep]
    ds_gqa = ds_gqa.map(map_transform_gqa, remove_columns=columns_to_remove, features=FEATURES, num_proc=NUM_PROC)
    return ds_gqa


ds_gqa_all_splits = DatasetDict(
    {"train": load_gqa("train"), "validation": load_gqa("validation"), "test": load_gqa("test")}
)

ds_gqa_all_splits.push_to_hub("HuggingFaceM4/GQA", private=True)
