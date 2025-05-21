import os
import random

import datasets
from datasets import DatasetDict, load_dataset

from datasets_processing_scripts.build_concatenation_datasets_sft.build_ds_sft import convert_img_to_bytes, prompts_vsr


NUM_PROC = 96

FEATURES = datasets.Features(
    {
        "image": datasets.Image(decode=True),
        "question": datasets.Value("string"),
        "answer": datasets.ClassLabel(names=["No.", "Yes."]),
    }
)


def map_transform_vsr(example):
    try:
        path_image = os.path.join("/fsx/hugo/coco/train2017/", os.path.basename(example["image"]))
        image_bytes = convert_img_to_bytes(img_path=path_image, format="JPEG")
        example["image"] = {"path": None, "bytes": image_bytes}
        question = random.choice(prompts_vsr).format(caption=example["caption"])
        question += "\nAnswer yes or no."
        answer = "Yes." if (example["label"] == 1) else "No."
        example["question"] = question
        example["answer"] = answer
    except Exception:
        example["image"] = None
        example["question"] = None
        example["answer"] = None
    return example


def load_vsr(split):
    ds_vsr = load_dataset("cambridgeltl/vsr_zeroshot", split=split)
    columns_to_keep = ["image", "question", "answer"]
    columns_to_remove = [c_n for c_n in ds_vsr.column_names if c_n not in columns_to_keep]
    ds_vsr = ds_vsr.map(map_transform_vsr, remove_columns=columns_to_remove, features=FEATURES, num_proc=NUM_PROC)
    ds_vsr = ds_vsr.filter(lambda example: example["image"] is not None, num_proc=NUM_PROC)
    return ds_vsr


ds_vsr_all_splits = DatasetDict(
    {"train": load_vsr("train"), "validation": load_vsr("validation"), "test": load_vsr("test")}
)

ds_vsr_all_splits.push_to_hub("HuggingFaceM4/VSR_chatbot", private=True)
