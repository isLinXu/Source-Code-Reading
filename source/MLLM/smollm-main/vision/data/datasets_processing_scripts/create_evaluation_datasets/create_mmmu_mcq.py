"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4
"""


import ast

import datasets
from datasets import DatasetDict, load_dataset


NAME_DS = "HuggingFaceM4/MMMU"
ALL_SPLITS = ["dev", "validation", "test"]

NUM_PROC = 96

POSSIBLE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]

IMAGE_NUMBERS = ["1", "2", "3", "4", "5", "6", "7"]

FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=POSSIBLE_LABELS),
        "images": datasets.Sequence(datasets.Image(decode=True)),
    }
)

NAME_DS_PUSH_HUB = "HuggingFaceM4/MMMU_MCQ"


def filter_func_mmmu_mcq_ds(example):
    if example["question_type"] == "open":
        return False
    elif example["question_type"] == "multiple-choice":
        return True
    else:
        raise ValueError("Different `question_type`")


def format_value_to_list(value):
    try:
        evaluated_value = ast.literal_eval(value)
        if isinstance(evaluated_value, list):
            return evaluated_value
        else:
            return [value]
    except (SyntaxError, ValueError):
        return [value]


def map_func_transform_mmmu_mcq_ds(example):
    example["images"] = [
        example[f"image_{image_number}"]
        for image_number in IMAGE_NUMBERS
        if example[f"image_{image_number}"] is not None
    ]
    image_string = ""
    for idx_image in range(len(example["images"])):
        image_string += f"<image {IMAGE_NUMBERS[idx_image]}>:<image>\n"
    image_string = image_string.strip()
    #
    choices = format_value_to_list(example["options"])
    question = example["question"]
    question = f"Question: {question}\n"
    question += "Choices:\n"
    for idx_choice, choice in enumerate(choices):
        letter_choice = POSSIBLE_LABELS[idx_choice]
        question += f"{letter_choice}. {choice}\n"
    question += "Answer with the letter."
    question = f"{image_string}\n{question}"
    example["question"] = question
    #
    if example["answer"] == "?":  # split "test"
        example["label"] = None
    else:
        example["label"] = example["answer"]
    return example


def prepare_mmmu_mcq(split):
    ds = load_dataset(NAME_DS, split=split)
    ds = ds.filter(filter_func_mmmu_mcq_ds, num_proc=NUM_PROC)
    columns_to_remove = [c_n for c_n in ds.column_names if c_n not in list(FEATURES.keys())]
    ds = ds.map(map_func_transform_mmmu_mcq_ds, remove_columns=columns_to_remove, features=FEATURES, num_proc=NUM_PROC)
    return ds


ds_all_splits = DatasetDict({split: prepare_mmmu_mcq(split=split) for split in ALL_SPLITS})
ds_all_splits.push_to_hub(NAME_DS_PUSH_HUB, private=True)

# Cache dataset
test_loading = load_dataset(NAME_DS_PUSH_HUB)
