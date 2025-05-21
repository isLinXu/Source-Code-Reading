from copy import deepcopy

import datasets
from datasets import DatasetDict, load_dataset


DS_NAME = "derek-thomas/ScienceQA"

NUM_PROC = 48

REPO_ID = "HuggingFaceM4/ScienceQAImg_Modif_NoMCQ"

WRONG_LABEL = "cfghjvkljhgjffxghckjvlbknlmjghfdfgfghggdfxsdgfg"


def find_max_num_choices_and_all_tested_labels(all_splits):
    max_num_choices = []
    all_tested_labels = []
    for split in all_splits:
        ds_scienceqa = load_dataset(DS_NAME, split=split)
        all_choices = ds_scienceqa["choices"]
        max_num_choices.append(max([len(choices) for choices in all_choices]))
        all_tested_labels.extend([choice for choices in all_choices for choice in choices] + [WRONG_LABEL])
    all_tested_labels = list(set(all_tested_labels))
    max_num_choices = max(max_num_choices)
    return max_num_choices, all_tested_labels


def create_ds_scienceqa(split, max_num_choices, all_tested_labels):
    """`split` is "train", "validation" or "test"."""
    ds_scienceqa = load_dataset(DS_NAME, split=split)

    #
    def map_transform_scienceqa(example):
        question = example["question"]
        lecture = example["lecture"]
        hint = example["hint"]
        context = ""
        if lecture != "":
            context += f"Lecture: {lecture}\n"
        context += f"Question: {question}\n"
        if hint != "":
            context += f"Hint: {hint}\n"
        example["context"] = context.strip()
        #
        all_choices = example["choices"]
        all_choices = all_choices + (max_num_choices - len(all_choices)) * [WRONG_LABEL]
        example["label"] = all_choices[example["answer"]]
        example["tested_labels"] = all_choices
        return example

    #
    def filter_no_image_scienceqa(example):
        if example["image"] is None:
            return False
        return True

    #
    new_features = deepcopy(ds_scienceqa.features)
    new_features["image"] = datasets.Image(decode=True)
    new_features["context"] = datasets.Value("string")
    new_features["label"] = datasets.features.ClassLabel(names=all_tested_labels)
    new_features["tested_labels"] = datasets.Sequence(datasets.Value("string"))
    #
    ds_scienceqa = ds_scienceqa.map(map_transform_scienceqa, features=new_features, num_proc=NUM_PROC)
    #
    columns_to_keep = ["image", "context", "label", "tested_labels"]
    ds_scienceqa = ds_scienceqa.remove_columns(
        column_names=[c_n for c_n in ds_scienceqa.column_names if c_n not in columns_to_keep]
    )
    #
    ds_scienceqa = ds_scienceqa.filter(filter_no_image_scienceqa, num_proc=NUM_PROC)
    return ds_scienceqa


all_splits = ["train", "validation", "test"]
max_num_choices, all_tested_labels = find_max_num_choices_and_all_tested_labels(all_splits=all_splits)

ds_scienceqa_train = create_ds_scienceqa(
    split="train", max_num_choices=max_num_choices, all_tested_labels=all_tested_labels
)
ds_scienceqa_validation = create_ds_scienceqa(
    split="validation", max_num_choices=max_num_choices, all_tested_labels=all_tested_labels
)
ds_scienceqa_test = create_ds_scienceqa(
    split="test", max_num_choices=max_num_choices, all_tested_labels=all_tested_labels
)
ds_scienceqa_all_splits = DatasetDict(
    {"train": ds_scienceqa_train, "validation": ds_scienceqa_validation, "test": ds_scienceqa_test}
)

ds_scienceqa_all_splits.push_to_hub(REPO_ID)
