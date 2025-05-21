"""
srun --pty --cpus-per-task=8 --partition=hopper-cpu --qos high bash -i
conda activate shared-m4
"""


import datasets
from datasets import DatasetDict, load_dataset


ORIGINAL_NAME_DS = "lmms-lab/ai2d"
ORIGINAL_SPLIT_DS = "test"

NUM_PROC = 8

POSSIBLE_LABELS = ["A", "B", "C", "D"]

FEATURES = datasets.Features(
    {
        "question": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=POSSIBLE_LABELS),
        "image": datasets.Image(decode=True),
    }
)

NAME_DS_PUSH_HUB = "HuggingFaceM4/AI2D_4"


ds_test = load_dataset(ORIGINAL_NAME_DS, split=ORIGINAL_SPLIT_DS)  # 3088 examples


def filter_ds(example):
    options = example["options"]
    if any([option.isalpha() and (len(option) == 1) for option in options]):
        return False
    return True


ds_test = ds_test.filter(filter_ds, num_proc=NUM_PROC)  # 2497 examples


def map_func_transform_ai2d_ds(example):
    example["label"] = POSSIBLE_LABELS[int(example["answer"])]
    question = example["question"].strip()
    question = f"Question: {question}\nChoices:\n"
    choices = example["options"]
    for idx_choice, choice in enumerate(choices):
        letter = POSSIBLE_LABELS[idx_choice]
        choice = choice
        question += f"{letter}. {choice}\n"
    question += "Answer with the letter."
    example["question"] = question
    return example


columns_to_remove = [c_n for c_n in ds_test.column_names if c_n not in list(FEATURES.keys())]
ds_test = ds_test.map(
    map_func_transform_ai2d_ds, remove_columns=columns_to_remove, features=FEATURES, num_proc=NUM_PROC
)
print(ds_test[0]["question"])


ds_all_splits = DatasetDict({"test": ds_test})
ds_all_splits.push_to_hub(NAME_DS_PUSH_HUB, private=True)

# Cache dataset
test_loading = load_dataset(NAME_DS_PUSH_HUB)
