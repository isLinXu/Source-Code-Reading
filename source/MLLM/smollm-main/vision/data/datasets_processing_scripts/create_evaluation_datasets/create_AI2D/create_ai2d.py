"""
srun --pty --cpus-per-task=8 --partition=hopper-cpu --qos high bash -i
conda activate shared-m4
"""


import datasets
from datasets import DatasetDict, load_dataset


ORIGINAL_NAME_DS = "lmms-lab/ai2d"
ORIGINAL_SPLIT_DS = "test"

NUM_PROC = 32

POSSIBLE_LABELS = ["1", "2", "3", "4"]

FEATURES = datasets.Features(
    {
        "question": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=POSSIBLE_LABELS),
        "image": datasets.Image(decode=True),
    }
)

NAME_DS_PUSH_HUB = "HuggingFaceM4/AI2D"


def map_func_transform_ai2d_ds(example):
    example["label"] = str(int(example["answer"]) + 1)
    question = example["question"].strip()
    question = f"Question: {question}\nChoices:\n"
    choices = example["options"]
    for idx_choice, choice in enumerate(choices):
        question += f"Choice {idx_choice + 1}: {choice}\n"
    # question += "Answer with the option number."  # Commented because should be defined in the evaluation prompt
    example["question"] = question.strip()
    return example


ds_test = load_dataset(ORIGINAL_NAME_DS, split=ORIGINAL_SPLIT_DS)
columns_to_remove = [c_n for c_n in ds_test.column_names if c_n not in list(FEATURES.keys())]
ds_test = ds_test.map(
    map_func_transform_ai2d_ds, remove_columns=columns_to_remove, features=FEATURES, num_proc=NUM_PROC
)
print(ds_test[0]["question"])


ds_all_splits = DatasetDict({"test": ds_test})
ds_all_splits.push_to_hub(NAME_DS_PUSH_HUB, private=True)

# Cache dataset
test_loading = load_dataset(NAME_DS_PUSH_HUB)
