"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4
"""


import datasets
from datasets import DatasetDict, load_dataset


NAME_DS = "AI4Math/MathVista"
ALL_SPLITS = ["testmini", "test"]

NUM_PROC = 96

POSSIBLE_LABELS = ["A", "B", "C", "D", "E", "F", "G", "H"]

COLUMNS_TO_KEEP = ["pid", "question", "label", "image"]

FEATURES = datasets.Features(
    {
        "pid": datasets.Value("int64"),
        "question": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=POSSIBLE_LABELS),
        "image": datasets.Image(decode=True),
    }
)

NAME_DS_PUSH_HUB = "HuggingFaceM4/MathVista_MCQ"


def filter_func_math_vista_mcq_ds(example):
    if example["question_type"] == "free_form":
        return False
    elif example["question_type"] == "multi_choice":
        return True
    else:
        raise ValueError("Different `question_type`")


def map_func_transform_math_vista_mcq_ds(example):
    example["pid"] = int(example["pid"])
    example["image"] = example["decoded_image"]
    #
    question = example["question"]
    question = f"Question: {question}\n"
    question += "Choices:\n"
    for idx_choice, choice in enumerate(example["choices"]):
        letter_choice = POSSIBLE_LABELS[idx_choice]
        question += f"{letter_choice}. {choice}\n"
    question += "Answer with the letter."
    example["question"] = question
    #
    if example["answer"] == "":  # split "test"
        example["label"] = None
    else:
        example["label"] = POSSIBLE_LABELS[example["choices"].index(example["answer"])]
    return example


def prepare_math_vista_mcq(split):
    ds = load_dataset(NAME_DS, split=split)
    ds = ds.filter(filter_func_math_vista_mcq_ds, num_proc=NUM_PROC)
    columns_to_remove = [c_n for c_n in ds.column_names if c_n not in COLUMNS_TO_KEEP]
    ds = ds.map(
        map_func_transform_math_vista_mcq_ds, remove_columns=columns_to_remove, features=FEATURES, num_proc=NUM_PROC
    )
    return ds


ds_all_splits = DatasetDict({split: prepare_math_vista_mcq(split=split) for split in ALL_SPLITS})
ds_all_splits.push_to_hub(NAME_DS_PUSH_HUB, private=True)
