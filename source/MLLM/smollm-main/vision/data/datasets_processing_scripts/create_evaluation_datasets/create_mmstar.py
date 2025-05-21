"""
srun --pty --cpus-per-task=24 --mem-per-cpu=20G --partition=hopper-cpu --qos high bash -i
conda activate shared-m4
"""


import datasets
from datasets import DatasetDict, load_dataset


ORIGINAL_NAME_DS = "Lin-Chen/MMStar"
ORIGINAL_SPLIT_DS = "val"

NUM_PROC = 24

POSSIBLE_LABELS = ["A", "B", "C", "D"]

FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "label": datasets.features.ClassLabel(names=POSSIBLE_LABELS),
        "image": datasets.Image(decode=True),
    }
)

NAME_DS_PUSH_HUB = "HuggingFaceM4/MMStar"


def map_func_transform_mmstar_ds(example):
    example["id"] = str(example["index"])
    example["label"] = example["answer"]
    #
    question = example["question"]
    strings_to_remove = [
        "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n",
        (
            "Hint: Please answer the question requiring a floating-point number with two decimal places and provide"
            " the final value, e.g., 1.23, 1.34, 1.45, at the end.\n"
        ),
        (
            "Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3,"
            " at the end.\n"
        ),
        "H+B53int: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.\n",
    ]
    for string_to_remove in strings_to_remove:
        question = question.replace(string_to_remove, "")
    #
    question = question.strip()
    #
    if "Question:" not in question:
        question = "Question: " + question
    #
    question = question.replace("Options: ", "Choices:\n")
    if "\nChoices:\n" not in question:
        question = question.replace("(A)", "Choices:\n(A)")
    assert "\nChoices:\n" in question
    #
    question = question.replace("A: ", "A. ")
    question = question.replace(", B: ", "\nB. ")
    question = question.replace(", C: ", "\nC. ")
    question = question.replace(", D: ", "\nD. ")
    #
    question = question.replace("(A) ", "A. ")
    question = question.replace("(B) ", "B. ")
    question = question.replace("(C) ", "C. ")
    question = question.replace("(D) ", "D. ")
    question = question.replace("(D)", "D. ")
    #
    assert "\nA. " in question
    assert "\nB. " in question
    assert "\nC. " in question
    # Commented because of one counter example
    # assert "\nD. " in question
    #
    question = question.strip()
    question = question + "\nAnswer with the letter."
    #
    example["question"] = question
    return example


ds_val = load_dataset(ORIGINAL_NAME_DS, split=ORIGINAL_SPLIT_DS)
columns_to_remove = [c_n for c_n in ds_val.column_names if c_n not in list(FEATURES.keys())]
ds_val = ds_val.map(
    map_func_transform_mmstar_ds, remove_columns=columns_to_remove, features=FEATURES, num_proc=NUM_PROC
)


ds_all_splits = DatasetDict({"validation": ds_val})
ds_all_splits.push_to_hub(NAME_DS_PUSH_HUB, private=True)

# Cache dataset
test_loading = load_dataset(NAME_DS_PUSH_HUB)
