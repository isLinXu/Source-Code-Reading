import datasets
from datasets import load_dataset

from m4.training.utils import END_OF_UTTERANCE_TOKEN


validation_ds = load_dataset("AI4Math/MathVista", split="testmini")
test_ds = load_dataset("AI4Math/MathVista", split="test")

columns_to_remove = [
    "decoded_image",
    "choices",
    "unit",
    "precision",
    "answer",
    "question_type",
    "answer_type",
    "metadata",
    "query",
]
FEATURES = datasets.Features(
    {
        "pid": datasets.Value("string"),
        "question": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "image": datasets.Image(decode=True),
    }
)
options_choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]


def map_transform_val_ds_mathvista(example):
    example["answers"] = [""]
    if example["choices"] is not None:
        assert example["question_type"] == "multi_choice"
        example["question"] += "\nChoices:"
        for i, choice in enumerate(example["choices"]):
            example["question"] += f"\n{options_choices[i]}. {choice}"
            if example["answer"] == choice:
                example["answers"] = [options_choices[i]]
        example["question"] += f"\nAnswer with the letter.{END_OF_UTTERANCE_TOKEN}\nAssistant: Answer:"
        assert example["answers"][0] != ""
    else:
        example["question"] += f"{END_OF_UTTERANCE_TOKEN}\nAssistant:"
        example["answers"] = [example["answer"]]

    example["image"] = example["decoded_image"]
    return example


def map_transform_test_ds_mathvista(example):
    example["answers"] = None
    if example["choices"] is not None:
        assert example["question_type"] == "multi_choice"
        example["question"] += "\nChoices:"
        for i, choice in enumerate(example["choices"]):
            example["question"] += f"\n{options_choices[i]}. {choice}"
        example["question"] += f"\nAnswer with the letter.{END_OF_UTTERANCE_TOKEN}\nAssistant: Answer:"
    else:
        example["question"] += f"{END_OF_UTTERANCE_TOKEN}\nAssistant:"

    example["image"] = example["decoded_image"]
    return example


validation_ds = validation_ds.map(
    map_transform_val_ds_mathvista, remove_columns=columns_to_remove, features=FEATURES, num_proc=1
)
test_ds = test_ds.map(map_transform_test_ds_mathvista, remove_columns=columns_to_remove, features=FEATURES, num_proc=5)

print(test_ds)
test_ds.push_to_hub("HuggingFaceM4/MathVista-modif", split="test", private=True)
validation_ds.push_to_hub("HuggingFaceM4/MathVista-modif", split="validation", private=True)
