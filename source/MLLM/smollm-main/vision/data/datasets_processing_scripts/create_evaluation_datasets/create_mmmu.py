import ast

import datasets
from datasets import load_dataset

from m4.training.utils import END_OF_UTTERANCE_TOKEN


# Create coealesced MMMU dataset needs to be done only once. leaving the commented code for reference
# from tqdm import tqdm
# mmmu_subjects = [
#     "Accounting",
#     "Agriculture",
#     "Architecture_and_Engineering",
#     "Art",
#     "Art_Theory",
#     "Basic_Medical_Science",
#     "Biology",
#     "Chemistry",
#     "Clinical_Medicine",
#     "Computer_Science",
#     "Design",
#     "Diagnostics_and_Laboratory_Medicine",
#     "Economics",
#     "Electronics",
#     "Energy_and_Power",
#     "Finance",
#     "Geography",
#     "History",
#     "Literature",
#     "Manage",
#     "Marketing",
#     "Materials",
#     "Math",
#     "Mechanical_Engineering",
#     "Music",
#     "Pharmacy",
#     "Physics",
#     "Psychology",
#     "Public_Health",
#     "Sociology",
# ]

#
# dev_datasets = []
# validation_datasets = []
# test_datasets = []
# for subject in tqdm(mmmu_subjects):
#     dev_datasets.append(load_dataset("MMMU/MMMU", f"{subject}", split="dev", use_auth_token=True))
#     validation_datasets.append(load_dataset("MMMU/MMMU", f"{subject}", split="validation", use_auth_token=True))
#     test_datasets.append(load_dataset("MMMU/MMMU", f"{subject}", split="test", use_auth_token=True))

# dev_ds = datasets.concatenate_datasets(dev_datasets)
# validation_ds = datasets.concatenate_datasets(validation_datasets)
# test_ds = datasets.concatenate_datasets(test_datasets)

# dev_ds.push_to_hub("HuggingFaceM4/MMMU", split="dev", private=True)
# validation_ds.push_to_hub("HuggingFaceM4/MMMU", split="validation", private=True)
# test_ds.push_to_hub("HuggingFaceM4/MMMU", split="test", private=True)


def format_value_to_list(value):
    try:
        evaluated_value = ast.literal_eval(value)
        if isinstance(evaluated_value, list):
            return evaluated_value
        else:
            return [value]
    except (SyntaxError, ValueError):
        return [value]


dev_ds = load_dataset("HuggingFaceM4/MMMU", split="dev")
validation_ds = load_dataset("HuggingFaceM4/MMMU", split="validation")
test_ds = load_dataset("HuggingFaceM4/MMMU", split="test")
# Make MMMU-modif
FEATURES = datasets.Features(
    {
        "id": datasets.Value("string"),
        "question": datasets.Value("string"),
        "answers": datasets.Sequence(datasets.Value("string")),
        "images": datasets.Sequence(datasets.Image(decode=True)),
        "question_type": datasets.Value("string"),
        "explanation": datasets.Value("string"),
        "topic_difficulty": datasets.Value("string"),
        "subfield": datasets.Value("string"),
        "img_type": datasets.Value("string"),
        "broad_category": datasets.Value("string"),
        "narrow_category": datasets.Value("string"),
    }
)

MMMU_CATEGORIES = {
    "Art & Design": ["Art", "Design", "Music", "Art_Theory"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health & Medicine": [
        "Basic_Medical_Science",
        "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine",
        "Pharmacy",
        "Public_Health",
    ],
    "Humanities & Social Science": ["History", "Literature", "Psychology", "Sociology"],
    "Tech & Engineering": [
        "Agriculture",
        "Architecture_and_Engineering",
        "Computer_Science",
        "Electronics",
        "Energy_and_Power",
        "Materials",
        "Mechanical_Engineering",
    ],
}

removed_columns = ["options", "answer", "image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]
options_choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]
image_numbers = ["1", "2", "3", "4", "5", "6", "7"]


def map_transform_ds_mmmu(example):
    example["images"] = [
        example[f"image_{image_number}"]
        for image_number in image_numbers
        if example[f"image_{image_number}"] is not None
    ]
    example["narrow_category"] = "_".join(example["id"].split("_")[1:-1])
    for broad_category, subfields in MMMU_CATEGORIES.items():
        if example["narrow_category"] in subfields:
            example["broad_category"] = broad_category
            break

    image_string = ""
    for i in range(len(example["images"])):
        image_string += f"<image {image_numbers[i]}>:<image>\n"
    example["question"] = image_string.strip() + "\n" + example["question"]
    options = format_value_to_list(example["options"])
    example["answers"] = format_value_to_list(example["answer"]) if example["answer"] != "?" else [""]

    len_options = len(options)
    if len_options != 0:
        example["question"] = "Question: " + example["question"]
        option_string = "\nChoices:"
        for i in range(len_options):
            option_string += f"\n{options_choices[i]}. {options[i]}"
        option_string += f"\nAnswer with the letter.{END_OF_UTTERANCE_TOKEN}\nAssistant: Answer:"
        example["question"] = example["question"] + option_string
    else:
        example["question"] = example["question"] + f"{END_OF_UTTERANCE_TOKEN}\nAssistant:"
    return example


test_ds = test_ds.map(map_transform_ds_mmmu, remove_columns=removed_columns, features=FEATURES, num_proc=6)
validation_ds = validation_ds.map(map_transform_ds_mmmu, remove_columns=removed_columns, features=FEATURES, num_proc=5)
dev_ds = dev_ds.map(map_transform_ds_mmmu, remove_columns=removed_columns, features=FEATURES, num_proc=5)

test_ds.push_to_hub("HuggingFaceM4/MMMU-modif-with-categories", split="test", private=True)
validation_ds.push_to_hub("HuggingFaceM4/MMMU-modif-with-categories", split="validation", private=True)
dev_ds.push_to_hub("HuggingFaceM4/MMMU-modif-with-categories", split="dev", private=True)
