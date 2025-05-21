# Download images and dialogs
"""
mkdir /scratch/coco
cd /scratch/coco
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm train2014.zip
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip

mkdir /scratch/visdial
cd /scratch/visdial
aws s3 sync s3://m4-datasets/trash/visdial/ ./
unzip VisualDialog_val2018.zip
unzip visdial_1.0_train.zip
unzip visdial_1.0_val.zip
rm *.zip
"""


import copy
import json
import os
import random

import datasets
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm


with open("/scratch/visdial/visdial_1.0_train.json") as f:
    data_train = json.load(f)["data"]
with open("/scratch/visdial/visdial_1.0_val.json") as f:
    data_val = json.load(f)["data"]
with open("/scratch/visdial/visdial_1.0_val_dense_annotations.json") as f:
    data_relevance_scores = json.load(f)

data = {
    "train": {
        "dialogs": data_train["dialogs"],
        "questions": data_train["questions"],
        "answers": data_train["answers"],
    },
    "validation": {
        "dialogs": data_val["dialogs"],
        "questions": data_val["questions"],
        "answers": data_val["answers"],
    },
}

incomplete_ds = load_dataset("jxu124/visdial")

all_answers = list(set(data["train"]["answers"] + data["validation"]["answers"]))

ds_data = {split: [] for split in ["train", "validation"]}
for split in ["train", "validation"]:
    assert len(data[split]["dialogs"]) == len(incomplete_ds[split])
    questions = data[split]["questions"]
    answers = data[split]["answers"]
    for idx_ex, example in enumerate(tqdm(data[split]["dialogs"])):
        assert incomplete_ds[split][idx_ex]["image_path"].replace(".jpg", "").endswith(str(example["image_id"]))
        image_path = os.path.join("/scratch/", incomplete_ds[split][idx_ex]["image_path"])
        # Discard examples using images from COCO val from the train set (but doesn't remove anything in practice).
        if "COCO_val2014" in image_path:
            pass
        caption = example["caption"]
        questions_example = []
        answers_example = []
        possible_answers_example = []
        assert len(example["dialog"]) == 10
        for d_ex in example["dialog"]:
            questions_example.append(questions[d_ex["question"]])
            answers_example.append(answers[d_ex["answer"]])
            assert len(d_ex["answer_options"]) == 100
            possible_answers_example.append([answers[ans_id] for ans_id in d_ex["answer_options"]])
        ds_data[split].append(
            {
                "image_path": image_path,
                "caption": caption,
                "questions": questions_example,
                "answers": answers_example,
                "answers_options": possible_answers_example,
            }
        )

number_of_examples_support_sets = 2048
number_of_examples_qa_validation_query_sets = 1024
repo_id = "HuggingFaceM4/VisDial_modif_support_query_sets"

indices_train_set = list(range(0, len(ds_data["train"])))
random.shuffle(indices_train_set)
remaining_indices_train_set = indices_train_set

indices_validation_support_set, remaining_indices_train_set = (
    remaining_indices_train_set[:number_of_examples_support_sets],
    remaining_indices_train_set[number_of_examples_support_sets:],
)

indices_test_support_set, remaining_indices_train_set = (
    remaining_indices_train_set[:number_of_examples_support_sets],
    remaining_indices_train_set[number_of_examples_support_sets:],
)

indices_validation_query_set, remaining_indices_train_set = (
    remaining_indices_train_set[:number_of_examples_qa_validation_query_sets],
    remaining_indices_train_set[number_of_examples_qa_validation_query_sets:],
)

# print lengths
print(
    f"Lengths of the sets:\nvalidation query set: {len(indices_validation_query_set)}\nvalidation support set:"
    f" {len(indices_validation_support_set)}\ntest support set: {len(indices_test_support_set)}"
)

# Check that we have no overlap between the sets
print("Intersection between the sets:\n")
print(set(indices_validation_query_set).intersection(set(indices_validation_support_set)))
print(set(indices_validation_query_set).intersection(set(indices_test_support_set)))


# Validation support set
new_ds_image_path = []
new_ds_caption = []
new_ds_context = []
new_ds_answer = []
new_ds_answer_options = []
new_ds_relevance_scores = []
set_indices_validation_support_set = set(indices_validation_support_set)
for idx_ex, example in enumerate(ds_data["train"]):
    if idx_ex in set_indices_validation_support_set:
        new_ds_image_path.append(example["image_path"])
        caption = example["caption"]
        new_ds_caption.append(caption)
        context = ""
        for idx_q_a, (ques, ans) in enumerate(zip(example["questions"], example["answers"])):
            if idx_q_a < len(example["questions"]) - 1:
                context += f"Question: {ques}? Answer: {ans}. "
            else:
                context += f"Question: {ques}? Answer: "
        new_ds_context.append(context)
        new_ds_answer.append(example["answers"][-1])
        new_ds_answer_options.append(example["answers_options"][-1])
        new_ds_relevance_scores.append([0.0] * 100)
ds_val_support_set = Dataset.from_dict(
    {
        "image_path": new_ds_image_path,
        "caption": new_ds_caption,
        "context": new_ds_context,
        "answer": new_ds_answer,
        "answer_options": new_ds_answer_options,
        "relevance_scores": new_ds_relevance_scores,
    }
)

# Test support set
new_ds_image_path = []
new_ds_caption = []
new_ds_context = []
new_ds_answer = []
new_ds_answer_options = []
new_ds_relevance_scores = []
set_indices_test_support_set = set(indices_test_support_set)
for idx_ex, example in enumerate(ds_data["train"]):
    if idx_ex in set_indices_test_support_set:
        new_ds_image_path.append(example["image_path"])
        caption = example["caption"]
        new_ds_caption.append(caption)
        context = ""
        for idx_q_a, (ques, ans) in enumerate(zip(example["questions"], example["answers"])):
            if idx_q_a < len(example["questions"]) - 1:
                context += f"Question: {ques}? Answer: {ans}. "
            else:
                context += f"Question: {ques}? Answer: "
        new_ds_context.append(context)
        new_ds_answer.append(example["answers"][-1])
        new_ds_answer_options.append(example["answers_options"][-1])
        new_ds_relevance_scores.append([0.0] * 100)
ds_test_support_set = Dataset.from_dict(
    {
        "image_path": new_ds_image_path,
        "caption": new_ds_caption,
        "context": new_ds_context,
        "answer": new_ds_answer,
        "answer_options": new_ds_answer_options,
        "relevance_scores": new_ds_relevance_scores,
    }
)

# Validation query set
new_ds_image_path = []
new_ds_caption = []
new_ds_context = []
new_ds_answer = []
new_ds_answer_options = []
new_ds_relevance_scores = []
set_indices_validation_query_set = set(indices_validation_query_set)
for idx_ex, example in enumerate(ds_data["train"]):
    if idx_ex in set_indices_validation_query_set:
        new_ds_image_path.append(example["image_path"])
        caption = example["caption"]
        new_ds_caption.append(caption)
        context = ""
        idx_q_a_chosen = random.randint(
            0, 9
        )  # We choose one question-answer pair and the previous ones will form the dialog history
        for idx_q_a, (ques, ans) in enumerate(zip(example["questions"], example["answers"])):
            if idx_q_a < idx_q_a_chosen:
                context += f"Question: {ques}? Answer: {ans}. "
            elif idx_q_a == idx_q_a_chosen:
                context += f"Question: {ques}? Answer: "
        new_ds_context.append(context)
        new_ds_answer.append(example["answers"][idx_q_a_chosen])
        new_ds_answer_options.append(example["answers_options"][idx_q_a_chosen])
        # Artificially created relevance scores, since they don't provide them for the training set
        relevance_scores = [0] * len(example["answers_options"][idx_q_a_chosen])
        relevance_scores[example["answers_options"][idx_q_a_chosen].index(example["answers"][idx_q_a_chosen])] = 1
        new_ds_relevance_scores.append(relevance_scores)
ds_val_query_set = Dataset.from_dict(
    {
        "image_path": new_ds_image_path,
        "caption": new_ds_caption,
        "context": new_ds_context,
        "answer": new_ds_answer,
        "answer_options": new_ds_answer_options,
        "relevance_scores": new_ds_relevance_scores,
    }
)

# Test query set
new_ds_image_path = []
new_ds_caption = []
new_ds_context = []
new_ds_answer = []
new_ds_answer_options = []
new_ds_relevance_scores = []
for idx_ex, example in enumerate(ds_data["validation"]):  # We now consider the true validation for the test
    info_relevance_scores = data_relevance_scores[idx_ex]
    assert str(data_relevance_scores[idx_ex]["image_id"]) in example["image_path"]
    # We only have the relevance scores for one round out of the 10 per example
    # Otherwise, we should have considered all the question-answer pairs for each example for the test set
    idx_q_a_chosen = data_relevance_scores[0]["round_id"] - 1
    new_ds_image_path.append(example["image_path"])
    caption = example["caption"]
    new_ds_caption.append(caption)
    context = ""
    for idx_q_a, (ques, ans) in enumerate(zip(example["questions"], example["answers"])):
        if idx_q_a < idx_q_a_chosen:
            context += f"Question: {ques}? Answer: {ans}. "
        elif idx_q_a == idx_q_a_chosen:
            context += f"Question: {ques}? Answer: "
    new_ds_context.append(context)
    new_ds_answer.append(example["answers"][idx_q_a_chosen])
    new_ds_answer_options.append(example["answers_options"][idx_q_a_chosen])
    new_ds_relevance_scores.append(info_relevance_scores["gt_relevance"])
ds_test_query_set = Dataset.from_dict(
    {
        "image_path": new_ds_image_path,
        "caption": new_ds_caption,
        "context": new_ds_context,
        "answer": new_ds_answer,
        "answer_options": new_ds_answer_options,
        "relevance_scores": new_ds_relevance_scores,
    }
)


def func_map_add_images(example):
    path = example["image_path"]
    example["image"] = Image.open(os.path.join("/scratch/", path))
    return example


new_features = copy.deepcopy(ds_val_support_set.features)
new_features["image"] = datasets.Image()
new_features["answer"] = datasets.ClassLabel(num_classes=len(all_answers), names=all_answers)

ds_val_support_set = ds_val_support_set.map(func_map_add_images, num_proc=20, features=copy.deepcopy(new_features))
ds_test_support_set = ds_test_support_set.map(func_map_add_images, num_proc=20, features=copy.deepcopy(new_features))
ds_val_query_set = ds_val_query_set.map(func_map_add_images, num_proc=20, features=copy.deepcopy(new_features))
ds_test_query_set = ds_test_query_set.map(func_map_add_images, num_proc=20, features=copy.deepcopy(new_features))

# Save and push to hub newly created splits
ds_val_support_set.push_to_hub(repo_id, "validation_support_set", private=True)
ds_val_query_set.push_to_hub(repo_id, "validation_query_set", private=True)
ds_test_support_set.push_to_hub(repo_id, "test_support_set", private=True)
ds_test_query_set.push_to_hub(repo_id, "test_query_set", private=True)

# Load the newly created dataset from hub
ds_final = load_dataset(repo_id, use_auth_token=True)

# Print the final composition of the dataset
print(f"Composition of the final dataset: {ds_final}")
