import random

from datasets import DatasetDict, load_dataset


ds = load_dataset(
    "/Users/leotronchon/Documents/Github/m4/datasets_processing_scripts/create_evaluation_datasets/ScienceQA/scienceqa.py",
    use_auth_token=True,
)
print(f"Composition of the original dataset: {ds}")
number_of_examples_support_sets = 2048
number_of_examples_captioning_validation_query_sets = 1024
repo_id = "HuggingFaceM4/ScienceQA_support_query_sets"


validation_questions = ds["validation"]["question"]
validation_solutions = ds["validation"]["solution"]
validation_questions_solutions = set(zip(validation_questions, validation_solutions))
train_validation_deduplicated = ds["train"].filter(
    lambda example: ((example["question"], example["solution"]) not in validation_questions_solutions)
)

indices_validation_support_set = list(range(0, len(train_validation_deduplicated)))
random.shuffle(indices_validation_support_set)
indices_validation_support_set = indices_validation_support_set[:number_of_examples_support_sets]
validation_support_set = train_validation_deduplicated.select(indices_validation_support_set)


test_questions = ds["test"]["question"]
test_solutions = ds["test"]["solution"]
test_questions_solutions = set(zip(test_questions, test_solutions))
train_test_deduplicated = ds["train"].filter(
    lambda example: ((example["question"], example["solution"]) not in test_questions_solutions)
)

indices_test_support_set = list(range(0, len(train_test_deduplicated)))
random.shuffle(indices_test_support_set)
indices_test_support_set = indices_test_support_set[:number_of_examples_support_sets]
test_support_set = train_test_deduplicated.select(indices_test_support_set)


indices_val_set = list(range(0, len(ds["validation"])))
random.shuffle(indices_val_set)
indices_validation_query_set = indices_val_set[:number_of_examples_captioning_validation_query_sets]
validation_query_set = ds["validation"].select(indices_validation_query_set)
test_query_set = ds["test"]


# print lengths
print(
    f"Lengths of the sets:\nvalidation query set: {len(validation_query_set)}\nvalidation support set:"
    f" {len(indices_validation_support_set)}\ntest query set: {len(test_query_set)}\ntest support set:"
    f" {len(indices_test_support_set)}"
)

ds_to_push = DatasetDict(
    {
        "validation_support_set": validation_support_set,
        "validation_query_set": validation_query_set,
        "test_support_set": test_support_set,
        "test_query_set": test_query_set,
    }
)
ds_to_push.push_to_hub(repo_id, private=True)

# Load the newly created dataset from hub
ds_final = load_dataset(repo_id, use_auth_token=True)

# Print the final composition of the dataset
print(f"Composition of the final dataset: {ds_final}")
