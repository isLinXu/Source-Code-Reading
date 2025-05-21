import random

from datasets import DatasetDict, load_dataset


ds = load_dataset(
    "HuggingFaceM4/clevr",
    name="classification",
    use_auth_token=True,
)
print(f"Composition of the original dataset: {ds}")
number_of_examples_support_sets = 2048
number_of_examples_captioning_validation_query_sets = 1024
repo_id = "HuggingFaceM4/Clevr_support_query_sets"
# Assign the train split indices to the validation support set, validation query set, and test support set
indices_train_set = list(range(0, len(ds["train"])))
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

indices_validation_query_set = list(range(0, len(ds["validation"])))
random.shuffle(indices_validation_query_set)
indices_validation_query_set = indices_validation_query_set[:number_of_examples_captioning_validation_query_sets]


# Save and push to hub newly created splits
validation_support_set = ds["train"].select(indices_validation_support_set)

validation_query_set = ds["validation"].select(indices_validation_query_set)

test_support_set = ds["train"].select(indices_test_support_set)

test_query_set = ds["test"]

# print lengths
print(
    f"Lengths of the sets:\nvalidation query set: {len(indices_validation_query_set)}\nvalidation support set:"
    f" {len(indices_validation_support_set)}\ntest query set:{len(test_query_set)} \ntest support set:"
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
