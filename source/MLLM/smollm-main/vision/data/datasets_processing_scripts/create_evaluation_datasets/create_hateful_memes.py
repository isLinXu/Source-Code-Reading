import random

from datasets import load_dataset


ds = load_dataset("HuggingFaceM4/Hateful-Memes", "seen", use_auth_token=True)
print(f"Composition of the original dataset: {ds}")

number_of_examples_support_sets = 2048
number_of_examples_captioning_validation_query_sets = 1000
repo_id = "HuggingFaceM4/HatefulMemes_support_query_sets"

# Check how the train set is composed
ds_train_not_hateful = ds["train"].filter(lambda example: example["label"] == 0)
ds_train_hateful = ds["train"].filter(lambda example: example["label"] == 1)

print(
    f"Number of examples in the train set: {len(ds['train'])}\nNumber of examples in the train set that are not"
    f" hateful: {len(ds_train_not_hateful)}\nNumber of examples in the train set that are hateful:"
    f" {len(ds_train_hateful)}"
)

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

indices_validation_query_set, remaining_indices_train_set = (
    remaining_indices_train_set[:number_of_examples_captioning_validation_query_sets],
    remaining_indices_train_set[number_of_examples_captioning_validation_query_sets:],
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


# Save and push to hub newly created splits
validation_support_set = ds["train"].select(indices_validation_support_set)
validation_support_set.push_to_hub(repo_id, "validation_support_set", private=True)

validation_query_set = ds["train"].select(indices_validation_query_set)
validation_query_set.push_to_hub(repo_id, "validation_query_set", private=True)

test_support_set = ds["train"].select(indices_test_support_set)
test_support_set.push_to_hub(repo_id, "test_support_set", private=True)

test_query_set = ds["test"]
test_query_set.push_to_hub(repo_id, "test_query_set", private=True)

# Load the newly created dataset from hub
ds_final = load_dataset(repo_id, use_auth_token=True)

# Print the final composition of the dataset
print(f"Composition of the final dataset: {ds_final}")
