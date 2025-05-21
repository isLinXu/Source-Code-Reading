import random

from datasets import DatasetDict, concatenate_datasets, load_dataset


ds = load_dataset(
    "datasets_processing_scripts/create_evaluation_datasets/NLVR2/NLVR2.py",
    use_auth_token=True,
)
print(f"Composition of the original dataset: {ds}")
number_of_examples_support_sets = 2048
number_of_examples_captioning_validation_query_sets = 1024
repo_id = "HuggingFaceM4/NLVR2_support_query_sets"

# split the train dataset per bucket (100 of them) to get less repeated examples for each support set
datasets_splitted_in_buckets = []
for i in range(100):
    new_ds = ds["train"].filter(lambda exs: [int(directory) == i for directory in exs["directory"]], batched=True)
    datasets_splitted_in_buckets.append(new_ds)

validation_support_set = []
test_support_set = []
for i, curr_ds in enumerate(datasets_splitted_in_buckets):
    # Condition necessary to have 2048 examples exactly, with 20 examples per dataset until dataset 51 and 21 examples for datasets > 51
    if i < 52:
        number_of_examples_support_sets_per_dataset = number_of_examples_support_sets // len(
            datasets_splitted_in_buckets
        )
    else:
        number_of_examples_support_sets_per_dataset = (
            number_of_examples_support_sets // len(datasets_splitted_in_buckets) + 1
        )
    indices_current_train_set = list(range(0, len(curr_ds)))
    random.shuffle(indices_current_train_set)
    remaining_indices_current_train_set = indices_current_train_set

    indices_validation_support_set = remaining_indices_current_train_set[:number_of_examples_support_sets_per_dataset]
    remaining_indices_current_train_set = remaining_indices_current_train_set[
        number_of_examples_support_sets_per_dataset:
    ]

    indices_test_support_set, remaining_indices_train_set = (
        remaining_indices_current_train_set[:number_of_examples_support_sets_per_dataset],
        remaining_indices_current_train_set[number_of_examples_support_sets_per_dataset:],
    )

    validation_support_set.append(curr_ds.select(indices_validation_support_set))
    test_support_set.append(curr_ds.select(indices_test_support_set))

validation_support_set = concatenate_datasets(validation_support_set)
test_support_set = concatenate_datasets(test_support_set)

# Get validation query set from validation split
indices_validation_set = list(range(0, len(ds["validation"])))
random.shuffle(indices_validation_set)
indices_validation_query_set = indices_validation_set[:number_of_examples_captioning_validation_query_sets]

validation_query_set = ds["validation"].select(indices_validation_query_set)

print(
    f"Lengths of the sets:\nvalidation query set: {len(validation_query_set)}\nvalidation support set:"
    f" {len(validation_support_set)}\ntest support set: {len(test_support_set)}"
)

test_query_set = ds["test"]

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
