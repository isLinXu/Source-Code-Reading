import random
from collections import Counter

from datasets import concatenate_datasets, load_dataset


ds = load_dataset("imagenet-1k", use_auth_token=True)
print(f"Composition of the original dataset: {ds}")

for number_of_examples_per_class_support_sets in [5, 1]:
    number_of_examples_per_class_query_sets = 10
    num_classes = 1000
    repo_id = f"HuggingFaceM4/imagenet1k_support_{number_of_examples_per_class_support_sets}k_query_sets"

    all_sub_ds = {
        "validation_support_set": [],
        "validation_query_set": [],
        "test_support_set": [],
    }
    for class_id in range(num_classes):
        sub_ds = ds["train"].filter(lambda x: [item == class_id for item in x["label"]], batched=True)
        indices_set = list(range(0, len(sub_ds)))
        random.shuffle(indices_set)
        remaining_indices_set = indices_set

        indices_validation_support_set, remaining_indices_set = (
            remaining_indices_set[:number_of_examples_per_class_support_sets],
            remaining_indices_set[number_of_examples_per_class_support_sets:],
        )
        indices_test_support_set, remaining_indices_set = (
            remaining_indices_set[:number_of_examples_per_class_support_sets],
            remaining_indices_set[number_of_examples_per_class_support_sets:],
        )
        indices_validation_query_set, remaining_indices_set = (
            remaining_indices_set[:number_of_examples_per_class_query_sets],
            remaining_indices_set[number_of_examples_per_class_query_sets:],
        )

        all_sub_ds["validation_support_set"].append(sub_ds.select(indices_validation_support_set))
        all_sub_ds["test_support_set"].append(sub_ds.select(indices_test_support_set))
        all_sub_ds["validation_query_set"].append(sub_ds.select(indices_validation_query_set))

    all_sub_ds = {key: concatenate_datasets(all_sub_ds[key]) for key in all_sub_ds.keys()}

    # Print the final number of examples per class in each sub_ds
    for key, sub_ds in all_sub_ds.items():
        print(f"{key}: {len(sub_ds)}")
        count = Counter(sub_ds["label"])
        print(count)
        print("--------------------")

    # Save and push to hub newly created splits
    for key, sub_ds in all_sub_ds.items():
        print(f"{key}: {len(sub_ds)}")
        sub_ds.push_to_hub(repo_id, key, private=True)

    test_query_set = ds["validation"]
    key = "test_query_set"
    print(f"{key}: {len(test_query_set)}")
    test_query_set.push_to_hub(repo_id, key, private=True)

    # Load the newly created dataset from hub
    ds_final = load_dataset(repo_id, use_auth_token=True)

    # Print the final composition of the dataset
    print(f"Composition of the final dataset: {ds_final}")
