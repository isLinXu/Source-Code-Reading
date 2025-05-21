import random

from datasets import DatasetDict, Features, Image, Value, load_dataset


ds = load_dataset("HuggingFaceM4/NoCaps", use_auth_token=True)
print(f"Composition of the original dataset: {ds}")
ds_coco = load_dataset("HuggingFaceM4/COCO", "2014_captions", use_auth_token=True)
print(f"Composition of the original dataset: {ds_coco}")

# Rename columns and delete unnecessary columns
ds_coco = ds_coco.rename_column("sentences_raw", "annotations_captions")
ds_coco = ds_coco.rename_column("imgid", "image_id")

ds_coco = ds_coco.remove_columns(
    [
        col_name
        for col_name in ds_coco["train"].column_names
        if col_name not in ["image", "annotations_captions", "image_id"]
    ]
)
ds = ds.remove_columns(
    [col_name for col_name in ds["test"].column_names if col_name not in ["image", "annotations_captions", "image_id"]]
)
ds = ds.cast(
    Features({"annotations_captions": [Value("string")], "image": Image(decode=True), "image_id": Value("int32")})
)


number_of_examples_support_sets = 2048
number_of_examples_captioning_validation_query_sets = 4500

repo_id = "HuggingFaceM4/NoCaps_support_query_sets"

# Assign the train split indices to the validation support set, and test support set
indices_train_set = list(range(0, len(ds_coco["train"])))
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

# Assign the train split indices to the validation support set, validation query set, and test support set
indices_validation_set = list(range(0, len(ds["validation"])))
random.shuffle(indices_validation_set)

indices_validation_query_set = indices_validation_set[:number_of_examples_captioning_validation_query_sets]

# print lengths
print(
    f"Lengths of the sets:\nvalidation query set: {len(indices_validation_query_set)}\nvalidation support set:"
    f" {len(indices_validation_support_set)}\ntest support set: {len(indices_test_support_set)}"
)

validation_support_set = ds_coco["train"].select(indices_validation_support_set)
test_support_set = ds_coco["train"].select(indices_test_support_set)
server_check_support_set = ds_coco["train"].select(indices_validation_support_set)

validation_query_set = ds["validation"].select(indices_validation_query_set)
test_query_set = ds["test"]
server_check_query_set = ds["validation"]

ds_to_push = DatasetDict(
    {
        "validation_support_set": validation_support_set,
        "validation_query_set": validation_query_set,
        "test_support_set": test_support_set,
        "test_query_set": test_query_set,
        "server_check_support_set": server_check_support_set,
        "server_check_query_set": server_check_query_set,
    }
)

ds_to_push.push_to_hub(repo_id, private=True)


# Load the newly created dataset from hub
ds_final = load_dataset(repo_id, use_auth_token=True)

# Print the final composition of the dataset
print(f"Composition of the final dataset: {ds_final}")
