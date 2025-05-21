import glob
import os

import datasets
import numpy as np
from datasets import DatasetDict
from PIL import Image


LOCAL_PATH = "/fsx/m4/victor/raw_datasets/neutral"


FEATURES = datasets.Features(
    {
        "panels": [datasets.Image(decode=True)],
        "choices": [datasets.Image(decode=True)],
        "relation_structure_encoded": datasets.Array2D(shape=(4, 12), dtype="uint8"),
        "relation_structure": datasets.Array2D(shape=(1, 3), dtype="string"),
        "meta_target": datasets.Array2D(shape=(1, 12), dtype="uint8"),
        "target": datasets.Value("uint8"),
        "id": datasets.Value("int32"),
    }
)


def init_dict():
    dict_to_fill = {
        "panels": [],
        "choices": [],
        "relation_structure_encoded": [],
        "relation_structure": [],
        "meta_target": [],
        "target": [],
        "id": [],
    }
    return dict_to_fill


def extract_all_files_for_split(split_files, split):
    print(f"start extracting {len(split_files)} files")
    result_dict = init_dict()

    for idx, path in enumerate(split_files):
        base_name = os.path.basename(path)
        _, _, s, id = base_name.split("_")
        id = int(id.split(".")[0])
        assert s == split

        expl_data = np.load(path)

        images = expl_data["image"].reshape(16, 160, 160)

        panels = images[:8]
        panels = [Image.fromarray(pan.astype("uint8")) for pan in panels]

        choices = images[8:]
        choices = [Image.fromarray(cho.astype("uint8")) for cho in choices]

        relation_structure_encoded = expl_data["relation_structure_encoded"]
        relation_structure = expl_data["relation_structure"].astype(str)
        meta_target = expl_data["meta_target"][np.newaxis, ...]
        target = expl_data["target"].item()

        result_dict["id"].append(id)
        result_dict["panels"].append(panels)
        result_dict["choices"].append(choices)
        result_dict["relation_structure_encoded"].append(relation_structure_encoded)
        result_dict["relation_structure"].append(relation_structure)
        result_dict["meta_target"].append(meta_target)
        result_dict["target"].append(target)

        if idx % 1_000 == 0:
            print(split, "Done", idx)

    return result_dict


# Train
train_split_files = glob.glob(f"{LOCAL_PATH}/*train*.npz")
train_dict = extract_all_files_for_split(train_split_files, "train")
train_dataset = datasets.Dataset.from_dict(train_dict, features=FEATURES, split="train")
print("Finished building train")


# Validation
val_split_files = glob.glob(f"{LOCAL_PATH}/*val*.npz")
val_dict = extract_all_files_for_split(val_split_files, "val")
val_dataset = datasets.Dataset.from_dict(val_dict, features=FEATURES, split="validation")
print("Finished building valdation")

# Test
test_split_files = glob.glob(f"{LOCAL_PATH}/*test*.npz")
test_dict = extract_all_files_for_split(test_split_files, "test")
test_dataset = datasets.Dataset.from_dict(test_dict, features=FEATURES, split="test")
print("Finished building test")


dataset = DatasetDict(
    {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    }
)
dataset.push_to_hub("HuggingFaceM4/PGM")
