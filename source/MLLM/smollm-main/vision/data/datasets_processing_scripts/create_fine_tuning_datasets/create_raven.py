import glob
import os

import datasets
import numpy as np
from datasets import DatasetDict
from PIL import Image


LOCAL_PATH = "/fsx/m4/victor/raw_datasets/RAVEN-10000"


FEATURES = datasets.Features(
    {
        "panels": [datasets.Image(decode=True)],
        "choices": [datasets.Image(decode=True)],
        "structure": datasets.Array2D(shape=(1, 8), dtype="string"),
        "meta_matrix": datasets.Array2D(shape=(8, 9), dtype="uint8"),
        "meta_target": datasets.Array2D(shape=(1, 9), dtype="uint8"),
        "meta_structure": datasets.Array2D(shape=(1, 21), dtype="uint8"),
        "target": datasets.Value("uint8"),
        "id": datasets.Value("int32"),
        "metadata": datasets.Value("string"),
    }
)


def init_dict():
    dict_to_fill = {
        "panels": [],
        "choices": [],
        "structure": [],
        "meta_matrix": [],
        "meta_target": [],
        "meta_structure": [],
        "target": [],
        "id": [],
        "metadata": [],
    }
    return dict_to_fill


def extract_all_files_for_split(split_files, split):
    print(f"start extracting {len(split_files)} files")
    result_dict = init_dict()

    for idx, path in enumerate(split_files):
        base_name = os.path.basename(path)
        _, id, s = base_name.split("_")
        s = s.split(".")[0]
        assert s == split

        expl_data = np.load(path)

        images = expl_data["image"].reshape(16, 160, 160)

        panels = images[:8]
        panels = [Image.fromarray(pan.astype("uint8")) for pan in panels]

        choices = images[8:]
        choices = [Image.fromarray(cho.astype("uint8")) for cho in choices]

        structure = expl_data["structure"].astype(str)[np.newaxis, ...]
        meta_matrix = expl_data["meta_matrix"]
        meta_target = expl_data["meta_target"][np.newaxis, ...]
        meta_structure = expl_data["meta_structure"][np.newaxis, ...]
        target = expl_data["target"].item()

        result_dict["id"].append(id)
        result_dict["panels"].append(panels)
        result_dict["choices"].append(choices)
        result_dict["structure"].append(structure)
        result_dict["meta_matrix"].append(meta_matrix)
        result_dict["meta_target"].append(meta_target)
        result_dict["meta_structure"].append(meta_structure)
        result_dict["target"].append(target)

        with open(path.replace("npz", "xml"), "r") as xml_file:
            xml_string = xml_file.read()
            result_dict["metadata"].append(xml_string)

        if idx % 1_000 == 0:
            print(split, "Done", idx)

    return result_dict


for config_name in [
    "center_single",
    "distribute_four",
    "distribute_nine",
    "in_center_single_out_center_single",
    "in_distribute_four_out_center_single",
    "left_center_single_right_center_single",
    "up_center_single_down_center_single",
]:
    print(f"start {config_name}")
    # Train
    train_split_files = glob.glob(f"{LOCAL_PATH}/{config_name}/*train*.npz")
    train_dict = extract_all_files_for_split(train_split_files, "train")
    train_dataset = datasets.Dataset.from_dict(train_dict, features=FEATURES, split="train")
    print("Finished building train")

    # Validation
    val_split_files = glob.glob(f"{LOCAL_PATH}/{config_name}/*val*.npz")
    val_dict = extract_all_files_for_split(val_split_files, "val")
    val_dataset = datasets.Dataset.from_dict(val_dict, features=FEATURES, split="validation")
    print("Finished building valdation")

    # Test
    test_split_files = glob.glob(f"{LOCAL_PATH}/{config_name}/*test*.npz")
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
    dataset.push_to_hub("HuggingFaceM4/RAVEN", config_name=config_name)
