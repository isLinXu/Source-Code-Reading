"""
srun --pty --cpus-per-task=48 --mem-per-cpu=11G bash -i
conda activate /fsx/m4/conda/shared-m4
"""

import json
import os

import datasets
from datasets import Dataset
from PIL import Image
from tqdm import tqdm


PATH_IMAGES = "/fsx/hugo/sugarcrepe/val2017"  # http://images.cocodataset.org/zips/val2017.zip

PATH_JSON_DATA = "/fsx/hugo/sugarcrepe/json_data/"

ALL_CONFIGS = [
    "swap_obj",
    "swap_att",
    "replace_rel",
    "replace_obj",
    "replace_att",
    "add_obj",
    "add_att",
]


ds_dict = {config: {"image": [], "tested_labels": [], "true_label": []} for config in ALL_CONFIGS}
all_tested_labels = []

for config in ALL_CONFIGS:
    path_data_config = os.path.join(PATH_JSON_DATA, config + ".json")
    with open(path_data_config) as f:
        data_config = json.load(f)
        for _, ex in data_config.items():
            path_image_ex = os.path.join(PATH_IMAGES, ex["filename"])
            image_ex = Image.open(path_image_ex)
            ds_dict[config]["image"].append(image_ex)
            #
            tested_labels_ex = [ex["caption"], ex["negative_caption"]]
            ds_dict[config]["tested_labels"].append(tested_labels_ex)
            all_tested_labels.extend(tested_labels_ex)
            #
            true_label_ex = ex["caption"]
            ds_dict[config]["true_label"].append(true_label_ex)


features = datasets.Features(
    {
        "image": datasets.Image(),
        "tested_labels": datasets.Sequence(datasets.Value("string")),
        "true_label": datasets.ClassLabel(names=list(set(all_tested_labels))),
    }
)
ds = {config: Dataset.from_dict(ds_dict[config], features=features) for config in ALL_CONFIGS}

for config in tqdm(ALL_CONFIGS):
    ds[config].push_to_hub(f"HuggingFaceM4/SugarCrepe_{config}", split="test")
