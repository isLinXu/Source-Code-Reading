"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4
"""


import hashlib
import json

from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm


NAME_DS_TO_HASH = [
    "HuggingFaceM4/MMBench_modif_chatbot",
    "HuggingFaceM4/MathVista-modif",
    "HuggingFaceM4/MMMU-modif",
]

PATH_SAVE_LIST_HASHES = "/fsx/hugo/fine_tuning_datasets_merge_image_individual/list_hashes_test_images.json"


list_hashes = []

for name_ds in tqdm(NAME_DS_TO_HASH):
    potential_subset_names = ["testmini", "test", "validation", "dev"]
    all_splits = []
    for split in potential_subset_names:
        try:
            all_splits.append(load_dataset(name_ds, split=split))
        except Exception:
            pass
    ds = concatenate_datasets(all_splits)
    if "image" in ds.column_names:
        images = ds["image"]
    elif "images" in ds.column_names:
        images = ds["images"]
        images = [img for list_images in images for img in list_images]
    else:
        raise ValueError("images not found in the dataset")
    for img in tqdm(images):
        md5hash = hashlib.md5(img.tobytes()).hexdigest()
        list_hashes.append(md5hash)


with open(PATH_SAVE_LIST_HASHES, "w") as f:
    json.dump(list_hashes, f)
