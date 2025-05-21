"""
srun --pty --cpus-per-task=96 --mem-per-cpu=20G --partition=hopper-prod bash -i
conda activate shared-m4
"""

from pathlib import Path

import numpy as np
from datasets import concatenate_datasets, load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm


COMMON_PATH_DATASETS = Path("/fsx/hugo/fine_tuning_datasets_merge_image_individual")

NUM_PROC = 96

# The datasets for which it's obvious there is no overlap are not considered
ALL_DS_NAMES = [
    "vistext",
    "textcaps",
    "vqav2",
    "visual7w",
    "okvqa",
    "cocoqa",
    "vsr",
    "iam",
    "textvqa",
    "st_vqa",
    "diagram_image_to_text",
    "docvqa",
    "infographic_vqa",
    "visualmrc",
    "ai2d",
    "ocrvqa",
    "scienceqa",
    "aokvqa",
    "nlvr2",
    "iconqa",
    "clevr",
    "tallyqa",
    "intergps",
    "geomverse",
]


all_datasets = [load_from_disk(COMMON_PATH_DATASETS / ds_name) for ds_name in tqdm(ALL_DS_NAMES)]

all_datasets = concatenate_datasets(all_datasets)


def compute_ahash(image, hash_size=10):
    image = image.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    image = image.convert("L")
    pixels = np.array(image)
    avg = pixels.mean()
    diff = pixels > avg
    hash = np.packbits(diff.astype(int))
    hex_hash = "".join(f"{byte:02x}" for byte in hash)
    return hex_hash


def map_compute_hash_images(example):
    if "images" in example:
        image = example["images"][0]
    elif "image" in example:
        image = example["image"]
    image_hash = compute_ahash(image, hash_size=10)
    example["image_hash"] = image_hash
    return example


all_datasets_hashes = all_datasets.map(
    map_compute_hash_images, remove_columns=all_datasets.column_names, num_proc=NUM_PROC
)
all_train_hashes = all_datasets_hashes["image_hash"]
all_train_hashes = set(all_train_hashes)


ds_mmbench = load_dataset("HuggingFaceM4/MMBench_modif_chatbot")

hashes_mmbench = ds_mmbench["validation"].map(map_compute_hash_images, num_proc=NUM_PROC)
hashes_mmbench = hashes_mmbench["image_hash"]

idx_dedup = [idx for idx, hash in enumerate(hashes_mmbench) if hash not in all_train_hashes]

ds_mmbench["validation"] = ds_mmbench["validation"].select(idx_dedup)
ds_mmbench.push_to_hub("HuggingFaceM4/MMBench_modif_chatbot_dedup_val")
