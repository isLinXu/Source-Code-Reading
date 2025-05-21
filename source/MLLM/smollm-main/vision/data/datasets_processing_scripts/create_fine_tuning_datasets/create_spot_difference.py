import base64
import copy
import json
from io import BytesIO

import datasets
from datasets import Dataset, concatenate_datasets
from PIL import Image, PngImagePlugin
from tqdm import tqdm


LARGE_ENOUGH_NUMBER = 10000000
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


REPO_ID = "HuggingFaceM4/SpotDifference_4"
NUM_PROC = 48

# s3://m4-datasets/SpotDifference/
PATH_INSTRUCTIONS_SD = "/fsx/hugo/SpotDifference/SD/SD_instructions.json"
PATH_IMAGES_SD = "/fsx/hugo/SpotDifference/SD/SD.json"
PATH_INSTRUCTIONS_GSD = "/fsx/hugo/SpotDifference/GCD/CGD_instructions.json"
PATH_IMAGES_GSD = "/fsx/hugo/SpotDifference/GCD/CGD.json"


def map_func_image_64_to_PIL(example):
    images = example["images"]
    new_images = []
    for idx in range(2):
        new_images.append(Image.open(BytesIO(base64.b64decode(images[idx]))))
    example["images"] = new_images
    return example


# SD (Subtle Difference)

with open(PATH_INSTRUCTIONS_SD) as f:
    # 15989 examples for SD
    data_instructions_sd = json.load(f)["data"]

with open(PATH_IMAGES_SD) as f:
    # 26154 images for SD
    data_images_sd = json.load(f)

instructions_sd = []
answers_sd = []
images_sd = []
for _, data_instruction_ex in tqdm(data_instructions_sd.items()):
    instructions_sd.append(data_instruction_ex["instruction"])
    answers_sd.append(data_instruction_ex["answer"])
    images = []
    for idx in range(2):
        images.append(data_images_sd[data_instruction_ex["image_ids"][idx]])
    images_sd.append(images)

ds_sd = Dataset.from_dict(
    {
        "instruction": instructions_sd,
        "answer": answers_sd,
        "images": images_sd,
    }
)

new_features = copy.deepcopy(ds_sd.features)
new_features["images"] = datasets.Sequence(datasets.Image())
ds_sd = ds_sd.map(map_func_image_64_to_PIL, num_proc=NUM_PROC, features=new_features)


# GSD (General Scene Difference)

with open(PATH_INSTRUCTIONS_GSD) as f:
    # 141869 examples for GSD
    data_instructions_gsd = json.load(f)["data"]

with open(PATH_IMAGES_GSD) as f:
    # 118287 images for GSD
    data_images_gsd = json.load(f)

instructions_gsd = []
answers_gsd = []
images_gsd = []
for _, data_instruction_ex in tqdm(data_instructions_gsd.items()):
    instructions_gsd.append(data_instruction_ex["instruction"])
    answers_gsd.append(data_instruction_ex["answer"])
    images = []
    for idx in range(2):
        images.append(data_images_gsd[data_instruction_ex["image_ids"][idx]])
    images_gsd.append(images)

ds_gsd = Dataset.from_dict(
    {
        "instruction": instructions_gsd,
        "answer": answers_gsd,
        "images": images_gsd,
    }
)

new_features = copy.deepcopy(ds_gsd.features)
new_features["images"] = datasets.Sequence(datasets.Image())
ds_gsd = ds_gsd.map(map_func_image_64_to_PIL, num_proc=NUM_PROC, features=new_features)


all_ds = concatenate_datasets([ds_sd, ds_gsd])
all_ds = all_ds.shuffle(seed=42)

# all_ds.save_to_disk("/fsx/hugo/SpotDifferenceDataset", num_proc=NUM_PROC)
all_ds.push_to_hub(REPO_ID, private=True)
