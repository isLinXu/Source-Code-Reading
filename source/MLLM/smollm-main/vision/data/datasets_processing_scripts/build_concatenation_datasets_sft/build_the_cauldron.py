import hashlib
import json
import os
import sys

from datasets import load_from_disk


IDX_JOB = int(sys.argv[1])
ALL_NAMES_DS = [
    "ai2d",
    "aokvqa",
    "chart2text",
    "chartqa",
    "clevr_math",
    "clevr",
    "cocoqa",
    "datikz",
    "diagram_image_to_text",
    "docvqa",
    "dvqa",
    "figureqa",
    "finqa",
    "geomverse",
    "hateful_memes",
    "hitab",
    "iam",
    "iconqa",
    "infographic_vqa",
    "intergps",
    "localized_narratives",
    "mapqa",
    "mimic_cgd",
    "multihiertt",
    "nlvr2",
    "ocrvqa",
    "okvqa",
    "plotqa",
    "raven",
    "rendered_text",
    "robut_sqa",
    "robut_wikisql",
    "robut_wtq",
    "scienceqa",
    "screen2words",
    "spot_the_diff",
    "st_vqa",
    "tabmwp",
    "tallyqa",
    "tat_qa",
    "textcaps",
    "textvqa",
    "tqa",
    "vistext",
    "visual7w",
    "visualmrc",
    "vqarad",
    "vqav2",
    "vsr",
    "websight",
]
NAME_DS = ALL_NAMES_DS[IDX_JOB]

COMMON_FOLDER_DS = "/fsx/hugo/fine_tuning_datasets_merge_image_individual"

MAX_NUM_EXAMPLES = 200_000

PATH_LIST_HASHES_TEST_IMAGES = "/fsx/hugo/fine_tuning_datasets_merge_image_individual/list_hashes_test_images.json"

NUM_PROC = 88

NAME_DS_HUB = "HuggingFaceM4/the_cauldron"


with open(PATH_LIST_HASHES_TEST_IMAGES, "r") as f:
    hashes_test_images = set(json.load(f))


def filter_hash_test_images(example):
    images = example["images"]
    hashes_example_images = [hashlib.md5(img.tobytes()).hexdigest() for img in images]
    if any([hash_ in hashes_test_images for hash_ in hashes_example_images]):
        return False
    return True


ds = load_from_disk(os.path.join(COMMON_FOLDER_DS, NAME_DS))
if ds.num_rows > MAX_NUM_EXAMPLES:
    ds = ds.select(range(MAX_NUM_EXAMPLES))
num_original_examples = ds.num_rows

ds = ds.filter(filter_hash_test_images, num_proc=NUM_PROC)
print("Filtering done")
print("Removed", num_original_examples - ds.num_rows, "examples.")

ds.push_to_hub(NAME_DS_HUB, config_name=NAME_DS)
print("Push to hub done")
