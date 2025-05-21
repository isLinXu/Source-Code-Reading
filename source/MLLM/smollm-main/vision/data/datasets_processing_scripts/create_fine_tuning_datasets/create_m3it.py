import base64
from io import BytesIO

from datasets import concatenate_datasets, load_dataset
from PIL import Image
from tqdm import tqdm


DS_NAME = "MMInstruction/M3IT"
NUM_PROC = 48
REPO_ID = "HuggingFaceM4/M3IT"

# We remove the Chinese and video benchmarks
INTERESTING_CONFIGS = [
    # Image Captioning
    "coco",
    "textcap",
    "image-paragraph-captioning",
    # Classification
    "coco-goi",
    "coco-text",
    "imagenet",
    "coco-itm",
    "snli-ve",
    "mocheg",
    "iqa",
    # Visual Question Answering
    "vqa-v2",
    "shapes",
    "docvqa",
    "ocr-vqa",
    "st-vqa",
    "text-vqa",
    "gqa",
    # Knowledgeable Visual QA
    "okvqa",
    "a-okvqa",
    "science-qa",
    "viquae",
    # Reasoning
    "clevr",
    # "nlvr",
    "vcr",
    "visual-mrc",
    # "winoground",  # Doesn't have train split
    # Generation
    # "vist",  # Doesn't work
    "visual-dialog",
    # "multi30k",  # For translation
]


# Specifically for DocVQA, because the big images cause problems
# Keeps 23097/39463 examples
def filter_func_remove_huge_images(example):
    if len(example["image_base64_str"][0]) > 1_000_000:
        return False
    return True


def filter_func_check_one_image(example):
    if len(example["image_base64_str"]) > 1:
        raise ValueError("Found 2 images instead of 1")
    return True


def map_func_add_image_column(example):
    if len(example["image_base64_str"]) > 1:
        raise ValueError("Found 2 images instead of 1")
    img_base64_str = example["image_base64_str"][0]
    if img_base64_str is None:  # Sometimes for ScienceQA
        example["image"] = None
    else:
        example["image"] = Image.open(BytesIO(base64.b64decode(img_base64_str)))
    return example


all_ds = [load_dataset(DS_NAME, config, split="train") for config in tqdm(INTERESTING_CONFIGS)]

idx_docvqa = INTERESTING_CONFIGS.index("docvqa")
all_ds[idx_docvqa] = all_ds[idx_docvqa].filter(filter_func_remove_huge_images, num_proc=NUM_PROC)

all_ds = [ds.filter(filter_func_check_one_image, num_proc=NUM_PROC) for ds in tqdm(all_ds)]

all_ds = [ds.map(map_func_add_image_column, num_proc=NUM_PROC) for ds in tqdm(all_ds)]

all_ds = concatenate_datasets(all_ds)

all_ds = all_ds.remove_columns("image_base64_str")

all_ds = all_ds.shuffle(seed=42)

all_ds.save_to_disk("/fsx/hugo/M3IT", num_proc=NUM_PROC)

all_ds.push_to_hub(REPO_ID)
