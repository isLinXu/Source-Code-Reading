import json
import logging
from pathlib import Path

from datasets import load_from_disk

from m4.sourcing.data_collection.processors.web_document_extractor import urls_to_images


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PATH_SAVE_FILE_MAP_URL_IDX = Path("/home/lucile/local_datasets/enwiki/enwiki-v2-map-url-idx.json")
PATH_SAVE_DIR_DATASET_IMAGES = Path("/home/lucile/local_datasets/enwiki/enwiki-v2-ds-images")

NUM_SHARDS = 68
DATA_DIR = Path("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION")
DATASET_NAME_INCOMPLETE_EXAMPLES = "wikipedia_html_enterprise-with-images-incomplete-v1-v2"
DATASET_NAME_COMPLETE_EXAMPLES_V2 = "wikipedia_html_enterprise-with-images-full-v2-v3"
DATASET_NAME_INCOMPLETE_EXAMPLES_V2 = "wikipedia_html_enterprise-with-images-incomplete-v2-v3"
EXCLUDE_SHARD_IDS = [34]  # This shard is corrupted since the beginning

NUM_PROC = 32

logger.info("Starting reloading image dataset...")
dataset_images = load_from_disk(PATH_SAVE_DIR_DATASET_IMAGES)
logger.info("Finished reloading image dataset")

logger.info("Starting reloading the url to idx mapping dict...")
with open(PATH_SAVE_FILE_MAP_URL_IDX) as f:
    map_url_idx = json.load(f)
logger.info("Finished reloading the url to idx mapping dict")

logger.info("Starting reloading the dataset with incomplete examples...")
# ds_list = []
for shard_id in range(0, NUM_SHARDS):
    if shard_id in EXCLUDE_SHARD_IDS:
        continue
    logger.info(f"Processing shard {shard_id}...")
    shard_dir = DATA_DIR / f"shard_{shard_id}"
    ds_path = shard_dir / DATASET_NAME_INCOMPLETE_EXAMPLES

    ds = load_from_disk(ds_path)
    logger.info("Finished reloading the dataset with incomplete examples")
    dataset = ds.filter(
        lambda examples: [not_found > 0 for not_found in examples["num_not_found"]], batched=True, num_proc=NUM_PROC
    )
    logger.info(f"Number of examples with missing images: {len(dataset)}")

    final_ds = urls_to_images(dataset, dataset_images, map_url_idx, NUM_PROC, some_urls_are_already_retrieved=True)

    complete_examples = final_ds.filter(lambda x: x["num_not_found"] == 0, num_proc=NUM_PROC)
    print(f"Shard {shard_id}: {len(complete_examples)} examples with all images found")
    complete_examples.save_to_disk(shard_dir / DATASET_NAME_COMPLETE_EXAMPLES_V2)

    incomplete_examples = final_ds.filter(lambda x: x["num_not_found"] != 0, num_proc=NUM_PROC)
    print(f"Shard {shard_id}: {len(incomplete_examples)} examples with some images not found")
    incomplete_examples.save_to_disk(shard_dir / DATASET_NAME_INCOMPLETE_EXAMPLES_V2)


# dataset = concatenate_datasets(ds_list)
