import logging
from pathlib import Path

from datasets import load_from_disk


NUM_SHARDS = 68
DATA_DIR = Path("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION")
DATASET_NAME_COMPLETE_EXAMPLES_V1 = "wikipedia_html_enterprise-with-images-full-v1"
DATASET_NAME_INCOMPLETE_EXAMPLES_V1 = "wikipedia_html_enterprise-with-images-incomplete-v1"
DATASET_NAME_COMPLETE_EXAMPLES_V2 = "wikipedia_html_enterprise-with-images-full-v1-v2"
DATASET_NAME_INCOMPLETE_EXAMPLES_V2 = "wikipedia_html_enterprise-with-images-incomplete-v1-v2"
NUM_PROC = 32

EXCLUDE_SHARD_IDS = [34]  # This shard is corrupted since the beginning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

for shard_id in range(0, NUM_SHARDS):
    if shard_id in EXCLUDE_SHARD_IDS:
        continue
    ds_1_path = DATA_DIR / f"shard_{shard_id}" / DATASET_NAME_COMPLETE_EXAMPLES_V1
    ds_2_path = DATA_DIR / f"shard_{shard_id}" / DATASET_NAME_COMPLETE_EXAMPLES_V2

    ds_1 = load_from_disk(ds_1_path)
    ds_2 = None
    if ds_2_path.exists():
        ds_2 = load_from_disk(ds_2_path)

    len_ds_1 = len(ds_1)
    len_ds_2 = len(ds_2) if ds_2 is not None else 0

    if shard_id == 36:
        print(
            f"Shard {shard_id} has {len_ds_1} examples in {DATASET_NAME_COMPLETE_EXAMPLES_V1} and {len_ds_2} examples"
            f" in {DATASET_NAME_COMPLETE_EXAMPLES_V2}"
        )

    if len_ds_2 != 0 and len_ds_1 > len_ds_2:
        print(
            f"Shard {shard_id} has {len_ds_1} examples in {DATASET_NAME_COMPLETE_EXAMPLES_V1} and {len_ds_2} examples"
            f" in {DATASET_NAME_COMPLETE_EXAMPLES_V2}"
        )

    if len_ds_2 != 0 and len_ds_2 < 19_000:
        print(f"SURPRISE! Shard {shard_id} has {len_ds_2} examples in {DATASET_NAME_COMPLETE_EXAMPLES_V2}")

    if len_ds_2 == 0:
        print(f"No 2nd extraction! Shard {shard_id} has {len_ds_1} examples in {DATASET_NAME_COMPLETE_EXAMPLES_V1}")
