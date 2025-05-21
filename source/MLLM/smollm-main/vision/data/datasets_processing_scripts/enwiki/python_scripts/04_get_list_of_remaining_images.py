import logging
from pathlib import Path

from datasets import load_from_disk


NUM_SHARDS = 68
DATA_DIR = Path("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION")
DATASET_NAME_INCOMPLETE_EXAMPLES = "wikipedia_html_enterprise-with-images-incomplete-v1-v2"
NUM_PROC = 32 // 2
REMAINING_URLS_FILENAME = "remaining_urls_v2.txt"
EXCLUDE_SHARD_IDS = [1, 34]  # This shard is corrupted since the beginning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# def get_remaining_urls_from_shard(args):
#     (ds_path, sub_shard_id) = args
#     ds = load_from_disk(ds_path)
#     ds_shard = ds.shard(NUM_PROC, sub_shard_id)

#     remaining_urls = []
#     for img_urls, imgs in zip(ds_shard["images_urls"], ds_shard["images"]):
#         for img_url, img in zip(img_urls, imgs):
#             if img is None:
#                 remaining_urls.append(img_url)

#     return remaining_urls


def get_remaining_urls(examples):
    remaining_urls = [
        img_url
        for img_urls, imgs in zip(examples["images_urls"], examples["images"])
        for img_url, img in zip(img_urls, imgs)
        if img is None
    ]
    return {"remaining_urls": remaining_urls}


all_remaining_urls = set()
for shard_id in range(0, NUM_SHARDS):
    if shard_id in EXCLUDE_SHARD_IDS:
        continue
    logger.info(f"Processing shard {shard_id}...")
    ds_path = DATA_DIR / f"shard_{shard_id}" / DATASET_NAME_INCOMPLETE_EXAMPLES

    # args_pool = [(ds_path, sub_shard_id) for sub_shard_id in range(0, NUM_PROC)]
    # pool = Pool(NUM_PROC)
    # remaining_urls = pool.map(get_remaining_urls_from_shard, args_pool)
    # remaining_urls = [url for sublist in remaining_urls for url in sublist]
    ds = load_from_disk(ds_path)
    ds_remaining_urls = ds.map(
        get_remaining_urls, remove_columns=ds.column_names, num_proc=NUM_PROC, batched=True, batch_size=100
    )

    logger.info(f"Shard {shard_id} has {len(ds_remaining_urls['remaining_urls'])} remaining urls")
    remaining_urls = set(ds_remaining_urls["remaining_urls"])
    logger.info(f"Shard {shard_id} has {len(remaining_urls)} unique remaining urls")
    all_remaining_urls.update(remaining_urls)

logger.info(f"Total number of remaining urls: {len(all_remaining_urls)}")
with open(DATA_DIR / REMAINING_URLS_FILENAME, "w") as f:
    to_write = "\n".join(all_remaining_urls)
    f.write(to_write)
