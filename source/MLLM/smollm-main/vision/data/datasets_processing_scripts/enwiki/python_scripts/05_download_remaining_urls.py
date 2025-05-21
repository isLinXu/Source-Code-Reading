from pathlib import Path

from m4.sourcing.data_collection.processors.web_document_extractor import download_images


SHARD_ID = 9
NUM_SHARDS = 33
DATA_DIR = Path("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION")
DATASET_NAME_INCOMPLETE_EXAMPLES = "wikipedia_html_enterprise-with-images-incomplete-v1-v2"
NUM_PROC = 32 // 2
REMAINING_URLS_FILENAME = f"remaining_urls_v2_shard_{SHARD_ID}.txt"
DOWNLOADED_IMAGES_DIRNAME = f"downloaded_images-v3_shard_{SHARD_ID}"


path_save_file_image_urls = DATA_DIR / REMAINING_URLS_FILENAME
path_save_dir_downloaded_images = DATA_DIR / DOWNLOADED_IMAGES_DIRNAME
number_sample_per_shard = 10_000
image_size = 256
resize_mode = "no"
num_proc = 1
thread_count = 1

download_images(
    path_save_file_image_urls,
    path_save_dir_downloaded_images,
    number_sample_per_shard,
    image_size,
    resize_mode,
    num_proc,
    thread_count,
)
