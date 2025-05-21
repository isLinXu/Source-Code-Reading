import json
import logging
import os
import sys

from datasets import load_from_disk
from PIL import Image, ImageFile


# Useful to avoid DecompressionBombError and truncated image error
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IDX_JOB = sys.argv[1]
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

MAX_NUM_RETRIES_SYNC = 3

PATH_DS_LAION_S3 = f"s3://m4-datasets/LAION_data/laion_dataset_filtered_dedup/{IDX_JOB}"
PATH_DS_LAION_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "ds_laion")

PATH_SAVE_DISK_LAION_URLS = os.path.join(PATH_SAVE_DISK_TMP_FILES, "laion_urls.json")
PATH_SAVE_S3_LAION_URLS = f"s3://m4-datasets/LAION_data/urls_laion_dataset_filtered_dedup/{IDX_JOB}/laion_urls.json"


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting loading the LAION dataset")
    command_sync_s3 = f"aws s3 sync {PATH_DS_LAION_S3} {PATH_DS_LAION_LOCAL}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    ds_laion = load_from_disk(PATH_DS_LAION_LOCAL)
    logger.info("Finished loading the LAION dataset")

    logger.info("Starting retrieving the URLs")
    metadata = ds_laion["meta"]
    urls = [json.loads(meta)["url"] for meta in metadata]
    logger.info("Finished retrieving the URLs")

    logger.info("Starting saving the URLs of LAION")
    with open(PATH_SAVE_DISK_LAION_URLS, "w") as f:
        json.dump(urls, f)

    command_sync_s3 = f"aws s3 cp {PATH_SAVE_DISK_LAION_URLS} {PATH_SAVE_S3_LAION_URLS}"
    os.system(command_sync_s3)
    logger.info("Starting saving the URLs of LAION")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")
