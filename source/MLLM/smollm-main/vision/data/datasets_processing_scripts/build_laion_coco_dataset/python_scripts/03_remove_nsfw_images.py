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

PATH_DS_LAION_COCO_S3 = (
    f"s3://m4-datasets-us-east-1/LAION_data/laion_coco_dataset_optoutrmv/{IDX_JOB}/ds_laion_coco_optoutrmv/"
)
PATH_DS_LAION_COCO_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "ds_laion_coco")

NUM_PROC = 10

PATH_SAVE_DISK_DS_LAION_COCO_NSFW_FILTERED = os.path.join(PATH_SAVE_DISK_TMP_FILES, "dl_laion_coco_nsfwfiltered")
PATH_SAVE_S3_DS_LAION_COCO_NSFW_FILTERED = (
    f"s3://m4-datasets-us-east-1/LAION_data/laion_coco_dataset_optoutrmv_nsfwfiltered/{IDX_JOB}/"
)


def filter_nsfw(example):
    nsfw = json.loads(example["meta"])["NSFW"]
    if (nsfw == "UNSURE") or (nsfw == "NSFW"):
        return False
    return True


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting loading the LAION COCO dataset")
    command_sync_s3 = f"aws s3 sync {PATH_DS_LAION_COCO_S3} {PATH_DS_LAION_COCO_LOCAL}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    ds_laion_coco = load_from_disk(PATH_DS_LAION_COCO_LOCAL)
    num_pairs_before_filtering = ds_laion_coco.num_rows
    logger.info("Finished loading the LAION COCO dataset")

    logger.info("Starting removing the NSFW images")
    ds_laion_coco = ds_laion_coco.filter(filter_nsfw, num_proc=NUM_PROC)
    logger.info("Finished removing the NSFW images")

    logger.info("Starting saving the LAION COCO dataset with the NSFW images removed")
    ds_laion_coco.save_to_disk(PATH_SAVE_DISK_DS_LAION_COCO_NSFW_FILTERED, num_proc=NUM_PROC)

    command_sync_s3 = (
        f"aws s3 sync {PATH_SAVE_DISK_DS_LAION_COCO_NSFW_FILTERED} {PATH_SAVE_S3_DS_LAION_COCO_NSFW_FILTERED}"
    )
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
    logger.info("Finished saving the LAION COCO dataset with the NSFW images removed")

    logger.info(
        "Number of pairs in the LAION COCO dataset before the filtering of the NSFW images:"
        f" {num_pairs_before_filtering}"
    )
    logger.info(
        f"Number of pairs in the LAION COCO dataset after the filtering of the NSFW images: {ds_laion_coco.num_rows}"
    )

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")
