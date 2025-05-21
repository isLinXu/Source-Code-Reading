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

PATH_OPT_OUT_URLS_S3 = (  # Made by converting s3://m4-datasets-us-east-1/LAION_data/ds_opt_out_image_urls_laion_coco/ to a list and saving in json
    "s3://m4-datasets-us-east-1/LAION_data/list_opt_out_image_urls_laion_coco_09_01_2024.json"
)
PATH_OPT_OUT_URLS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "list_opt_out.json")

PATH_DS_LAION_COCO_S3 = f"s3://m4-datasets-us-east-1/LAION_data/laion_coco_dataset/{IDX_JOB}"
PATH_DS_LAION_COCO_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "ds_laion_coco")

NUM_PROC = 22

PATH_SAVE_DISK_DS_LAION_COCO_OPT_OUT_FILTERED = os.path.join(PATH_SAVE_DISK_TMP_FILES, "ds_laion_coco_optoutrmv")
PATH_SAVE_S3_DS_LAION_COCO_OPT_OUT_FILTERED = (
    f"s3://m4-datasets-us-east-1/LAION_data/laion_coco_dataset_optoutrmv/{IDX_JOB}/"
)


class OptOutFiltering:
    def __init__(self, path_opt_out_image_urls):
        self.path_opt_out_image_urls = path_opt_out_image_urls
        with open(path_opt_out_image_urls) as f:
            self.opt_out_image_urls = set(json.load(f))

    def __call__(self, example):
        url = json.loads(example["meta"])["url"]
        if url in self.opt_out_image_urls:
            return False
        return True

    def __reduce__(self):
        return self.__class__, (self.path_opt_out_image_urls,)


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading the list of opt out image urls")
    command_sync_s3 = f"aws s3 cp {PATH_OPT_OUT_URLS_S3} {PATH_OPT_OUT_URLS_LOCAL}"
    os.system(command_sync_s3)
    logger.info("Finished downloading the list of opt out image urls")

    logger.info("Starting loading the LAION COCO dataset")
    command_sync_s3 = f"s5cmd sync {PATH_DS_LAION_COCO_S3}/* {PATH_DS_LAION_COCO_LOCAL}"
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)

    ds_laion_coco = load_from_disk(PATH_DS_LAION_COCO_LOCAL)
    num_pairs_before_filtering = ds_laion_coco.num_rows
    logger.info("Finished loading the LAION COCO dataset")

    logger.info("Starting removing the opt out images")
    opt_out_filtering = OptOutFiltering(path_opt_out_image_urls=PATH_OPT_OUT_URLS_LOCAL)
    ds_laion_coco = ds_laion_coco.filter(opt_out_filtering, num_proc=NUM_PROC)
    logger.info("Finished removing the opt out images")

    logger.info("Starting saving the LAION COCO dataset with the opt out images removed")
    ds_laion_coco.save_to_disk(PATH_SAVE_DISK_DS_LAION_COCO_OPT_OUT_FILTERED, num_proc=NUM_PROC)

    command_sync_s3 = (
        f"s5cmd sync {PATH_SAVE_DISK_DS_LAION_COCO_OPT_OUT_FILTERED} {PATH_SAVE_S3_DS_LAION_COCO_OPT_OUT_FILTERED}"
    )
    for _ in range(MAX_NUM_RETRIES_SYNC):
        os.system(command_sync_s3)
    logger.info("Finished saving the LAION COCO dataset with the opt out images removed")

    logger.info(
        "Number of pairs in the LAION COCO dataset before the filtering of the opt out images:"
        f" {num_pairs_before_filtering}"
    )
    logger.info(
        "Number of pairs in the LAION COCO dataset after the filtering of the opt out images:"
        f" {ds_laion_coco.num_rows}"
    )

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")
