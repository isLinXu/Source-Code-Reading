import logging
import os
import sys
from multiprocessing import cpu_count

import yaml
from datasets import load_from_disk
from PIL import Image, ImageFile

from m4.sourcing.data_collection.processors.laion_pair_filtering import LaionPairFiltering
from m4.sourcing.data_collection.utils import SPECIAL_CHARACTERS


# Useful to avoid DecompressionBombError
Image.MAX_IMAGE_PIXELS = None
# Load even truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


IDX_JOB = int(sys.argv[1])
PATH_SAVE_DISK_TMP_FILES = f"/scratch/storage_hugo_{IDX_JOB}/"

PATH_LAION_DATASET_S3 = f"s3://m4-datasets/LAION_data/laion_raw_hf_dataset/{IDX_JOB}"
PATH_LAION_DATASET_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "laion_dataset")

PATH_CONFIG_FILTERING = "./m4/sourcing/data_collection/configs/config_filter_laion_pairs.yaml"
PATH_COMMON_WORDS_S3 = "s3://m4-datasets/trash/common_words.json"
PATH_COMMON_WORDS_LOCAL = os.path.join(PATH_SAVE_DISK_TMP_FILES, "common_words.json")
NUM_PROC = cpu_count()

PATH_SAVE_DISK_LAION_DATASET_FILTERED = os.path.join(PATH_SAVE_DISK_TMP_FILES, "laion_dataset_filtered")
PATH_SAVE_S3_LAION_DATASET_FILTERED = os.path.join("s3://m4-datasets/LAION_data/laion_dataset_filtered", str(IDX_JOB))


if __name__ == "__main__":
    if os.path.exists(PATH_SAVE_DISK_TMP_FILES):
        os.system(f"rm -rf {PATH_SAVE_DISK_TMP_FILES}")
    os.system(f"mkdir {PATH_SAVE_DISK_TMP_FILES}")

    logger.info("Starting downloading and loading the LAION dataset")
    command_sync_s3 = f"aws s3 sync {PATH_LAION_DATASET_S3} {PATH_LAION_DATASET_LOCAL}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)

    laion_dataset = load_from_disk(PATH_LAION_DATASET_LOCAL)
    num_pairs_before_filtering = laion_dataset.num_rows
    logger.info("Finished downloading and loading the LAION dataset")

    logger.info("Starting filtering the LAION dataset")
    command_sync_s3 = f"aws s3 cp {PATH_COMMON_WORDS_S3} {PATH_COMMON_WORDS_LOCAL}"
    os.system(command_sync_s3)

    with open(PATH_CONFIG_FILTERING) as f:
        filtering_params = yaml.load(f, Loader=yaml.FullLoader)

    laion_pair_filtering = LaionPairFiltering(
        cond_check_size_image=filtering_params["cond_check_size_image"],
        original_width_min_cutoff=filtering_params["original_width_min_cutoff"],
        original_width_max_cutoff=filtering_params["original_width_max_cutoff"],
        original_height_min_cutoff=filtering_params["original_height_min_cutoff"],
        original_height_max_cutoff=filtering_params["original_height_max_cutoff"],
        aspect_ratio_max_cutoff=filtering_params["aspect_ratio_max_cutoff"],
        cond_check_number_words=filtering_params["cond_check_number_words"],
        strip_characters=SPECIAL_CHARACTERS,
        number_words_min_cutoff=filtering_params["number_words_min_cutoff"],
        number_words_max_cutoff=filtering_params["number_words_max_cutoff"],
        cond_check_word_repetition_ratio=filtering_params["cond_check_word_repetition_ratio"],
        word_repetition_length=filtering_params["word_repetition_length"],
        word_repetition_max_cutoff=filtering_params["word_repetition_max_cutoff"],
        cond_check_special_character_ratio=filtering_params["cond_check_special_character_ratio"],
        special_character_ratio_max_cutoff=filtering_params["special_character_ratio_max_cutoff"],
        cond_check_common_word_ratio=filtering_params["cond_check_common_word_ratio"],
        path_common_words=PATH_COMMON_WORDS_LOCAL,
        common_word_ratio_min_cutoff=filtering_params["common_word_ratio_min_cutoff"],
    )
    laion_dataset = laion_dataset.filter(laion_pair_filtering, num_proc=NUM_PROC)
    logger.info("Finished filtering the LAION dataset")

    logger.info("Starting saving the LAION dataset filtered")
    laion_dataset.save_to_disk(PATH_SAVE_DISK_LAION_DATASET_FILTERED, num_proc=NUM_PROC)

    command_sync_s3 = f"aws s3 sync {PATH_SAVE_DISK_LAION_DATASET_FILTERED} {PATH_SAVE_S3_LAION_DATASET_FILTERED}"
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    os.system(command_sync_s3)
    logger.info("Finished saving the LAION dataset filtered")

    logger.info(f"Number of pairs in the LAION dataset before the filtering: {num_pairs_before_filtering}")
    logger.info(f"Number of pairs in the LAION dataset after the filtering: {laion_dataset.num_rows}")

    logger.info("Starting deleting the tmp files")
    os.system(f"rm -r {PATH_SAVE_DISK_TMP_FILES}")
    logger.info("Finished deleting the tmp files")
