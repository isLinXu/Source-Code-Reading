import argparse
import logging
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


def get_args():
    parser = argparse.ArgumentParser(description="Image/text pair filtering.")
    parser.add_argument(
        "--path_image_text_pair_dataset",
        type=str,
        default="./large_files/laion_2b_en_100k",
        help="Path of the dataset containing the image/text pairs.",
    )
    parser.add_argument(
        "--path_save_dir_image_text_pair_dataset_filtered",
        type=str,
        default="./large_files/laion_2b_en_100k_filtered",
        help="The directory to save the filtered image/text pair dataset.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of processes to use for the multiprocessing.",
    )
    parser.add_argument(
        "--path_config_filter_image_text_pairs",
        type=str,
        default="./m4/sourcing/data_collection/configs/config_filter_laion_pairs.yaml",
        help="The path of the config file containing the filtering parameters.",
    )
    parser.add_argument(
        "--path_common_words",
        type=str,
        default="./large_files/words_oscar.json",  # Find at s3://m4-datasets/trash/common_words.json
        help="The path of the file containing the common words.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    logger.info("Starting loading the image/text pair dataset")
    image_text_pair_dataset = load_from_disk(args.path_image_text_pair_dataset)
    logger.info("Finished loading the image/text pair dataset")

    with open(args.path_config_filter_image_text_pairs) as f:
        filtering_params = yaml.load(f, Loader=yaml.FullLoader)

    image_text_pair_filtering = LaionPairFiltering(
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
        path_common_words=args.path_common_words,
        common_word_ratio_min_cutoff=filtering_params["common_word_ratio_min_cutoff"],
    )

    logger.info("Starting filtering the image/text pair dataset")
    image_text_pair_dataset_filtered = image_text_pair_dataset.filter(
        image_text_pair_filtering, num_proc=args.num_proc
    )
    logger.info("Finished filtering the image/text pair dataset")

    logger.info("Starting saving the filtered image/text pair dataset")
    image_text_pair_dataset_filtered.save_to_disk(args.path_save_dir_image_text_pair_dataset_filtered)
    logger.info("Finished saving the filtered image/text pair dataset")
