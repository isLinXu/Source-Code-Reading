import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path

from datasets import load_from_disk

from m4.sourcing.data_collection.processors import DOMTreeSimplificator, PreExtractionSimplificator
from m4.sourcing.data_collection.processors.web_document_extractor import html_to_web_documents


# from m4.sourcing.data_collection.utils import InterestingAttributesSetCategory


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description="Extract web documents.")
    # ---------------- Path args ----------------
    parser.add_argument(
        "--arrow_shard_dir",
        type=Path,
        help="The directory containing the wikipedia html enterprise dataset.",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        help="The shard to process.",
    )
    parser.add_argument(
        "--extraction_dir",
        type=Path,
        help="The directory to save all the extracted artifacts",
    )
    # ---------------- Extraction args ----------------
    parser.add_argument(
        "--num_proc",
        type=int,
        default=cpu_count(),
        help="Number of processes to use for the multiprocessing.",
    )
    args = parser.parse_args()
    return args


def main(args):
    html_dataset = load_from_disk(args.arrow_shard_dir / f"shard_{args.shard_id}")

    dom_tree_simplificator = DOMTreeSimplificator(
        strip_multiple_linebreaks=True,
        strip_multiple_spaces=True,
        remove_html_comments=True,
        replace_line_break_tags=True,
        unwrap_tags=True,
        strip_tags=True,
        strip_special_divs=True,
        remove_dates=True,
        remove_empty_leaves=True,
        unnest_nodes=True,
        remake_tree=True,
        css_rules=[
            "[class~='footer']",
            "[class~='site-info']",
        ],
        css_rules_replace_with_text={"[class~='more-link']": "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED"},
    )
    pre_extraction_simplificator = PreExtractionSimplificator(
        only_text_image_nodes=True,
        format_texts=True,
        merge_consecutive_text_nodes=True,
    )

    dataset = html_to_web_documents(
        html_dataset,
        dom_tree_simplificator,
        pre_extraction_simplificator,
        args.num_proc,
        remove_columns=False,
        html_column_name="html",
        url_column_name="document_url",
        extraction_suffix="_v2",
    )

    path_save_dir_dataset = args.extraction_dir / args.dataset_name
    logger.info("Starting saving the dataset")
    dataset.save_to_disk(path_save_dir_dataset)
    logger.info("Finished saving the dataset")


if __name__ == "__main__":
    args = get_args()
    main(args)
