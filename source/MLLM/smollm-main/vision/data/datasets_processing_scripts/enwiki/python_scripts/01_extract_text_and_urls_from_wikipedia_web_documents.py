import argparse
import logging
from multiprocessing import cpu_count
from pathlib import Path

from datasets import load_dataset

from m4.sourcing.data_collection.processors import DOMTreeSimplificator, PreExtractionSimplificator
from m4.sourcing.data_collection.processors.web_document_extractor import html_to_web_documents
from m4.sourcing.data_collection.utils import InterestingAttributesSetCategory


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
        "--wikipedia_html_enterprise_dir",
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
    html_dataset = load_dataset(
        "SaulLu/wikipedia_html_enterprise",
        shard=args.shard_id,
        data_dir=args.wikipedia_html_enterprise_dir,
        split="train",
    )

    dom_tree_simplificator = DOMTreeSimplificator(
        strip_multiple_linebreaks=True,
        strip_multiple_spaces=True,
        remove_html_comments=True,
        replace_line_break_tags=True,
        unwrap_tags=True,
        strip_tags=False,
        strip_special_divs=True,
        remove_dates=True,
        remove_empty_leaves=True,
        unnest_nodes=True,
        remake_tree=True,
        preserve_img_children=True,
        remove_everything_after_node_id=["Notes", "References", "See_also", "Further_reading", "External_links"],
        css_rules=[
            "[class~='locmap']",
            "[class~='reference']",
            "[role='presentation']",
            "[role~='note']",
        ],
        interesting_attributes_set_cat=InterestingAttributesSetCategory.WIKIPEDIA,
    )
    pre_extraction_simplificator = PreExtractionSimplificator(
        only_text_image_nodes=True,
        format_texts=True,
        merge_consecutive_text_nodes=True,
        interesting_attributes_set_cat=InterestingAttributesSetCategory.WIKIPEDIA,
    )

    dataset = html_to_web_documents(html_dataset, dom_tree_simplificator, pre_extraction_simplificator, args.num_proc)

    path_save_dir_dataset = args.extraction_dir / args.dataset_name
    logger.info("Starting saving the dataset")
    dataset.save_to_disk(path_save_dir_dataset)
    logger.info("Finished saving the dataset")


if __name__ == "__main__":
    args = get_args()
    main(args)
