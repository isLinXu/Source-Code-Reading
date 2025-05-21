import argparse
import logging
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path

from datasets import load_from_disk
from selectolax.parser import HTMLParser


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(description="Extract bannel urls.")
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
    parser.add_argument(
        "--shard_name",
        type=str,
        help="The shard name to process.",
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
    html_dataset = load_from_disk(args.arrow_shard_dir / args.shard_name)["train"]

    css_rule = "[class~='more-link']"

    def filter_doc_with_matching_class(example, css_rule):
        current_html = example["html"]
        tree = HTMLParser(current_html)
        match = tree.css_first(css_rule)
        if match:
            return True
        return False

    html_dataset = html_dataset.filter(partial(filter_doc_with_matching_class, css_rule=css_rule), num_proc=32)

    with open(args.extraction_dir / f"shard_{args.shard_id}_banned_urls.txt", "w") as f:
        f.write("\n".join(html_dataset["url"]))


if __name__ == "__main__":
    args = get_args()
    main(args)
