import argparse
import os
from typing import Dict

from datasets import load_from_disk
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_example_with_none_image(batch: Dict) -> Dict:
    result_bools = [im is None for im in batch["image"]]
    return result_bools


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count the number of rows with `image==None`")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        required=True,
        help="Path to folder containing the dataset. Example: `$cnw_ALL_CCFRSCRATCH/general_pmd/image/wit`",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processed for multiprocessing (in particular calls to `map`s).",
    )
    args = parser.parse_args()

    for split_name in os.listdir(args.dataset_folder):
        split_size = 0
        split_filtered_size = 0
        for shard_name in sorted(os.listdir(os.path.join(args.dataset_folder, split_name))):
            print(f"Doing: {os.path.join(args.dataset_folder, split_name, shard_name)}")
            shard = load_from_disk(os.path.join(args.dataset_folder, split_name, shard_name))
            shard_no_ims = shard.filter(
                get_example_with_none_image,
                batched=True,
                batch_size=250,
                num_proc=args.num_proc,
            )
            split_size += len(shard)
            split_filtered_size += len(shard_no_ims)
        print(f"Dataset: {args.dataset_folder}")
        print(f"---Split name: {split_name}")
        print(f"------Initial size: {split_size}")
        print(f"------Nb instances without image: {split_filtered_size}")
        print(f"------Last 10 instances with no images: {shard_no_ims[-10:]}\n")
