import argparse
import logging
from pathlib import Path

from datasets import set_caching_enabled

from m4.training.types import DatasetTypes
from m4.utils.datasets.create_webdataset_tar import export_dataset_shard_idx_to_tar


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

set_caching_enabled(False)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saving_dir", type=Path, required=True)
    parser.add_argument("--num_examples_per_shard", type=int, required=True)
    parser.add_argument("--num_proc", type=int, required=True)
    parser.add_argument("--shard_idx", type=int, required=True)
    parser.add_argument("--min_num_shards", type=int)
    args = parser.parse_args()
    return args


def main(args):
    ds_type = DatasetTypes.LLaVA

    export_dataset_shard_idx_to_tar(
        hf_datasets_paths=["HuggingFaceM4/LLaVAR-Instruct-16K:train"],
        saving_dir=args.saving_dir,
        ds_type=ds_type,
        num_examples_per_shard=args.num_examples_per_shard,
        num_proc=args.num_proc,
        shard_idx=args.shard_idx,
        min_num_shards=args.min_num_shards,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
