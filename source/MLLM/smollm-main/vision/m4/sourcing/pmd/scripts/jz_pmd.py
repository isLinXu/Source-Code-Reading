import argparse
import math
import os

from datasets import DatasetDict, load_dataset

from m4.utils.datasets.get_self_contained_ds import process_ds_wrapped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create arrow files for subsets of image PMD - JZ version/")
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="Should be either `jz_conceptual_captions` or `jz_wit`."
    )
    parser.add_argument("--loading_script_path", type=str, required=True, help="Path to the loading script.")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processed for multiprocessing (in particular calls to `map`s).",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name.replace("jz_", "")

    dataset = load_dataset(args.loading_script_path)

    print("Start converting the images to bytes.")
    dataset = process_ds_wrapped(dataset, batch_size=1_000, num_proc=args.num_proc)

    print("Start saving shards.")
    if isinstance(dataset, DatasetDict):
        for split_name, dset in dataset.items():
            nb_of_shards = math.ceil(len(dset) / 50_000)
            shards = [dset.shard(num_shards=nb_of_shards, index=i, contiguous=True) for i in range(nb_of_shards)]
            for i, shard in enumerate(shards):
                shard.save_to_disk(
                    f"{os.environ['cnw_ALL_CCFRSCRATCH']}/general_pmd/image/{dataset_name}/{split_name}/{i:05}-{nb_of_shards:05}"
                )
    else:
        raise ValueError(f"`datasets` is of type {type(dataset)} which is not supported yet.")
