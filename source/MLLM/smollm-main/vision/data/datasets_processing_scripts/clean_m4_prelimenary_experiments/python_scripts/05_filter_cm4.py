import json
from pathlib import Path

import tqdm
from datasets import concatenate_datasets, load_from_disk


# from datasets.utils.logging import set_verbosity_info


# set_verbosity_info()


def load_html_ds(dir_path: Path):
    html_ds_list = []
    for arrow_shard_dir in tqdm.tqdm(dir_path.iterdir()):
        if not arrow_shard_dir.is_dir() or not arrow_shard_dir.name.startswith("c4-"):
            print(f"Skipping {arrow_shard_dir}")
            continue
        print(f"Loading {arrow_shard_dir}")
        html_ds = load_from_disk(arrow_shard_dir)["train"]
        html_ds_list.append(html_ds)
    html_ds = concatenate_datasets(html_ds_list)
    return html_ds


def load_processed_shard(dir_path: Path, shard_idx: int):
    shard_dir = dir_path / f"shard_{shard_idx}"
    processed_shard = load_from_disk(shard_dir)
    return processed_shard


def add_doc_url(example):
    metadata = json.loads(example["metadata"])
    document_url = None
    for meta in metadata:
        if meta is not None:
            document_url = meta["document_url"]
            break
    example["document_url"] = document_url
    return example


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir_path", type=Path, required=True)
    parser.add_argument("--output_dir_path", type=Path, required=True)
    parser.add_argument("--shard_idx", type=int, required=True)
    parser.add_argument("--num_proc", type=int, required=True)
    parser.add_argument("--banned_urls_dir_path", type=Path, required=True)
    return parser.parse_args()


def main():
    args = get_args()

    save_path = args.output_dir_path / f"shard_{args.shard_idx}"
    if save_path.exists():
        print(f"==================== Dataset already exists at {save_path} ====================")
        return

    # Load processed shard
    processed_shard_ds = load_processed_shard(args.processed_dir_path, args.shard_idx)
    print(f"Loaded {len(processed_shard_ds)} processed examples")

    # Add document url column
    processed_shard_ds = processed_shard_ds.map(add_doc_url, num_proc=args.num_proc)

    # Load banned urls into a set
    banned_urls = set()
    for banned_urls_file in args.banned_urls_dir_path.iterdir():
        with open(banned_urls_file, "r") as f:
            for line in f:
                banned_urls.add(line.strip())

    num_exemple_before = len(processed_shard_ds)

    # Filter out banned urls
    processed_shard_ds = processed_shard_ds.filter(
        lambda example: example["document_url"] not in banned_urls, num_proc=args.num_proc
    )

    num_exemple_after = len(processed_shard_ds)
    print(
        f"Filtered out {num_exemple_before - num_exemple_after} examples. {num_exemple_after} examples left out of"
        f" {num_exemple_before}."
    )
    processed_shard_ds.save_to_disk(save_path)
    print(f"==================== Saved to {save_path} ====================")


if __name__ == "__main__":
    main()
