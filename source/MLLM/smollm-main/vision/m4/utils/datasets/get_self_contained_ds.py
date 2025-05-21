import argparse
import logging
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import datasets
from datasets import Dataset, DatasetDict, load_dataset
from datasets.utils.logging import set_verbosity_info


set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    return args


def extract_bytes_and_format(path: Optional[str], bytes: Optional[bytes]):
    if path is not None:
        with open(path, "rb") as f:
            image_bytes = f.read()
        # (Quentin advice) so that datasets knows how to decode the image based on the filename's extension
        image_format = Path(path).name
    elif bytes is not None:
        image_bytes = bytes
        image_format = None
    return {"path": image_format, "bytes": image_bytes}


def store_bytes_only_for_images(exs, columns_to_change):
    for col_name in columns_to_change:
        new_values = [extract_bytes_and_format(**image) if image is not None else None for image in exs[col_name]]
        exs[col_name] = new_values
    return exs


def change_format(ds: Dataset, columns_to_change: List, batch_size: int, num_proc: int):
    state = {}
    for col_name in columns_to_change:
        state[col_name] = ds.features[col_name]
        ds = ds.cast_column(col_name, datasets.Image(decode=False))
    ds = ds.map(
        partial(store_bytes_only_for_images, columns_to_change=["image"]),
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )
    # Revert the changes regarding the state of the colum
    for col_name in columns_to_change:
        ds = ds.cast_column(col_name, state[col_name])
    return ds


def get_image_columns(ds: Dataset):
    return [
        column_name for column_name, feature_type in ds.features.items() if isinstance(feature_type, datasets.Image)
    ]


def process_ds(ds: Dataset, batch_size: int, num_proc: int):
    logger.info(" ===== Identifying images columns =====")
    columns_to_change = get_image_columns(ds)

    logger.info(" ===== Changing format of images columns =====")
    return change_format(ds, columns_to_change, batch_size=batch_size, num_proc=num_proc)


def process_ds_wrapped(ds: Union[DatasetDict, Dataset], batch_size: int, num_proc: int):
    if isinstance(ds, DatasetDict):
        new_ds = DatasetDict()
        for key in ds.keys():
            new_ds[key] = process_ds(ds[key], batch_size=batch_size, num_proc=num_proc)
        return new_ds
    elif isinstance(ds, Dataset):
        return process_ds(ds, batch_size=batch_size, num_proc=num_proc)
    else:
        raise ValueError(f"dataset is not if a regular type. Type: {type(ds)}")


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(f"** The job is runned with the following arguments: **\n{args}\n **** ")

    logger.info(f" ===== Loading {args.dataset_path} =====")
    ds = load_dataset(str(args.dataset_path))

    logger.info(f"ds info: {ds}")
    ds = process_ds_wrapped(ds, batch_size=args.batch_size, num_proc=args.num_proc)
    logger.info(f"ds_final info: {ds}")

    logger.info(" ===== Saving Final dataset =====")
    logger.info(f"Saving to final dataset at {args.save_path}.")
    tmp_save_path = Path(args.save_path.parent, f"tmp-{args.save_path.name}")
    if len(ds) == 0:
        logger.info("Dataset was empty. Not saving anything.")
    ds.save_to_disk(tmp_save_path)
    tmp_save_path.rename(args.save_path)
    logger.info(" ===== Final dataset saved successfully =====")


if __name__ == "__main__":
    main()
