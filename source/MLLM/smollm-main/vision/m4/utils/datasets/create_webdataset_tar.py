import json
import logging
import math
import os
import shutil
import subprocess
import uuid
from functools import partial

# import PIL.Image
from pathlib import Path
from typing import List, Optional, Union

from datasets import DatasetDict, concatenate_datasets
from pathos.multiprocessing import ProcessingPool as Pool

from m4.training.dataset import load_hf_dataset
from m4.training.types import DatasetTypes
from m4.training.utils import _convert_to_rgb


logger = logging.getLogger(__name__)


def check_img_exception(img):
    try:
        _ = img.convert("RGB")
        return False
    except Exception as e:
        logger.info(e)
        return True


# Utils for web documents
def save_web_document_example_in_files(example, idx, saving_dir):
    example_id = f"{idx}_{str(uuid.uuid4())}"
    saved = 0
    num_img = 0
    file_paths = []

    for i, (text, image) in enumerate(zip(example["texts"], example["images"])):
        if text is not None and text != "":
            text_file_path = saving_dir / f"{example_id}.{i}.text.txt"
            with open(text_file_path, "w") as f:
                f.write(text)
            file_paths.append(text_file_path)
        elif image is not None and image != "":
            num_img += 1
            image = _convert_to_rgb(image)
            if check_img_exception(image):
                logger.info(f"Example {idx} has image with exception")
                continue
            image_path = saving_dir / f"{example_id}.{i}.image.jpeg"
            image.save(image_path, "jpeg")
            saved += 1
            file_paths.append(image_path)
    if saved == 0:
        for file_path in file_paths:
            os.remove(file_path)
        return {"saved": saved, "num_img": num_img}

    if len(file_paths) == 0:
        return {"saved": saved, "num_img": num_img}

    metadata_file_path = saving_dir / f"{example_id}.metadata.txt"
    with open(metadata_file_path, "w") as f:
        f.write("\n".join([path.name for path in file_paths]))
    return {"saved": saved, "num_img": num_img}


def save_web_document_example_in_files_with_num_shards(
    example, idx, saving_dir, num_examples_per_shard, save_shard_prefix
):
    shard_idx = idx // num_examples_per_shard
    saving_dir_shard = saving_dir / f"shard_{save_shard_prefix}{shard_idx}"

    return save_web_document_example_in_files(example, idx, saving_dir_shard)


# Utils for image caption pairs
def save_image_caption_pair_example_in_files(example, idx, saving_dir):
    image = example["image"]
    if image is None:
        logger.info(f"Example {idx} has None as image")
        return {"saved": 0, "num_img": 1}

    image = _convert_to_rgb(image)
    if check_img_exception(image):
        logger.info(f"Example {idx} has image with exception")
        return {"saved": 0, "num_img": 1}

    example_id = f"{idx}_{str(uuid.uuid4())}"
    image_path = saving_dir / f"{example_id}.image.jpeg"
    image.save(image_path, "jpeg")
    text_file_path = saving_dir / f"{example_id}.text.txt"
    with open(text_file_path, "w") as f:
        f.write(example["text"])
    return {"saved": 1, "num_img": 1}


def save_image_caption_pair_example_in_files_with_num_shards(
    example, idx, saving_dir, num_examples_per_shard, save_shard_prefix
):
    shard_idx = idx // num_examples_per_shard
    saving_dir_shard = saving_dir / f"shard_{save_shard_prefix}{shard_idx}"

    return save_image_caption_pair_example_in_files(example, idx, saving_dir_shard)


# Utils for image/question/answer triplets (in particular for specific fine-tuning)
def save_image_question_answer_triplet_example_in_files(example, idx, saving_dir):
    image = example["image"]
    if image is None:
        logger.info(f"Example {idx} has None as image")
        return {"saved": 0, "num_img": 1}

    image = _convert_to_rgb(image)
    if check_img_exception(image):
        logger.info(f"Example {idx} has image with exception")
        return {"saved": 0, "num_img": 1}

    example_id = f"{idx}_{str(uuid.uuid4())}"
    image_path = saving_dir / f"{example_id}.image.jpeg"
    image.save(image_path, "jpeg")
    question_file_path = saving_dir / f"{example_id}.question.txt"
    with open(question_file_path, "w") as f:
        f.write(example["question"])
    answer_file_path = saving_dir / f"{example_id}.answer.txt"
    with open(answer_file_path, "w") as f:
        f.write(example["answer"])
    return {"saved": 1, "num_img": 1}


def save_image_question_answer_triplet_example_in_files_with_num_shards(
    example, idx, saving_dir, num_examples_per_shard, save_shard_prefix
):
    shard_idx = idx // num_examples_per_shard
    saving_dir_shard = saving_dir / f"shard_{save_shard_prefix}{shard_idx}"

    return save_image_question_answer_triplet_example_in_files(example, idx, saving_dir_shard)


# Utils for sft datasets now that they are all under the same format
def save_sft_example_in_files(example, idx, saving_dir):
    example_id = f"{idx}_{str(uuid.uuid4())}"
    saved = 0
    num_img = 0
    file_paths = []

    for i, image in enumerate(example["images"]):
        num_img += 1
        image = _convert_to_rgb(image)
        if check_img_exception(image):
            logger.info(f"Example {idx} has image with exception")
            continue
        if check_img_exception(image):
            logger.info(f"Example {idx} has image with exception")
            continue
        image_path = saving_dir / f"{example_id}.{i}.image.jpeg"
        image.save(image_path, "jpeg")
        saved += 1
        file_paths.append(image_path)

    for i, text in enumerate(example["texts"]):
        text_file_path = saving_dir / f"{example_id}.{i + num_img}.text.txt"
        with open(text_file_path, "w") as f:
            json.dump(text, f)  # Dump the user/assistant dict as a string into a text file
        file_paths.append(text_file_path)

    if len(file_paths) == 0:
        return {"saved": saved, "num_img": num_img}

    metadata_file_path = saving_dir / f"{example_id}.metadata.txt"
    with open(metadata_file_path, "w") as f:
        f.write("\n".join([path.name for path in file_paths]))
    return {"saved": saved, "num_img": num_img}


def save_sft_example_in_files_with_num_shards(example, idx, saving_dir, num_examples_per_shard, save_shard_prefix):
    shard_idx = idx // num_examples_per_shard
    shard_idx_padded = str(shard_idx).zfill(7)
    saving_dir_shard = saving_dir / f"shard_{save_shard_prefix}{shard_idx_padded}"

    return save_sft_example_in_files(example, idx, saving_dir_shard)


# General utils
def save_example_in_files(example, idx, saving_dir, ds_type):
    if ds_type == DatasetTypes.WEB_DOCUMENTS:
        saved = save_web_document_example_in_files(example, idx, saving_dir)
    elif ds_type == DatasetTypes.IMAGE_CAPTION_PAIRS:
        saved = save_image_caption_pair_example_in_files(example, idx, saving_dir)
    elif (ds_type == DatasetTypes.DOCVQA) or (ds_type == DatasetTypes.VQAV2_TASK_FINETUNING):
        saved = save_image_question_answer_triplet_example_in_files(example, idx, saving_dir)
    elif ds_type == DatasetTypes.SFT:
        saved = save_sft_example_in_files(example, idx, saving_dir)
    else:
        raise ValueError(f"Unsupported dataset type {ds_type}")
    return saved


def save_example_in_files_with_num_shards(
    example, idx, saving_dir, ds_type, num_examples_per_shard, save_shard_prefix
):
    if ds_type == DatasetTypes.WEB_DOCUMENTS:
        saved = save_web_document_example_in_files_with_num_shards(
            example, idx, saving_dir, num_examples_per_shard, save_shard_prefix
        )
    elif ds_type == DatasetTypes.IMAGE_CAPTION_PAIRS:
        saved = save_image_caption_pair_example_in_files_with_num_shards(
            example, idx, saving_dir, num_examples_per_shard, save_shard_prefix
        )
    elif (ds_type == DatasetTypes.DOCVQA) or (ds_type == DatasetTypes.VQAV2_TASK_FINETUNING):
        saved = save_image_question_answer_triplet_example_in_files_with_num_shards(
            example, idx, saving_dir, num_examples_per_shard, save_shard_prefix
        )
    elif ds_type == DatasetTypes.SFT:
        saved = save_sft_example_in_files_with_num_shards(
            example, idx, saving_dir, num_examples_per_shard, save_shard_prefix
        )
    else:
        raise ValueError(f"Unsupported dataset type {ds_type}")
    return saved


def export_dataset_all_shard_idx_to_tar(
    hf_datasets_paths: List[Union[str, Path]],
    saving_dir: Union[str, Path],
    ds_type: DatasetTypes,
    num_examples_per_shard: int,
    s3_uri: Optional[str] = None,
    num_proc: Optional[int] = None,
    min_num_shards: Optional[int] = None,
    save_shard_prefix: str = "",
    shard_idx: Optional[int] = None,
    save_shard_idx: Optional[str] = None,
):
    if save_shard_idx is not None:
        raise NotImplementedError("Use of `save_shard_idx` has been deprecated.")

    if num_proc is None:
        # by default the value of num_proc will be the minimum between 6 and the number of cpus
        num_proc = min(6, os.cpu_count())

    logger.info("Start loading the dataset")
    dataset_list = [load_hf_dataset(str(hf_dataset_path)) for hf_dataset_path in hf_datasets_paths]
    ds = concatenate_datasets(dataset_list)

    if isinstance(ds, DatasetDict):
        raise ValueError("DatasetDict not supported")

    if num_examples_per_shard is None:
        num_shards = 1
    else:
        num_shards = math.ceil(len(ds) / num_examples_per_shard)

    if min_num_shards is not None and num_shards < min_num_shards:
        num_examples_per_shard = len(ds) // min_num_shards
        logger.info(
            f"Number of examples per shard is too low, setting it to {num_examples_per_shard}. Without this, the"
            f" number of shards would be {num_shards} which is lower than the minimum number of shards"
            f" {min_num_shards}"
        )
        num_shards = min_num_shards

    num_examples_per_shard = len(ds) // num_shards
    num_shards = len(ds) // num_examples_per_shard
    logger.info(f"Number of shards: {num_shards} and number of examples per shard: {num_examples_per_shard}")

    if shard_idx is None:
        for idx in range(num_shards + 1):
            idx_leading_0s = str(idx).zfill(7)
            saving_dir_shard = saving_dir / f"shard_{save_shard_prefix}{idx_leading_0s}"
            saving_dir_shard.mkdir(parents=True, exist_ok=True)

        logger.info(f"The dataset has {len(ds)} examples and the columns are {ds.column_names}")
        ds_saved = ds.map(
            partial(
                save_example_in_files_with_num_shards,
                saving_dir=saving_dir,
                ds_type=ds_type,
                num_examples_per_shard=num_examples_per_shard,
                save_shard_prefix=save_shard_prefix,
            ),
            with_indices=True,
            num_proc=num_proc,
            load_from_cache_file=False,
            remove_columns=ds.column_names,
        )
    else:
        ds = ds.shard(num_shards=num_shards, index=shard_idx)
        saving_dir_shard = saving_dir / f"shard_{save_shard_prefix}{shard_idx}"
        saving_dir_shard.mkdir(parents=True, exist_ok=True)

        logger.info(f"The dataset has {len(ds)} examples and the columns are {ds.column_names}")
        ds_saved = ds.map(
            partial(save_example_in_files, saving_dir=saving_dir_shard, ds_type=ds_type),
            with_indices=True,
            num_proc=num_proc,
            load_from_cache_file=False,
        )
    finished_file_path = saving_dir / f"shard_{save_shard_prefix}_finished.txt"
    finished_file_path.touch()

    num_images = sum(ds_saved["num_img"])
    num_saved = sum(ds_saved["saved"])
    logger.info(
        f"Shard {save_shard_prefix} has {num_images} images and out of {num_saved} saved"
        f" ({num_saved / num_images * 100:.2f}"
    )

    def tar_shard_and_send_to_s3(saving_dir_shard):
        # check if the shard exists and is not empty
        if not os.path.exists(saving_dir_shard) or not os.listdir(saving_dir_shard):
            return

        tar_file = saving_dir_shard.parent / f"{saving_dir_shard.name}.tar"

        # Create tar file
        tar_cmd = ["tar", "--sort=name", "-cf", str(tar_file), "-C", str(saving_dir_shard), "."]
        subprocess.run(tar_cmd, check=True)

        # Remove original directory
        shutil.rmtree(saving_dir_shard, ignore_errors=True)

        # Upload to S3 if necessary
        if s3_uri is not None:
            s3_uri_file = f"{s3_uri}/{tar_file.name}"
            sync_cmd = ["s5cmd", "cp", str(tar_file), s3_uri_file]
            subprocess.run(sync_cmd, check=True)

        return

    if shard_idx is None:
        args_pool = [saving_dir / f"shard_{save_shard_prefix}{str(idx).zfill(7)}" for idx in range(num_shards + 1)]
    else:
        args_pool = [saving_dir / f"shard_{save_shard_prefix}{shard_idx}"]
    pool = Pool(num_proc)
    results = pool.amap(tar_shard_and_send_to_s3, args_pool)
    results = results.get()

    return 0


def export_dataset_to_tar(
    hf_datasets_paths: List[Union[str, Path]],
    saving_dir: Union[str, Path],
    ds_type: DatasetTypes,
    num_examples_per_shard: int,
    num_proc: Optional[int] = None,
):
    return export_dataset_all_shard_idx_to_tar(
        hf_datasets_paths=hf_datasets_paths,
        saving_dir=saving_dir,
        ds_type=ds_type,
        num_examples_per_shard=num_examples_per_shard,
        num_proc=num_proc,
    )


def export_dataset_shard_idx_to_tar(
    hf_datasets_paths: List[Union[str, Path]],
    saving_dir: Union[str, Path],
    ds_type: DatasetTypes,
    num_examples_per_shard: int,
    s3_uri: Optional[str] = None,
    num_proc: Optional[int] = None,
    shard_idx: int = 0,
    min_num_shards: Optional[int] = None,
    save_shard_idx: Optional[str] = None,
):
    logger.warning(
        "`export_dataset_shard_idx_to_tar` is deprecated, please favor `export_dataset_all_shard_idx_to_tar`."
    )
    return export_dataset_all_shard_idx_to_tar(
        hf_datasets_paths=hf_datasets_paths,
        saving_dir=saving_dir,
        ds_type=ds_type,
        num_examples_per_shard=num_examples_per_shard,
        s3_uri=s3_uri,
        num_proc=num_proc,
        min_num_shards=min_num_shards,
        save_shard_prefix="",
        shard_idx=shard_idx,
        save_shard_idx=save_shard_idx,
    )


if __name__ == "__main__":
    hf_datasets_paths = ["HuggingFaceM4/tmp-pmd-synthetic-testing:100.unique"]
    saving_dir = Path("/home/lucile/data/tmp-pmd-synthetic-testing-100-unique-tar")
    num_examples_per_shard = 20
    num_proc = 32
    ds_type = DatasetTypes.IMAGE_CAPTION_PAIRS
    export_dataset_to_tar(hf_datasets_paths, saving_dir, ds_type, num_examples_per_shard, num_proc)
