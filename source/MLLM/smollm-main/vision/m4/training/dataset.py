"""
This file defines the dataloader logic.
"""
import copy
import inspect
import logging
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import torch
import webdataset as wds
from accelerate.state import AcceleratorState
from PIL import Image, ImageFile
from torch.utils.data import Sampler

from m4.training.config import DataParams, DatasetParams, Parameters
from m4.training.dataset_utils import check_webdataset_command, get_webdataset
from m4.training.packing import (
    split_pack_and_pad_iqa_finetuning,
    split_pack_and_pad_ocr,
    split_pack_and_pad_pairs,
    split_pack_and_pad_sft,
    split_pack_and_pad_webdocs,
)
from m4.training.types import DatasetNames, DatasetTypes


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

"""

Possible dataloader/dataset nestings

DataLoaderForIterableWrapperDataset
  CustomChainDataset
    IterableWrapperDataset

DataLoaderForMappedWrapperDataset
  MapWrapperDataset

"""


# TODO(siddk) :: This file needs to be cleaned up a bit?
"""
dataset[idx]:
    {
        'images':[
            <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=256x256 at 0x7FD503595350>,
            None,
            None,
            None,
            None,
            <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=256x256 at 0x7FD503595D90>,
            <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=256x256 at 0x7FD503595E90>,
            <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=256x256 at 0x7FD503595F90>,
            <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=256x256 at 0x7FD5035930D0>,
            <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=256x256 at 0x7FD503595D50>,
            <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=256x256 at 0x7FD503593290>,
            ],
        'texts':[
            None,
            'VALMAS COMMERCE',
            'VALMAS SA imports top quality products and materials that meet the requirements of modern construction. The goal of our company is to provide material and technical know-how for reliable solutions in the field of construction, renovation and repair. Raw materials of high-quality for every particular need, which also remain environmentally sensitive. Investing continually in enhancing our infrastructure and with a steadily growing track record over the years, we are in a position to meet your every need. We will be glad to welcome you to our 500m² exhibition to choose from a range of quality materials the ones that best match your own requirements.',
            'FLOORING – PAINTS\n\nQuality and great variety in paneling materials for classic and traditional applications of modern residential aesthetics.\n\nFLOORING: With knowledge and experience in construction, we are able to propose solutions that combine the available budget with the specific demands for functionality and aesthetics. Wooden floors, ceramic tiles, marble, stone, as well as construction flooring, such as colored or special cement mortars, offer classic or traditional choices, following the original and modern residential aesthetic result.\n\n·'
            'Partners List',
            None,
            None,
            None,
            None,
            None,
            None,
        ]
    }
"""


def to_tensor(batch):
    for key in batch:
        if not torch.is_tensor(batch[key]):
            batch[key] = torch.tensor(batch[key])
    return batch


# We don't want collate_fn to add extra dim --> can't use a lambda because of `multiprocessing` pickle requirements!
def simple_collate(x):
    return x[0]


def get_mapper(
    tokenizer,
    image_transform,
    dataset_type: DatasetTypes,
    image_seq_len: int,
    max_seq_len: int = 256,
    max_num_images: int = 5,
    max_image_size: int = 384,
    vision_encoder_max_image_size: int = 384,
    pre_split_scale_up_max=1.0,
    pre_split_scale_up_frequency=0.0,
    is_t5: bool = False,
    pad_dataset: bool = True,
    max_num_samples_per_document: int = 1,
    t5_mlm_noise_density: Optional[float] = None,
    t5_mlm_mean_noise_span_length: Optional[int] = None,
    add_begin_of_doc_token: bool = True,
    add_end_of_doc_token: bool = True,
    max_num_images_per_document: Optional[int] = None,
):
    mapper_kwargs = {
        "tokenizer": tokenizer,
        "image_transform": image_transform,
        "max_seq_len": max_seq_len,
        "max_num_images": max_num_images,
        "max_image_size": max_image_size,
        "vision_encoder_max_image_size": vision_encoder_max_image_size,
        "pre_split_scale_up_max": pre_split_scale_up_max,
        "pre_split_scale_up_frequency": pre_split_scale_up_frequency,
        "image_seq_len": image_seq_len,
        "add_begin_of_doc_token": add_begin_of_doc_token,
        "add_end_of_doc_token": add_end_of_doc_token,
    }

    if not pad_dataset:
        raise ValueError("This feature has been deprecated. The dataset must be padded")

    if is_t5:
        mapper_kwargs["noise_density"] = t5_mlm_noise_density
        mapper_kwargs["mean_noise_span_length"] = t5_mlm_mean_noise_span_length
        raise ValueError("This feature has been deprecated. We can't pack for t5")
    elif dataset_type == DatasetTypes.IMAGE_CAPTION_PAIRS:
        split_fn = split_pack_and_pad_pairs
    elif dataset_type == DatasetTypes.OCR:
        split_fn = split_pack_and_pad_ocr
    elif (dataset_type == DatasetTypes.VQAV2_TASK_FINETUNING) or (dataset_type == DatasetTypes.DOCVQA):
        split_fn = split_pack_and_pad_iqa_finetuning
    elif dataset_type == DatasetTypes.SFT:
        split_fn = split_pack_and_pad_sft
    elif dataset_type == DatasetTypes.WEB_DOCUMENTS:
        split_fn = split_pack_and_pad_webdocs
        mapper_kwargs["max_num_samples_per_document"] = max_num_samples_per_document
        mapper_kwargs["max_num_images_per_document"] = max_num_images_per_document

    mapper_with_args = partial(split_fn, **mapper_kwargs)
    return mapper_with_args


def get_dataloaders(
    config: Parameters,
    rank: int,
    world_size: int,
    tokenizer,
    train_image_transforms,
    val_image_transforms,
    image_seq_len: int,
):
    logger.info("Getting the train dataloader")
    train_loader = get_dataloader_from_config(
        tokenizer=tokenizer,
        image_transforms=train_image_transforms,
        seed=config.data_param.train_seed,
        config=config,
        is_train=True,
        rank=rank,
        world_size=world_size,
        image_seq_len=image_seq_len,
    )

    if config.hparams.do_validation:
        logger.info("Getting the validation dataloader")
        val_loader = get_dataloader_from_config(
            tokenizer=tokenizer,
            image_transforms=val_image_transforms,
            seed=config.data_param.val_seed,
            config=config,
            is_train=False,
            rank=rank,
            world_size=world_size,
            image_seq_len=image_seq_len,
        )
    else:
        val_loader = None

    return train_loader, val_loader


def load_hf_dataset(dataset_path):
    split_name = None
    config_name = None
    if ":" in dataset_path:
        dataset_path_splitted = dataset_path.split(":")
        if len(dataset_path_splitted) == 2:
            dataset_path, split_name = dataset_path_splitted
        elif len(dataset_path_splitted) == 3:
            dataset_path, config_name, split_name = dataset_path_splitted

    if os.path.exists(dataset_path):
        # a local path dataset can be of two kinds
        # 1. generated by `save_to_disk` and thus must be loaded with `load_from_disk`
        # 2. hub-like dataset, but which is not online
        # so we try the first and if it fails with `FileNotFoundError` (despite the path existing) we try the second
        try:
            hf_dataset = datasets.load_from_disk(dataset_path)
        except FileNotFoundError:
            if config_name is not None:
                hf_dataset = datasets.load_dataset(dataset_path, name=config_name)
            else:
                hf_dataset = datasets.load_dataset(dataset_path)
    else:
        if config_name is not None:
            hf_dataset = datasets.load_dataset(
                dataset_path, name=config_name, use_auth_token=os.environ.get("HF_TOKEN", True)
            )
        else:
            hf_dataset = datasets.load_dataset(dataset_path, use_auth_token=os.environ.get("HF_TOKEN", True))

    if split_name is not None:
        hf_dataset = hf_dataset[split_name]

    return hf_dataset


def get_dataset_hf(
    dataset_config: DatasetParams,
    tokenizer,
    image_transform,
    is_train: bool = True,
    realtime_processing: bool = True,
    is_t5: bool = False,
):
    dataset_list = []
    hf_datasets_paths = (
        dataset_config.training_datasets_paths if is_train else dataset_config.validation_datasets_paths
    )
    # hf_datasets_paths can be a list of paths, or a .txt file path that contains the paths
    if len(hf_datasets_paths) == 1 and str(hf_datasets_paths[0]).endswith(".txt"):
        with open(hf_datasets_paths[0], "r") as file_shards:
            hf_datasets_paths = [path for path in file_shards.read().split("\n") if path]

    for dataset_path in hf_datasets_paths:
        hf_dataset = load_hf_dataset(dataset_path=str(dataset_path))

        is_paired_dataset = "meta" in hf_dataset[0] and "source" in hf_dataset[0]

        optional_kwargs_defaults = [
            ("pad_dataset", True),
            ("max_num_samples_per_document", 1),
            ("t5_mlm_noise_density", 0.15),
            ("t5_mlm_mean_noise_span_length", 3),
            ("add_begin_of_doc_token", True),
            ("add_end_of_doc_token", True),
            ("max_num_images_per_document", None),
        ]
        optional_kwargs = {}
        for key, default in optional_kwargs_defaults:
            optional_kwargs[key] = getattr(dataset_config, key, default)

        if not realtime_processing:
            mapper_with_args = get_mapper(
                tokenizer=tokenizer,
                image_transform=image_transform,
                image_seq_len=dataset_config.image_seq_len,
                max_seq_len=dataset_config.max_seq_len,
                max_num_images=dataset_config.max_num_images,
                max_image_size=dataset_config.max_image_size,
                vision_encoder_max_image_size=dataset_config.vision_encoder_max_image_size,
                pre_split_scale_up_max=dataset_config.pre_split_scale_up_max,
                pre_split_scale_up_frequency=dataset_config.pre_split_scale_up_frequency,
                dataset_type=DatasetTypes.IMAGE_CAPTION_PAIRS if is_paired_dataset else DatasetTypes.WEB_DOCUMENTS,
                is_t5=is_t5,
                **optional_kwargs,
            )
            hf_dataset = hf_dataset.map(
                mapper_with_args,
                batched=True,
                batch_size=dataset_config.map_batch_size,
                remove_columns=hf_dataset.column_names,
                num_proc=dataset_config.map_num_proc,
            )
        dataset_list.append(hf_dataset)
    return dataset_list


def get_dataset_webdataset(
    dataset_config: DatasetParams,
    is_train: bool = True,
    realtime_processing: bool = True,
):
    if not realtime_processing:
        raise NotImplementedError("WebDataset is only supported for realtime processing")

    webdataset_paths = dataset_config.training_datasets_paths if is_train else dataset_config.validation_datasets_paths

    if len(webdataset_paths) == 0:
        return None

    # webdataset_paths can be a list of paths/commands, or a .txt file path that contains the paths
    if len(webdataset_paths) == 1 and str(webdataset_paths[0]).endswith(".txt"):
        with open(webdataset_paths[0], "r") as file_shards:
            webdataset_paths = [path for path in file_shards.read().split("\n") if path]
    else:
        raise ValueError("WebDataset only supports a .txt file with the paths or the commands.")

    # Check if the paths/commands are valid
    checks = all([check_webdataset_command(path) for path in webdataset_paths])
    if not checks:
        raise ValueError("WebDataset paths/commands are not valid. Please check the paths/commands.")

    combined_dataset = get_webdataset(
        urls=webdataset_paths,
        ds_type=dataset_config.dataset_type,
        batch_size=dataset_config.map_batch_size,
        shuffle_initial_urls_list=dataset_config.shuffle_initial_urls_list if is_train else False,
        shuffle_before_split_by_node_buffer_size=(
            dataset_config.shuffle_before_split_by_node_buffer_size if is_train else None
        ),
        shuffle_before_split_by_worker_buffer_size=(
            dataset_config.shuffle_before_split_by_worker_buffer_size if is_train else None
        ),
        shuffle_after_tarfile_to_samples_buffer_size=(
            dataset_config.shuffle_after_tarfile_to_samples_buffer_size if is_train else None
        ),
        shuffle_after_batching_buffer_size=dataset_config.shuffle_after_batching_buffer_size if is_train else None,
    )
    return combined_dataset


def get_dataset(
    dataset_config: DatasetParams,
    tokenizer=None,
    image_transform=None,
    is_train: bool = True,
    realtime_processing: bool = True,
    is_t5: bool = False,
    use_webdataset: bool = False,
):
    if use_webdataset:
        return get_dataset_webdataset(
            dataset_config=dataset_config,
            is_train=is_train,
            realtime_processing=realtime_processing,
        )
    else:
        return get_dataset_hf(
            dataset_config=dataset_config,
            tokenizer=tokenizer,
            image_transform=image_transform,
            is_train=is_train,
            realtime_processing=realtime_processing,
            is_t5=is_t5,
        )


def get_dataloader(
    tokenizer,
    image_transforms,
    seed,
    num_workers=1,
    pin_memory=False,
    batch_size=10,
    is_train=True,
    persistent_workers=True,
    realtime_processing=False,
    # The following arguments only used for iterable dataset
    rank=None,
    world_size=None,
    # This argument is for controlling sample order randomness for Map-style Datasets when resuming a run
    sampler_rng=None,
    accumulate_datasets=False,
    model_name=None,
    data_param: Optional[DataParams] = None,
    image_seq_len=None,
):
    if is_train:
        select_n_examples = data_param.select_n_examples_train
    else:
        select_n_examples = data_param.select_n_examples_validation

    if data_param is None:
        raise ValueError("data_param must be provided")

    is_t5 = "t5" in model_name if model_name is not None else False
    dataset_list_map = {}

    # Try all possible datasets, if they don't have datasets paths in the config, they
    # will end up with an empty list
    for dataset_name in DatasetNames:
        curr_dataset_config = getattr(data_param, dataset_name.name.lower())
        dataset_list_map[dataset_name.name.lower()] = get_dataset(
            dataset_config=curr_dataset_config,
            tokenizer=tokenizer,
            image_transform=image_transforms[dataset_name.name.lower()],
            is_train=is_train,
            realtime_processing=realtime_processing,
            is_t5=is_t5,
            use_webdataset=data_param.use_webdataset,
        )

    if not realtime_processing:
        # => Important & gnarly: Set image transform based on a novel instance of `np.default_rng` seeded by the parent
        #   seed and the current rank; because of the way DataLoader `num_workers` multiprocessing works in tandem with
        #   the default `hf.dataset` Arrow backend + normal PyTorch `DataLoader, Sampler, and BatchSampler` behavior,
        #   any "global" reference to an rng object that gets to this transform will get "pickled and copied over" to
        #   each separate worker process.
        #
        #   This wouldn't be a problem if we could simply just "reset" the randomness of the Dataset, but that's opaque
        #   given the `hf.Dataset` wrapper; as such, we need to just handle the randomness ourselves by advancing the
        #   random state the appropriate amount in the `__getitem__` of the MapWrapperDataset (as that's only other
        #   place where we're sure we're in scope for a given worker's process block).
        full_dataset = datasets.concatenate_datasets(
            [dataset for dataset_list in dataset_list_map.values() for dataset in dataset_list]
        )
        if select_n_examples is not None:
            full_dataset = full_dataset.select(range(select_n_examples))
        transform_rng = np.random.default_rng(seed=[seed, rank])

        # Wrap `full_dataset` in custom MapWrapperDataset, and initialize a ResumableSampler
        full_dataset = MapWrapperDataset(full_dataset, transform_rng)
        resume_sampler = ResumableSampler(full_dataset, sampler_rng)

        return DataLoaderForMappedWrapperDataset(
            full_dataset,
            batch_size=batch_size,
            sampler=resume_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=persistent_workers,
        )
    else:
        realtime_processing_datasets = []
        for dataset_name in DatasetNames:
            dataset_list_or_combined = dataset_list_map.get(dataset_name.name.lower(), [])

            if dataset_list_or_combined is None or (
                isinstance(dataset_list_or_combined, list) and len(dataset_list_or_combined) == 0
            ):
                continue

            if isinstance(dataset_list_or_combined, list):
                # If we have a list of datasets, we know those are hf datasets, so we can concatenate them
                combined_dataset = datasets.concatenate_datasets(dataset_list_or_combined)
                if len(combined_dataset) // max(num_workers, 1) < batch_size:
                    raise ValueError(
                        f"For real-time processing, len(dataset) [={len(combined_dataset)}] // num_workers"
                        f" [={num_workers}] must be >= batch_size [={batch_size}]!"
                    )
                if select_n_examples is not None:
                    combined_dataset = combined_dataset.select(range(select_n_examples))
                wrapper_dataset_class = IterableWrapperHFDataset
            elif isinstance(dataset_list_or_combined, wds.pipeline.DataPipeline):
                combined_dataset = dataset_list_or_combined
                wrapper_dataset_class = IterableWrapperWebdataset
            else:
                raise ValueError("Type unrecognized")

            dataset_config: DatasetParams = getattr(data_param, dataset_name.name.lower())
            dataset_kwargs = asdict(dataset_config)
            signature = inspect.signature(wrapper_dataset_class.__init__)
            dataset_kwargs = {k: v for k, v in dataset_kwargs.items() if k in signature.parameters}
            iterable_dataset_instance = wrapper_dataset_class(
                combined_dataset,
                tokenizer=tokenizer,
                image_transform=image_transforms[dataset_name.name.lower()],
                batch_size=batch_size,
                seed=seed,
                shuffle=is_train,
                rank=rank,
                world_size=world_size,
                drop_last=True,
                is_t5=is_t5,
                image_seq_len=image_seq_len,
                **dataset_kwargs,
            )
            realtime_processing_datasets.append(iterable_dataset_instance)

        full_dataset = CustomChainDataset(
            realtime_processing_datasets,
            num_workers,
            rank,
            accumulate_datasets=accumulate_datasets,
            proba_interleaving_dataset=data_param.proba_interleaving_dataset,
            is_train=is_train,
        )

        return DataLoaderForIterableWrapperDataset(
            full_dataset,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=simple_collate,
            drop_last=True,
            rank=rank,
            world_size=world_size,
        )


def get_dataloader_from_config(
    tokenizer,
    image_transforms,
    image_seq_len,
    seed,
    config,
    is_train,
    rank=None,
    world_size=None,
    sampler_rng=None,
):
    dataloader = get_dataloader(
        tokenizer=tokenizer,
        image_transforms=image_transforms,
        seed=seed,
        num_workers=config.data_param.num_workers,
        pin_memory=config.data_param.pin_memory,
        batch_size=config.data_param.batch_size,
        is_train=is_train,
        persistent_workers=config.data_param.persistent_workers,
        realtime_processing=config.data_param.realtime_processing,
        rank=rank,
        world_size=world_size,
        sampler_rng=sampler_rng,
        accumulate_datasets=True if config.hparams.loss_weights_per_dataset is not None else False,
        model_name=config.hparams.model_name,
        data_param=config.data_param,
        image_seq_len=image_seq_len,
    )
    return dataloader


# Creates a Resumable Sampler for Map-style Datasets that predefines the set of indices to retrieve for
# the given epoch (seeded with a generator!). We are using this instead of just seeding the "default Sampler"
# so we can quickly "short-circuit" and bypass any and all "seen" indices (see `__iter__`).
class ResumableSampler(Sampler):
    def __init__(self, data_source, sample_generator):
        super().__init__(data_source)
        self.data_source, self.indices = data_source, None

        # Note: `accelerate` hardcodes a search for an attribute named `generator`, which it then uses to synchronize
        # generators across all ranks on each call to `__iter__`; in our case, this is really bad as we want full
        # control over "sample" randomness (only for Map-style datasets). To get around this, it suffices to just
        # name the instance attributes anything other than `generator`... so that's what we do!
        self.sample_generator = sample_generator

        # For "instant" resuming
        self.n_seen_examples = 0

    def set_state(self, n_seen_examples_per_worker):
        self.n_seen_examples = sum(n_seen_examples_per_worker.values())

    def get_index_order(self):
        return torch.randperm(len(self.data_source), generator=self.sample_generator).tolist()

    def __iter__(self):
        self.indices = self.get_index_order()

        # Resume logic -> advance by `n_seen_examples`
        self.indices = self.indices[self.n_seen_examples :]

        yield from self.indices

    def __len__(self):
        return len(self.data_source)


# Super simple wrapper around dataset; mostly for compatibility with IterableWrapperDataset
class MapWrapperDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_rng):
        self.wrapped_dataset, self.transform_rng = dataset, transform_rng

        # For "instant" resuming
        self.n_seen_examples_per_worker = {}

    def state_dict(self):
        state_dict = {}

        # recurse into its dataset
        state_dict["wrapped_dataset"] = self.wrapped_dataset.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        # recurse into its dataset
        self.wrapped_dataset.load_state_dict(state_dict["wrapped_dataset"])

    @property
    def dataset(self):
        return self.wrapped_dataset

    def set_state(self, n_seen_examples_per_worker):
        self.n_seen_examples_per_worker = n_seen_examples_per_worker

    def __getitem__(self, idx):
        # Dummy dataset idx used for compatibility with CustomChainDataset
        dummy_dataset_idx = 0
        worker_info = torch.utils.data.get_worker_info()
        n_seen_examples = self.n_seen_examples_per_worker.get(worker_info.id, 0)

        # If `n_seen_examples` is non-zero --> advance random state "quickly" then return new example!
        if n_seen_examples > 0:
            for _ in range(n_seen_examples):
                self.transform_rng.random()
            self.n_seen_examples_per_worker[worker_info.id] = 0

        return dummy_dataset_idx, worker_info.id, self.wrapped_dataset[idx]

    def __len__(self):
        return len(self.wrapped_dataset)


class IterableWrapperHFDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        image_transform,
        batch_size,
        seed,
        dataset_type,
        dataset_name,
        image_seq_len,
        shuffle=True,
        rank=None,
        world_size=None,
        drop_last=True,
        is_t5=False,
        # Dataset specific params
        max_num_images=5,
        max_seq_len=256,
        max_image_size=384,
        vision_encoder_max_image_size=384,
        pre_split_scale_up_max=1.0,
        pre_split_scale_up_frequency=0.0,
        pad_dataset=True,
        mapper_batch_size=128,
        # Setting default to 0 as PMD doesn't need it and it is set to 0.5 in CM4 config by default
        max_num_samples_per_document=1,
        t5_mlm_noise_density=None,
        t5_mlm_mean_noise_span_length=None,
        add_begin_of_doc_token=True,
        add_end_of_doc_token=True,
        shuffle_after_packing=False,
        max_num_images_per_document=None,
    ):
        self.dataset = dataset
        self.mapper = get_mapper(
            tokenizer=tokenizer,
            image_transform=image_transform,
            image_seq_len=image_seq_len,
            max_seq_len=max_seq_len,
            max_num_images=max_num_images,
            max_image_size=max_image_size,
            vision_encoder_max_image_size=vision_encoder_max_image_size,
            pre_split_scale_up_max=pre_split_scale_up_max,
            pre_split_scale_up_frequency=pre_split_scale_up_frequency,
            pad_dataset=pad_dataset,
            max_num_samples_per_document=max_num_samples_per_document,
            dataset_type=dataset_type,
            is_t5=is_t5,
            t5_mlm_noise_density=t5_mlm_noise_density,
            t5_mlm_mean_noise_span_length=t5_mlm_mean_noise_span_length,
            add_begin_of_doc_token=add_begin_of_doc_token,
            add_end_of_doc_token=add_end_of_doc_token,
            max_num_images_per_document=max_num_images_per_document,
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.mapper_batch_size = mapper_batch_size
        self.drop_last = drop_last

        self.shuffle_after_packing = shuffle_after_packing

        # To be initialized later in __iter__
        self.rng = None

        # Resume Tracking --> Dict[worker_idx] -> Tuple[map_idx, key_idx]; `map_idx` lets us jumpstart!
        self.worker_idx_tracker = {}
        self.start_worker_idx = 0

        self.dataset_name = dataset_name

    def set_state(self, worker_idx_tracker, start_worker_idx):
        self.worker_idx_tracker = worker_idx_tracker
        self.start_worker_idx = start_worker_idx

    def set_epoch(self, epoch):
        self.epoch = epoch

    def state_dict(self):
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict):
        pass

    def _get_worker_id_and_worker_total_num(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_total_num = 1
            worker_id = 0
        else:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id

        worker_id = (worker_id + self.start_worker_idx) % worker_total_num
        return worker_id, worker_total_num

    def _get_worker_indices(self):
        sampler = torch.utils.data.DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        sampler.set_epoch(self.epoch)

        # Get list of indices for current rank from the distributed sampler
        indices = list(iter(sampler))

        worker_id, worker_total_num = self._get_worker_id_and_worker_total_num()
        # Take the subset of indices that belong to this worker
        #   => It will look something like 0, 2, 4 when worker_id is 0 and num_workers is 2
        worker_indices = indices[worker_id::worker_total_num]
        return worker_indices, worker_id

    def __iter__(self):
        # Dummy dataset idx used for compatibility with CustomChainDataset
        dummy_dataset_idx = 0

        if self.rank is None or self.world_size is None:
            raise ValueError("rank and world_size must be provided")

        # Get worker indices & details for resuming...
        worker_indices, worker_id = self._get_worker_indices()
        num_worker_indices = len(worker_indices)

        # Set start idx of loop based on `self.worker_resume_idxs`
        map_start_idx, last_key_idx, overflow_batch = self.worker_idx_tracker.get(worker_id, (0, -1, {}))
        self.rng_seed = [self.seed, self.epoch, self.rank, worker_id, map_start_idx]
        self.rng = np.random.default_rng(seed=self.rng_seed)
        for i in range(map_start_idx, num_worker_indices, self.mapper_batch_size):
            # Set seed for the worker according to worker index and the index and then reset it work
            # This needs to be done so that torch random crop is deterministic
            rng_state = torch.get_rng_state()
            torch.manual_seed(f"{self.seed}{worker_id}{i}")
            # Feed `worker_indices[i]` to mapper to ensure "deterministic randomness" that we don't have to track...
            curr_mapped_batch = self.mapper(
                self.dataset[worker_indices[i : i + self.mapper_batch_size]],
                prefix_seed=(self.seed, self.epoch, self.rank, worker_id, i),
            )
            torch.set_rng_state(rng_state)
            keys = list(curr_mapped_batch.keys())
            overflow_batch_keys = overflow_batch.keys()

            # Check if overflow from previous batches is left, if yes, add it to the current batch
            # Specifically, we should prepend this overflow batch so as it goes out first and
            # current batch possibly becomes next overflow batch
            if len(overflow_batch_keys) > 0:
                if sorted(overflow_batch_keys) != sorted(keys):
                    raise ValueError(
                        "Overflow batch keys not equal to current keys. Make sure mapper is always returning"
                        "  dictionary with the same keys. "
                        f"Overflow: {sorted(overflow_batch_keys)}, Mapping: {sorted(keys)}"
                    )
                else:
                    mapped_batch = {}

                    if "pixel_values" in overflow_batch or "pixel_values" in curr_mapped_batch:
                        total_batch_size = overflow_batch["input_ids"].size(0) + curr_mapped_batch["input_ids"].size(0)
                        max_num_images = max(
                            overflow_batch["pixel_values"].size(1) if "pixel_values" in overflow_batch else 0,
                            curr_mapped_batch["pixel_values"].size(1) if "pixel_values" in curr_mapped_batch else 0,
                        )
                        max_height = max(
                            overflow_batch["pixel_values"].size(3) if "pixel_values" in overflow_batch else 0,
                            curr_mapped_batch["pixel_values"].size(3) if "pixel_values" in curr_mapped_batch else 0,
                        )
                        max_width = max(
                            overflow_batch["pixel_values"].size(4) if "pixel_values" in overflow_batch else 0,
                            curr_mapped_batch["pixel_values"].size(4) if "pixel_values" in curr_mapped_batch else 0,
                        )
                        padded_image_tensor = torch.zeros(total_batch_size, max_num_images, 3, max_height, max_width)
                        padded_pixel_attention_masks = torch.zeros(
                            total_batch_size, max_num_images, max_height, max_width, dtype=torch.bool
                        )

                        start = 0
                        for batch in [overflow_batch, curr_mapped_batch]:
                            if "pixel_values" not in batch:
                                continue
                            px = batch["pixel_values"]
                            px_attn_mask = batch["pixel_attention_mask"]
                            end = start + px.size(0)
                            padded_image_tensor[start:end, :, :, : px.size(3), : px.size(4)] = px
                            padded_pixel_attention_masks[start:end, :, : px.size(3), : px.size(4)] = px_attn_mask
                            start += px.size(0)

                        mapped_batch["pixel_values"] = padded_image_tensor.contiguous()
                        mapped_batch["pixel_attention_mask"] = padded_pixel_attention_masks.contiguous()

                    for key in keys:
                        if key in ["pixel_values", "pixel_attention_mask"]:
                            continue
                        mapped_batch[key] = torch.cat([overflow_batch[key], curr_mapped_batch[key]], dim=0)
                    previous_overflow_batch = copy.deepcopy(overflow_batch)
                    overflow_batch = {}
            else:
                previous_overflow_batch = {}
                mapped_batch = curr_mapped_batch

            first_key = keys[0]
            mapped_batch_length = len(mapped_batch[first_key])

            if self.shuffle_after_packing:
                indices = list(range(mapped_batch_length))
                self.rng.shuffle(indices)

                for key in mapped_batch.keys():
                    mapped_batch[key] = mapped_batch[key][indices, ...]

            if mapped_batch_length < self.batch_size:
                # We need to add more data to this batch to make it of size `self.batch_size`
                # Just setting mapped_batch to overflow_batch should be enough as the next iteration
                # will add more data to it
                overflow_batch = mapped_batch
            else:
                # Now, yield batches of size batch_size from the mapped batch
                for key_idx in range(0, mapped_batch_length, self.batch_size):
                    # Set "reproducible" randomness
                    self.rng_seed = [self.seed, self.epoch, self.rank, worker_id, i, key_idx]
                    self.rng = np.random.default_rng(seed=self.rng_seed)

                    if i == map_start_idx and key_idx <= last_key_idx:
                        # Handle Resume (only for "first" loop iteration) advance random state until `last_key_idx`
                        self.rng.random()
                    else:
                        overflow_batch = {key: mapped_batch[key][key_idx : key_idx + self.batch_size] for key in keys}
                        if len(overflow_batch[first_key]) != self.batch_size:
                            # Last batch
                            break
                        else:
                            dataset_state = {
                                "worker_idx": worker_id,
                                "map_start_idx": i,
                                "last_key_idx": key_idx,
                                "previous_overflow_batch": previous_overflow_batch,
                            }
                            yield dummy_dataset_idx, self.dataset_name.name.lower(), dataset_state, overflow_batch
                            overflow_batch = {}


class IterableWrapperWebdataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        image_transform,
        batch_size,
        seed,
        dataset_type,
        dataset_name,
        image_seq_len,
        shuffle=True,
        rank=None,
        world_size=None,
        drop_last=True,
        is_t5=False,
        # Dataset specific params
        max_num_images=5,
        max_seq_len=256,
        max_image_size=384,
        vision_encoder_max_image_size=384,
        pre_split_scale_up_max=1.0,
        pre_split_scale_up_frequency=0.0,
        pad_dataset=True,
        mapper_batch_size=128,
        # Setting default to 0 as PMD doesn't need it and it is set to 0.5 in CM4 config by default
        max_num_samples_per_document=1,
        t5_mlm_noise_density=None,
        t5_mlm_mean_noise_span_length=None,
        add_begin_of_doc_token=True,
        add_end_of_doc_token=True,
        shuffle_after_packing=False,
        max_num_images_per_document=None,
    ):
        self._webdataset = dataset
        self.dataset = iter(self._webdataset)
        self.mapper = get_mapper(
            tokenizer=tokenizer,
            image_transform=image_transform,
            image_seq_len=image_seq_len,
            max_seq_len=max_seq_len,
            max_num_images=max_num_images,
            max_image_size=max_image_size,
            vision_encoder_max_image_size=vision_encoder_max_image_size,
            pre_split_scale_up_max=pre_split_scale_up_max,
            pre_split_scale_up_frequency=pre_split_scale_up_frequency,
            pad_dataset=pad_dataset,
            max_num_samples_per_document=max_num_samples_per_document,
            dataset_type=dataset_type,
            is_t5=is_t5,
            t5_mlm_noise_density=t5_mlm_noise_density,
            t5_mlm_mean_noise_span_length=t5_mlm_mean_noise_span_length,
            add_begin_of_doc_token=add_begin_of_doc_token,
            add_end_of_doc_token=add_end_of_doc_token,
            max_num_images_per_document=max_num_images_per_document,
        )
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size
        self.mapper_batch_size = mapper_batch_size
        self.drop_last = drop_last
        self.shuffle_after_packing = shuffle_after_packing

        # To be initialized later in __iter__
        self.rng = None

        # Resume Tracking --> Dict[worker_idx] -> Tuple[map_idx, key_idx]; `map_idx` lets us jumpstart!
        self.worker_idx_tracker = {}
        self.start_worker_idx = 0

        self.dataset_name = dataset_name

    def set_state(self, worker_idx_tracker, start_worker_idx):
        self.worker_idx_tracker = worker_idx_tracker
        self.start_worker_idx = start_worker_idx

    def reset_state(self):
        for key in self.worker_idx_tracker.keys():
            self.worker_idx_tracker[key] = (0, -1, {})
        self.start_worker_idx = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.dataset = iter(self._webdataset)  # Reset dataset iterator

    def state_dict(self):
        state_dict = {}
        return state_dict

    def load_state_dict(self, state_dict):
        pass

    def _get_worker_id_and_worker_total_num(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_total_num = 1
            worker_id = 0
        else:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id

        worker_id = (worker_id + self.start_worker_idx) % worker_total_num
        return worker_id, worker_total_num

    def _get_worker_indices(self):
        sampler = torch.utils.data.DistributedSampler(
            self.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=self.shuffle,
            seed=self.seed,
            drop_last=self.drop_last,
        )
        sampler.set_epoch(self.epoch)

        # Get list of indices for current rank from the distributed sampler
        indices = list(iter(sampler))

        worker_id, worker_total_num = self._get_worker_id_and_worker_total_num()
        # Take the subset of indices that belong to this worker
        #   => It will look something like 0, 2, 4 when worker_id is 0 and num_workers is 2
        worker_indices = indices[worker_id::worker_total_num]
        return worker_indices, worker_id

    def __iter__(self):
        # Dummy dataset idx used for compatibility with CustomChainDataset
        dummy_dataset_idx = 0

        if self.rank is None or self.world_size is None:
            raise ValueError("rank and world_size must be provided")

        # Relic from previous implementation - but needed for rng seed
        worker_id, worker_total_num = self._get_worker_id_and_worker_total_num()

        # Relic from previous implementation - but needed for rng seed
        map_start_idx, last_key_idx, overflow_batch = self.worker_idx_tracker.get(worker_id, (0, -1, {}))

        # Relic from previous implementation - but needed for rng seed
        i = map_start_idx

        # Initialize rng_seed
        self.rng_seed = [self.seed, self.epoch, self.rank, worker_id, i]
        self.rng = np.random.default_rng(seed=self.rng_seed)
        while True:
            # Set seed for the worker according to worker index and the index and then reset it work
            # This needs to be done so that torch random crop is deterministic
            rng_state = torch.get_rng_state()
            torch.manual_seed(f"{self.seed}{worker_id}{i}")
            try:
                next_batch = next(self.dataset)
                i += 1
            except StopIteration:
                logger.info(
                    f"{self.dataset_name.name.lower()} has finished one epoch and is moving on to the next one."
                    f" (epoch={self.epoch} - rank={self.rank} - worker_id={worker_id})"
                )
                break
            curr_mapped_batch = self.mapper(
                next_batch,
                prefix_seed=(self.seed, self.epoch, self.rank, worker_id, i),
            )
            torch.set_rng_state(rng_state)
            keys = list(curr_mapped_batch.keys())
            overflow_batch_keys = overflow_batch.keys()

            # Check if overflow from previous batches is left, if yes, add it to the current batch
            # Specifically, we should prepend this overflow batch so as it goes out first and
            # current batch possibly becomes next overflow batch
            if len(overflow_batch_keys) > 0:
                if sorted(overflow_batch_keys) != sorted(keys):
                    raise ValueError(
                        "Overflow batch keys not equal to current keys. Make sure mapper is always returning"
                        "  dictionary with the same keys. "
                        f"Overflow: {sorted(overflow_batch_keys)}, Mapping: {sorted(keys)}"
                    )
                else:
                    mapped_batch = {}

                    if "pixel_values" in overflow_batch or "pixel_values" in curr_mapped_batch:
                        total_batch_size = overflow_batch["input_ids"].size(0) + curr_mapped_batch["input_ids"].size(0)
                        max_num_images = max(
                            overflow_batch["pixel_values"].size(1) if "pixel_values" in overflow_batch else 0,
                            curr_mapped_batch["pixel_values"].size(1) if "pixel_values" in curr_mapped_batch else 0,
                        )
                        max_height = max(
                            overflow_batch["pixel_values"].size(3) if "pixel_values" in overflow_batch else 0,
                            curr_mapped_batch["pixel_values"].size(3) if "pixel_values" in curr_mapped_batch else 0,
                        )
                        max_width = max(
                            overflow_batch["pixel_values"].size(4) if "pixel_values" in overflow_batch else 0,
                            curr_mapped_batch["pixel_values"].size(4) if "pixel_values" in curr_mapped_batch else 0,
                        )
                        padded_image_tensor = torch.zeros(total_batch_size, max_num_images, 3, max_height, max_width)
                        padded_pixel_attention_masks = torch.zeros(
                            total_batch_size, max_num_images, max_height, max_width, dtype=torch.bool
                        )

                        start = 0
                        for batch in [overflow_batch, curr_mapped_batch]:
                            if "pixel_values" not in batch:
                                continue
                            px = batch["pixel_values"]
                            px_attn_mask = batch["pixel_attention_mask"]
                            end = start + px.size(0)
                            padded_image_tensor[start:end, :, :, : px.size(3), : px.size(4)] = px
                            padded_pixel_attention_masks[start:end, :, : px.size(3), : px.size(4)] = px_attn_mask
                            start += px.size(0)

                        mapped_batch["pixel_values"] = padded_image_tensor.contiguous()
                        mapped_batch["pixel_attention_mask"] = padded_pixel_attention_masks.contiguous()

                    for key in keys:
                        if key in ["pixel_values", "pixel_attention_mask"]:
                            continue
                        mapped_batch[key] = torch.cat([overflow_batch[key], curr_mapped_batch[key]], dim=0)
                    overflow_batch = {}
            else:
                mapped_batch = curr_mapped_batch

            first_key = keys[0]
            mapped_batch_length = len(mapped_batch[first_key])

            if self.shuffle_after_packing:
                indices = list(range(mapped_batch_length))
                self.rng.shuffle(indices)

                for key in mapped_batch.keys():
                    mapped_batch[key] = mapped_batch[key][indices, ...]

            if mapped_batch_length < self.batch_size:
                # We need to add more data to this batch to make it of size `self.batch_size`
                # Just setting mapped_batch to overflow_batch should be enough as the next iteration
                # will add more data to it
                overflow_batch = mapped_batch
            else:
                # Now, yield batches of size batch_size from the mapped batch
                for key_idx in range(0, mapped_batch_length, self.batch_size):
                    # Set "reproducible" randomness
                    self.rng_seed = [self.seed, self.epoch, self.rank, worker_id, i, key_idx]
                    self.rng = np.random.default_rng(seed=self.rng_seed)

                    overflow_batch = {key: mapped_batch[key][key_idx : key_idx + self.batch_size] for key in keys}
                    if len(overflow_batch[first_key]) != self.batch_size:
                        # Last batch
                        break
                    else:
                        dataset_state = {
                            "worker_idx": worker_id,
                            "map_start_idx": i,
                            "last_key_idx": key_idx,
                            "previous_overflow_batch": {},
                        }
                        yield dummy_dataset_idx, self.dataset_name.name.lower(), dataset_state, overflow_batch
                        overflow_batch = {}


class CustomChainDataset(torch.utils.data.IterableDataset):
    r"""Dataset for chaining multiple :class:`IterableDataset` s.

    This class is useful to assemble different existing dataset streams. The
    chaining operation is done on-the-fly, so concatenating large-scale
    datasets with this class will be efficient.

    For validation set:
        This class iterates over each dataset one by one.
        One dataset is iterated over completely before moving to the next one.

    Args:
        datasets (iterable of IterableDataset): datasets to be chained together
        num_workers (int): number of workers to use for loading data
        rank (int): rank of the current process
        accumulate_datasets (bool): whether to accumulate the datasets or not
        proba_interleaving_dataset (list of float): probability of interleaving each dataset, first probability
            is of the PMD, second one of CM4
        is_train (bool): whether the dataset is used for training or not. Setting this to false will ignore
            accumulate_datasets and proba_interleaving_dataset parameters and set them to False and None.
            See commentary above for validation set to understand the behavior of this class when
            `is_train` is False.
    """

    def __init__(
        self, datasets, num_workers, rank, accumulate_datasets=False, proba_interleaving_dataset=None, is_train=True
    ):
        super(CustomChainDataset, self).__init__()

        for d in datasets:
            if not isinstance(d, torch.utils.data.IterableDataset):
                raise ValueError(f"CustomChainDataset only supports IterableDataset, but got {type(d)}")

        self.datasets = datasets
        self.num_workers = num_workers if num_workers > 1 else 1
        self.num_datasets = len(self.datasets)
        self.is_train = is_train

        if self.is_train is False:
            if accumulate_datasets is True or proba_interleaving_dataset is not None:
                logger.warn("accumulate_datasets and proba_interleaving_dataset are ignored when is_train is False")
            self.accumulate_datasets = False
            self.dataset_proba = None
        else:
            self.accumulate_datasets = accumulate_datasets
            if not self.accumulate_datasets:
                if proba_interleaving_dataset is not None:
                    self.dataset_proba = np.asarray(proba_interleaving_dataset)
                else:
                    self.dataset_proba = np.full((self.num_datasets), 1 / self.num_datasets, dtype=float)

                if abs(self.dataset_proba.sum() - 1) > 0.001:
                    # Allow a tolerance for floating points rounding errors.
                    raise ValueError("proba_interleaving_dataset must sum to 1")

        self.epoch = 0
        self.seed = sum(dataset.seed for dataset in self.datasets)
        self.rank = rank

        # state-related attributes
        self.start_worker_id = 0
        self.chain_dataset_last_idx_tracker = {}
        self.reset_state()

    def reset_state(self):
        self.chain_dataset_last_idx_tracker = {worker_id: 0 for worker_id in range(self.num_workers)}

    def update_state(self, dataset_state):
        self.chain_dataset_last_idx_tracker[dataset_state["worker_idx"]] = dataset_state["chain_dataset_last_idx"]

    def load_resume_states(self, resumable_states):
        for idx, d in enumerate(self.datasets):
            worker_idx_tracker, start_worker_id = resumable_states[idx]
            d.set_state(worker_idx_tracker, start_worker_id)
            self.start_worker_id = start_worker_id

    def state_dict(self):
        state_dict = {}

        state_dict["chain_dataset_last_idx_tracker"] = self.chain_dataset_last_idx_tracker

        # recurse into its datasets
        state_dict["datasets"] = [d.state_dict() for d in self.datasets]

        return state_dict

    def load_state_dict(self, state_dict):
        for key in state_dict["chain_dataset_last_idx_tracker"].keys():
            if key in self.chain_dataset_last_idx_tracker:
                self.chain_dataset_last_idx_tracker[key] = state_dict["chain_dataset_last_idx_tracker"][key]
            else:
                self.chain_dataset_last_idx_tracker[key] = 0

        # recurse into its datasets
        for idx, d in enumerate(self.datasets):
            d.load_state_dict(state_dict["datasets"][idx])

    def set_epoch(self, epoch):
        #  TODO: change this and fix trainer as well when epoch logic
        #  described in iter is implemented
        self.epoch = epoch
        for d in self.datasets:
            d.set_epoch(epoch)

    def _get_worker_id(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_total_num = 1
            worker_id = 0
        else:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id

        worker_id = (worker_id + self.start_worker_id) % worker_total_num
        return worker_id

    def __iter__(self):
        ds_iterators = [iter(d) for d in self.datasets]
        worker_id = self._get_worker_id()

        # Needed for dataset accumulation. Allows chain dataset to start at the right
        # dataset_idx, as it needs to know what was the last idx was.
        # Note that for validation set, this will always start from zero.
        dataset_idx = self.chain_dataset_last_idx_tracker[worker_id]
        while True:
            rng_seed = [
                self.seed,
                self.epoch,
                self.rank,
                worker_id,
                self.chain_dataset_last_idx_tracker[worker_id],
            ]
            rng = np.random.default_rng(seed=rng_seed)

            if self.is_train:
                if self.accumulate_datasets:
                    dataset_idx = (dataset_idx + 1) % self.num_datasets
                else:
                    dataset_idx = rng.choice(np.arange(0, self.num_datasets), p=self.dataset_proba)

            try:
                _, dataset_name, dataset_state, batch = next(ds_iterators[dataset_idx])
                self.chain_dataset_last_idx_tracker[worker_id] += 1
                dataset_state["chain_dataset_last_idx"] = self.chain_dataset_last_idx_tracker[worker_id]
                yield dataset_idx, dataset_name, dataset_state, batch
            except StopIteration:
                if not self.is_train and dataset_idx < self.num_datasets - 1:
                    dataset_idx += 1
                else:
                    self.epoch += 1
                    self.reset_state()

                    for d in self.datasets:
                        d.reset_state()
                        d.set_epoch(self.epoch)

                    ds_iterators = [iter(d) for d in self.datasets]

                    # TODO: Move epoch logic here instead of training loop
                    # ie: make an infinite dataloader that keeps track of the epochs of
                    # each dataset


# this class is currently not being maintained
class DataLoaderForMappedWrapperDataset(torch.utils.data.DataLoader):
    def state_dict(self):
        state_dict = {}

        # recurse into its dataset
        state_dict["dataset"] = self.dataset.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        # recurse into its dataset
        self.dataset.load_state_dict(state_dict["dataset"])


# These are notes from IterableResumableState that got folded into
# DataLoaderForIterableWrapperDataset, but the comments are still relevant:
#
# **IMPORTANT**: Serializing "transform" randomness is really really difficult to get right because any "global"
#   generators we initialize are not actually going to be respected in any DataLoader with `num_workers > 1`. When you
#   run a DataLoader w/ multiprocessing, each "rng" generator is "copied" over to the separate process; setting that
#   generator only works _within_ the parts of the Dataset/Sampler/BatchSampler/DataLoader that are actually in scope
#   while an individual worker process is live... and there's no way I can currently think of to cleanly do that...


class DataLoaderForIterableWrapperDataset(torch.utils.data.DataLoader):
    def __init__(self, dataset, seed, **kwargs):
        self.seed = seed
        self.rank = kwargs.pop("rank", None)
        self.world_size = kwargs.pop("world_size", None)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if self.rank is None or self.world_size is None:
            try:
                state = AcceleratorState()
                self.rank = state.process_index
                self.device = state.device
                self.world_size = state.num_processes
            except ValueError:
                # This will fail later on when we try to create the IterableWrapperDataset
                pass

        if self.world_size == 1:
            # we normally don't use one process, but when we do in testing there can be a problem
            # with Too many open files issue, and switching to "file_system" strategy overcomes it.
            torch.multiprocessing.set_sharing_strategy("file_system")

        dataset.rank = self.rank
        dataset.world_size = self.world_size
        super().__init__(dataset, **kwargs)
        self.dataset_count = len(self.dataset.datasets)

        self.worker_idx_tracker = [{} for i in range(self.dataset_count)]
        self.next_worker_idx = [0 for i in range(self.dataset_count)]
        self.reset_state()

    def reset_state(self):
        for dataset_idx in range(self.dataset_count):
            self.worker_idx_tracker[dataset_idx] = {}
            self.next_worker_idx[dataset_idx] = 0

        # TODO: Aman: It looks like this has somewhat changed from last I checked. On slack, we
        # discussed to not have the states in the dataset themselves but just on the dataloader and
        # have current state returned from __iter__ of the dataset.
        #
        # The reason for this is what we discussed on slack related to dataset object being unique
        # to each worker and being passed around. So, the current dataset object in your dataloader
        # possibly won't give you correct state rather the state it had when it was initialized. Are
        # we confident that it will work this way?

        self.dataset.reset_state()

    def update_state(self, dataset_idx, dataset_state):
        self.dataset.update_state(dataset_state)

        self.worker_idx_tracker[dataset_idx][dataset_state["worker_idx"]] = (
            dataset_state["map_start_idx"],
            dataset_state["last_key_idx"],
            dataset_state["previous_overflow_batch"],
        )

        for dataset_idx in range(self.dataset_count):
            self.next_worker_idx[dataset_idx] = (dataset_state["worker_idx"] + 1) % self.num_workers

    def set_epoch(self, epoch):
        # Ensure each epoch has a different sequence to sample from
        self.dataset.set_epoch(epoch)

    def load_resume_states(self):
        self.dataset.load_resume_states(self.get_resume_states())

    def get_resume_state(self, dataset_idx):
        return self.worker_idx_tracker[dataset_idx], self.next_worker_idx[dataset_idx]

    def get_resume_states(self):
        return [
            [self.worker_idx_tracker[dataset_idx], self.next_worker_idx[dataset_idx]]
            for dataset_idx in range(self.dataset_count)
        ]

    def state_dict(self):
        state_dict = {}

        state_dict["worker_idx_tracker"] = self.worker_idx_tracker
        state_dict["next_worker_idx"] = self.next_worker_idx

        # recurse into its dataset
        state_dict["dataset"] = self.dataset.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        self.worker_idx_tracker = state_dict["worker_idx_tracker"]
        self.next_worker_idx = state_dict["next_worker_idx"]

        # recurse into its dataset
        self.dataset.load_state_dict(state_dict["dataset"])

    def save_state(self, path):
        """
        Saves state_dict to `m4_states_{process_index}.pkl`
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        location = path / f"m4_states_{self.rank}.pkl"
        logger.info(f"Saving the DL states to {location}")
        torch.save(self.state_dict(), location)

    def load_state(self, path):
        """
        Loads the state_dict at `m4_states_{process_index}.pkl`
        """
        location = Path(path) / f"m4_states_{self.rank}.pkl"
        logger.info(f"Loading the DL states from {location}")
        self.load_state_dict(torch.load(location))

    def __iter__(self):
        main_iterator = super().__iter__()

        stopped = torch.tensor(0.0, device=self.device)
        while True:
            try:
                batch = next(main_iterator)
            except StopIteration:
                stopped += 1

            # check that dist is initialized in case this DL w/ world_size>1 was used w/o distributed environment
            if self.world_size > 1 and torch.distributed.is_initialized():
                torch.distributed.all_reduce(stopped, op=torch.distributed.ReduceOp.SUM)

            # stop iterating if one or more processes stopped to avoid blocking
            if stopped > 0:
                break

            yield batch


# This class isn't being maintained (or tested) at the moment
class MappedResumableState:
    def __init__(self, train_seed, val_seed, world_size, rank, dataloader_num_workers, using_iterable_dataset=False):
        self.train_seed, self.val_seed, self.world_size, self.rank = train_seed, val_seed, world_size, rank
        self.dataloader_num_workers, self.using_iterable_dataset = dataloader_num_workers, using_iterable_dataset
        self.epoch = None

        self.n_seen_examples_per_worker = {w: 0 for w in range(self.dataloader_num_workers)}

        # Create generators for the various sources of randomness
        self.sampler_rng, self.sampler_rng_val = torch.Generator(), torch.Generator()
        self.sampler_rng.manual_seed(self.train_seed)
        self.sampler_rng_val.manual_seed(self.val_seed)

        # Create instance variables to track the `sampler_rng` states (see comment in `set_epoch` for why!)
        self.sampler_rng_state = self.sampler_rng.get_state()
        self.sampler_rng_val_state = self.sampler_rng_val.get_state()

    def set_epoch(self, epoch):
        # At the start of each epoch, the iter(sampler) gets called, which advances the random state; this isn't
        #   great, as it means that if we save the random state post-hoc, when we resume, we're "skipping" to the
        #   next epoch. To preserve the _same_ order within an epoch, we actually need to save the random state
        #   prior to the call to __iter__!
        if self.epoch != epoch:
            self.epoch = epoch

            self.sampler_rng_state = self.sampler_rng.get_state()
            self.sampler_rng_val_state = self.sampler_rng_val.get_state()

    def update_state(self, state_trackers):
        # Gate based on `self.using_iterable_dataset`

        # For a Map-style dataset, `state_trackers` is a Tensor containing all worker_ids responsible for fetching
        #   the given examples of the batch; we'll be using this to increment `n_seen_examples_per_worker`. We'll
        #   be using the key assumption that a single worker generates all elements for the batch!
        self.n_seen_examples_per_worker[state_trackers[0].item()] += state_trackers.numel()

    def update_next_worker_idx(self, curr_worker_idx):
        self.next_worker_idx = (curr_worker_idx + 1) % self.dataloader_num_workers

    def get_resume_state(self):
        # Return minimal `state` to "fast-forward" a given Dataset/DataLoader
        return self.n_seen_examples_per_worker

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "world_size": self.world_size,
            "dataloader_num_workers": self.dataloader_num_workers,
            "sampler_rng_state": self.sampler_rng_state,
            "sampler_rng_state_checksum": self.sampler_rng_state.sum(),
            "sampler_rng_val_state": self.sampler_rng_val_state,
            "sampler_rng_val_state_checksum": self.sampler_rng_val_state.sum(),
            "n_seen_examples_per_worker": self.n_seen_examples_per_worker,
        }

    def load_state_dict(self, state_dict):
        # Verify same "world size" and "num workers" on a resume!
        if self.world_size != state_dict["world_size"]:
            raise ValueError(f"Current world_size {self.world_size} != Loaded world_size {state_dict['world_size']}")

        if self.dataloader_num_workers != state_dict["dataloader_num_workers"]:
            raise ValueError(
                f"Current num_workers `{self.dataloader_num_workers}` != Loaded num_workers"
                f" `{state_dict['dataloader_num_workers']}`"
            )

        # Set epoch
        self.epoch = state_dict["epoch"]

        # Set `sampler_rng`
        self.sampler_rng.set_state(state_dict["sampler_rng_state"])
        self.sampler_rng_val.set_state(state_dict["sampler_rng_val_state"])

        # Set `n_seen_examples_per_worker`
        self.n_seen_examples_per_worker = state_dict["n_seen_examples_per_worker"]
