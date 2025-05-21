import json
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Dict, Iterator, List

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from m4.sourcing.pmd import _FEATURES, get_m4_cache_dir
from m4.sourcing.pmd.helpers import (
    M4HTTPClient,
    PickableMediaDownloadGenerator,
    collapse_columns_to_meta,
    fetch_single_image,
)


# ---- Base loader builder -----


class BaseLoaderBuilder(ABC):
    def __init__(self, split, num_proc: int, batch_size: int = 1000):
        self.split = split
        self.num_proc = num_proc
        self.batch_size = batch_size

    @abstractmethod
    def _load_dataset(self) -> Dataset:
        raise NotImplementedError()

    @abstractmethod
    def update_before_collapsing_meta(self, batch: Dict) -> Dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def _DATASETS_NAME(self) -> str:
        raise NotImplementedError

    def _normalise(self, batch: Dict) -> Dict:
        """
        Create the `text` field, typically from the field `caption` and remove the `caption` column.
        Remove all the un-necessary columns and put them into a json dict (`meta` column).
        """
        # `datasets.map` requires function to return pure-functions, which is not the case here
        # https://github.com/huggingface/datasets/pull/4197#issue-1211342558
        batch = batch.copy()

        batch = self.update_before_collapsing_meta(batch)

        # Collapse columns to meta
        batch = collapse_columns_to_meta(
            batch, columns_to_collapse=self.get_batch_metadata_columns(batch), meta_column_name="meta"
        )

        # add `source`
        batch["source"] = [self._DATASETS_NAME for _ in batch["text"]]

        return batch

    @staticmethod
    def get_batch_metadata_columns(batch: Dict) -> List[str]:
        return list(set(batch.keys()) - {"image", "text"})

    def build(self):
        dset = self._load_dataset()

        if isinstance(dset, Dataset):
            dset = dset.map(
                self._normalise,
                batched=True,
                remove_columns=dset.column_names,
                features=_FEATURES,
                num_proc=self.num_proc,
                batch_size=self.batch_size,
            )

            # Make sure the features match the expected API.
            assert dset.features == _FEATURES, f"Got: {dset.features}, expected: {_FEATURES}"
        elif isinstance(dset, DatasetDict):
            new_dset = DatasetDict()
            for split, dset_split in dset.items():
                new_dset[split] = dset_split.map(
                    self._normalise,
                    batched=True,
                    remove_columns=dset_split.column_names,
                    features=_FEATURES,
                    num_proc=self.num_proc,
                    batch_size=self.batch_size,
                )

                # Make sure the features match the expected API.
                assert new_dset[split].features == _FEATURES, f"Got: {new_dset[split].features}, expected: {_FEATURES}"
            dset = new_dset
        return dset


class DownloadImageLoaderBuilder(ABC):
    def __init__(self, split, num_proc: int, num_threads_per_proc: int, **http_client_kwargs):
        assert "cache_dir" not in http_client_kwargs
        self.http_client = M4HTTPClient(
            cache_dir=get_m4_cache_dir() / self._DATASETS_NAME / "downloaded_images",
            **http_client_kwargs,
            user_agent="Googlebot-Image/1.0",
        )
        self.num_threads_per_proc = num_threads_per_proc
        self.split = split
        self.num_proc = num_proc

    @abstractmethod
    def _load_dataset(self) -> Dataset:
        """Load the original dataset"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def _DATASETS_NAME(self) -> str:
        raise NotImplementedError

    def get_image_urls(self, batch) -> List[str]:
        return batch["image_url"]

    def get_texts(self, batch) -> List[List[str]]:
        """A list, each element is a list of potential captions"""
        return batch["caption"]

    def pre_download_image_map(self, batch: Dict) -> Dict:
        # move `captions` to `text`
        if "text" in batch:
            return batch
        batch["text"] = self.get_texts(batch)
        return batch

    def _normalise(self, batch: Dict) -> Dict:
        """Create the `text` field, typically from the field `caption`."""
        # `datasets.map` requires function to return pure-functions, which is not the case here
        # https://github.com/huggingface/datasets/pull/4197#issue-1211342558
        batch = batch.copy()

        # Changes to do before downloading images: typically filtering.
        new_batch = self.pre_download_image_map(batch)

        return new_batch

    def _add_image_or_exception(self, batch: Dict, image_or_exception_iterator: Iterator) -> Dict:
        """Get the images from the iterator and put them in the batch dict.
        Remove all the un-necessary columns and put them into a json dict (`meta` column).
        Add the source info to the batch dict"""
        # `datasets.map` requires function to return pure-functions, which is not the case here
        # https://github.com/huggingface/datasets/pull/4197#issue-1211342558
        batch = batch.copy()

        # get image or exception
        batch_size = len(next(iter(batch.values())))
        images, exceptions = tuple(zip(*[next(image_or_exception_iterator) for _ in range(batch_size)]))

        # add them to batch
        batch["image_download_exception"] = list(exceptions)
        batch["image"] = [image if image is not None else None for image in images]

        # Collapse columns to meta
        batch = collapse_columns_to_meta(
            batch, columns_to_collapse=self.get_batch_metadata_columns(batch), meta_column_name="meta"
        )

        batch["source"] = [self._DATASETS_NAME for _ in range(batch_size)]

        return batch

    def map_shard(self, shard: Dataset) -> Dataset:
        """
        Prepare the `text` fields, and download (or fetch from cache) images.
        """
        # Decide which urls we need to query
        shard = shard.map(
            self._normalise,
            batched=True,
            remove_columns=shard.column_names,
            num_proc=1,  # This is handled manually
        )

        # actually download the image.
        with PickableMediaDownloadGenerator(
            download_media_url=partial(fetch_single_image, http_client=self.http_client),
            get_media_urls=self.get_image_urls,
            dset=shard,
            batch_size=1000,
            num_threads_per_proc=self.num_threads_per_proc,
        ) as image_iterator:
            # fill new dataset with that image_path of exception
            shard = shard.map(
                partial(self._add_image_or_exception, image_or_exception_iterator=image_iterator),
                batched=True,
                remove_columns=shard.column_names,
                features=_FEATURES,
                num_proc=1,  # SUPER IMPORTANT as `PickableMediaDownloadGenerator` is stateful
            )

        return shard

    @staticmethod
    def get_batch_metadata_columns(batch: Dict) -> List[str]:
        return list(set(batch.keys()) - {"image", "text"})

    def build(self):
        dset = self._load_dataset()

        if isinstance(dset, Dataset):
            shards = [
                dset.shard(num_shards=self.num_proc, index=rank, contiguous=True) for rank in range(self.num_proc)
            ]

            with get_context("spawn").Pool(self.num_proc) as pool:
                results = [pool.apply_async(self.map_shard, kwds={"shard": shard}) for shard in shards]
                transformed_shards = [result.get() for result in results]

                pool.terminate()
                pool.join()
                del pool

            dset = concatenate_datasets(transformed_shards)

            # Make sure the features match the expected API.
            assert dset.features == _FEATURES, f"Got: {dset.features}, expected: {_FEATURES}"
        elif isinstance(dset, DatasetDict):
            new_dset = DatasetDict()
            for split, dset_split in dset.items():
                shards = [
                    dset_split.shard(num_shards=self.num_proc, index=rank, contiguous=True)
                    for rank in range(self.num_proc)
                ]

                with get_context("spawn").Pool(self.num_proc) as pool:
                    results = [pool.apply_async(self.map_shard, kwds={"shard": shard}) for shard in shards]
                    transformed_shards = [result.get() for result in results]

                    pool.terminate()
                    pool.join()
                    del pool

                new_dset[split] = concatenate_datasets(transformed_shards)

                # Make sure the features match the expected API.
                assert new_dset[split].features == _FEATURES, f"Got: {new_dset[split].features}, expected: {_FEATURES}"
            dset = new_dset

        return dset


# ---- Dataset specific loader builder -----


class COCOLoaderBuilder(BaseLoaderBuilder, ABC):
    _DATASETS_NAME = "coco"

    def _load_dataset(self):
        return load_dataset(
            f"{Path(__file__).parent / 'local_loaders' / 'coco'}",
            split=self.split,
            use_auth_token=True,
        )

    def update_before_collapsing_meta(self, batch: Dict) -> Dict:
        # move `caption` to `text`
        batch["text"] = [sents["raw"] for sents in batch["sentences"]]
        return batch


class SBUCaptionsLoaderBuilder(DownloadImageLoaderBuilder):
    _DATASETS_NAME = "sbu_captions"

    def _load_dataset(self) -> Dataset:
        return load_dataset(self._DATASETS_NAME, split=self.split)


class LocalizedNarrativesOpenImagesLoaderBuilder(DownloadImageLoaderBuilder):
    _DATASETS_NAME = "localized_narratives__openimages"

    def _load_dataset(self) -> Dataset:
        return load_dataset(
            f"{Path(__file__).parent / 'local_loaders' / 'localized_narratives__openimages'}",
            split=self.split,
            use_auth_token=True,
        )


class LocalizedNarrativesCOCOLoaderBuilder(BaseLoaderBuilder, ABC):
    _DATASETS_NAME = "localized_narratives__coco"

    def _load_dataset(self) -> Dataset:
        return load_dataset(
            f"{Path(__file__).parent / 'local_loaders' / 'localized_narratives__coco'}",
            split=self.split,
            use_auth_token=True,
        )

    def update_before_collapsing_meta(self, batch: Dict) -> Dict:
        # move `caption` to `text`
        batch["text"] = batch["caption"]
        del batch["caption"]
        return batch


class LocalizedNarrativesFlickr30kLoaderBuilder(BaseLoaderBuilder, ABC):
    _DATASETS_NAME = "localized_narratives__flickr30k"

    def _load_dataset(self) -> Dataset:
        return load_dataset(
            f"{Path(__file__).parent / 'local_loaders' / 'localized_narratives__flickr30k'}",
            data_dir=get_m4_cache_dir() / "flickr30k",
            split=self.split,
            use_auth_token=True,
        )

    def update_before_collapsing_meta(self, batch: Dict) -> Dict:
        # move `caption` to `text`
        batch["text"] = batch["caption"]
        del batch["caption"]
        return batch


class LocalizedNarrativesADE20kLoaderBuilder(BaseLoaderBuilder, ABC):
    _DATASETS_NAME = "localized_narratives__ADE20k"

    def _load_dataset(self) -> Dataset:
        return load_dataset(
            f"{Path(__file__).parent / 'local_loaders' / 'localized_narratives__ADE20k'}",
            split=self.split,
            use_auth_token=True,
        )

    def update_before_collapsing_meta(self, batch: Dict) -> Dict:
        # move `caption` to `text`
        batch["text"] = batch["caption"]
        del batch["caption"]
        return batch


class VisualGenomeLoaderBuilder(BaseLoaderBuilder, ABC):
    _DATASETS_NAME = "visual_genome"

    def _load_dataset(self) -> Dataset:
        # Victor - Dirty as fuck loading the karpathy splits. but i am tired of this image pmd and i want to go fast
        karpathy_coco_file = get_m4_cache_dir() / "coco-captions" / "dataset_coco.json"
        with open(karpathy_coco_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
            invalid_images = {}
            for annotation in annotations["images"]:
                if annotation["split"] == "test" or annotation["split"] == "val":
                    invalid_images[int(annotation["cocoid"])] = 1
        self.invalid_images = invalid_images

        return load_dataset(self._DATASETS_NAME, "region_descriptions_v1.2.0", split=self.split)

    def update_before_collapsing_meta(self, batch: Dict) -> Dict:
        metadata_columns = self.get_batch_metadata_columns(batch)
        new_batch = {
            **{column: [] for column in metadata_columns},
            "image": [],
            "text": [],
        }
        for image, regions, coco_id, *values in zip(
            batch["image"], batch["regions"], batch["coco_id"], *[batch[column] for column in metadata_columns]
        ):
            if coco_id is None or int(coco_id) not in self.invalid_images:
                slices = [
                    (region["x"], region["x"] + region["width"], region["y"], region["y"] + region["height"])
                    for region in regions
                ]
                new_batch["image"] += [
                    image.crop((x_start, y_start, x_end, y_end)) for (x_start, x_end, y_start, y_end) in slices
                ]
                new_batch["text"] += [region["phrase"] for region in regions]
                for column, value in zip(metadata_columns, values):
                    new_batch[column] += [value for _ in regions]
        return new_batch


class Conceptual12MLoaderBuilder(DownloadImageLoaderBuilder):
    _DATASETS_NAME = "conceptual_12m"

    def _load_dataset(self) -> Dataset:
        return load_dataset(self._DATASETS_NAME, split=self.split)


class RedCapsLoaderBuilder(DownloadImageLoaderBuilder):
    _DATASETS_NAME = "red_caps"

    def _load_dataset(self) -> Dataset:
        return load_dataset(self._DATASETS_NAME, "all", split=self.split)

    def get_texts(self, batch: Dict) -> List[List[str]]:
        return batch["raw_caption"]


class YFCC100MLoaderBuilder(DownloadImageLoaderBuilder):
    _DATASETS_NAME = "yfcc100m"

    def _load_dataset(self) -> Dataset:
        return load_dataset(
            f"{Path(__file__).parent / 'local_loaders' / 'yfcc100m'}",
            split=self.split,
            use_auth_token=True,
        )
