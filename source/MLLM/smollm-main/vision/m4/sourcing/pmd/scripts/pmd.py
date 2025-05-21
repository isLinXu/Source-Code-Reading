import math
import os
from multiprocessing import cpu_count
from typing import List, Optional

from datasets import DatasetDict

from m4.sourcing.pmd.loader_builder import (
    BaseLoaderBuilder,
    COCOLoaderBuilder,
    Conceptual12MLoaderBuilder,
    LocalizedNarrativesADE20kLoaderBuilder,
    LocalizedNarrativesCOCOLoaderBuilder,
    LocalizedNarrativesFlickr30kLoaderBuilder,
    LocalizedNarrativesOpenImagesLoaderBuilder,
    RedCapsLoaderBuilder,
    SBUCaptionsLoaderBuilder,
    VisualGenomeLoaderBuilder,
    YFCC100MLoaderBuilder,
)
from m4.utils.datasets.get_self_contained_ds import process_ds_wrapped


def build_pmd_subsets(
    split: Optional[str] = None,
    num_proc: int = 16,
    num_threads_per_proc: int = 4,
    retries: int = 3,
    offline_mode: bool = False,
):
    return DatasetBuilderConfig(
        dataset_names=[
            "coco",
            "sbu_captions",
            "localized_narratives__openimages",
            "localized_narratives__flickr30k",
            "localized_narratives__ADE20k",
            "localized_narratives__coco",
            "visual_genome",
            "conceptual_12m",
            "red_caps",
            "yfcc100m",
        ],
        split=split,
        num_threads_per_proc=num_threads_per_proc,
        num_proc=num_proc,
        retries=retries,
        offline_mode=offline_mode,
    ).build_shard_and_save_to_disk()


class DatasetBuilderConfig:
    def __init__(
        self,
        dataset_names: List[str],
        split: Optional[str],
        num_proc: int,
        num_threads_per_proc: int,
        retries: int,
        offline_mode: bool,
    ):
        self.dataset_names = dataset_names
        self.split = split
        self.num_proc = num_proc
        self.num_threads_per_proc = num_threads_per_proc
        self.retries = retries
        self.offline_mode = offline_mode

    def __get_builders__(self) -> List[BaseLoaderBuilder]:
        all_builders = {
            "coco": COCOLoaderBuilder(split=self.split, num_proc=self.num_proc),
            "sbu_captions": SBUCaptionsLoaderBuilder(
                split=self.split,
                num_proc=self.num_proc,
                num_threads_per_proc=self.num_threads_per_proc,
                retries=self.retries,
                offline_mode=self.offline_mode,
            ),
            "localized_narratives__openimages": LocalizedNarrativesOpenImagesLoaderBuilder(
                split=self.split,
                num_proc=self.num_proc,
                num_threads_per_proc=self.num_threads_per_proc,
                retries=self.retries,
                offline_mode=self.offline_mode,
            ),
            "localized_narratives__flickr30k": LocalizedNarrativesFlickr30kLoaderBuilder(
                split=self.split, num_proc=self.num_proc
            ),
            "localized_narratives__ADE20k": LocalizedNarrativesADE20kLoaderBuilder(
                split=self.split, num_proc=self.num_proc
            ),
            "localized_narratives__coco": LocalizedNarrativesCOCOLoaderBuilder(
                split=self.split, num_proc=self.num_proc
            ),
            "visual_genome": VisualGenomeLoaderBuilder(
                batch_size=100,  # PIL has to load the image to actually crop it ...
                split=self.split,
                num_proc=self.num_proc,
            ),
            "conceptual_12m": Conceptual12MLoaderBuilder(
                split=self.split,
                num_proc=self.num_proc,
                num_threads_per_proc=self.num_threads_per_proc,
                retries=self.retries,
                offline_mode=self.offline_mode,
            ),
            "red_caps": RedCapsLoaderBuilder(
                split=self.split,
                num_proc=self.num_proc,
                num_threads_per_proc=self.num_threads_per_proc,
                retries=self.retries,
                offline_mode=self.offline_mode,
            ),
            "yfcc100m": YFCC100MLoaderBuilder(
                split=self.split,
                num_proc=self.num_proc,
                num_threads_per_proc=self.num_threads_per_proc,
                retries=self.retries,
                offline_mode=self.offline_mode,
            ),
        }

        return [(dataset_name, all_builders[dataset_name]) for dataset_name in self.dataset_names]

    def build_shard_and_save_to_disk(self):
        """Build, shard and save"""
        builders = self.__get_builders__()
        for dataset_name, builder in builders:
            dataset = builder.build()

            print("Start converting the images to bytes.")
            dataset = process_ds_wrapped(dataset, batch_size=1_000, num_proc=self.num_proc)

            print("Start saving shards.")
            if isinstance(dataset, DatasetDict):
                for split_name, dset in dataset.items():
                    nb_of_shards = math.ceil(len(dset) / 50_000)
                    shards = [
                        dset.shard(num_shards=nb_of_shards, index=i, contiguous=True) for i in range(nb_of_shards)
                    ]
                    for i, shard in enumerate(shards):
                        shard.save_to_disk(
                            f"{os.environ['HOME']}/general_pmd/image/{dataset_name}/{split_name}/{i:05}-{nb_of_shards:05}"
                        )
            else:
                raise ValueError(f"`datasets` is of type {type(dataset)} which is not supported yet.")


if __name__ == "__main__":
    build_pmd_subsets(
        num_proc=int(
            0.6 * cpu_count()
        ),  # Putting this 60% coefficient to avoid OOM errors (which are pretty frequent with extremely large sets)
        num_threads_per_proc=8,  # Value tuned on a 16 CPU cores machine
        retries=1,
    )
