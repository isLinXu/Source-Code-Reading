# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Localized narratives - ADE20k subset"""
import json
from pathlib import Path

import datasets


# TODO: @thomasw21
_CITATION = """"""

# TODO: @thomasw21
_DESCRIPTION = """"""

# TODO: @thomasw21
_HOMEPAGE = ""

# TODO: @thomasw21
_LICENSE = ""

_ANNOTATION_URLs = {
    "train": "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_ade20k_train_captions_multi.jsonl",
    "validation": (
        "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_ade20k_validation_captions_multi.jsonl"
    ),
}

_IMAGES_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

_SPLIT_MAP = {"train": "training", "validation": "validation"}

_FEATURES = datasets.Features(
    {
        "dataset_id": datasets.Value("string"),
        "image_id": datasets.Value("string"),
        "annotator_id": datasets.Value("int32"),
        "caption": datasets.Value("string"),
        "original_caption": datasets.Value("string"),
        "image": datasets.Image(),
    }
)


class LocalizedNarrativesADE20k(datasets.GeneratorBasedBuilder):
    """Builder for ADE20K subset of Localized Narratives."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        annotation_files = dl_manager.download(_ANNOTATION_URLs)
        image_base_dir = Path(dl_manager.download_and_extract(_IMAGES_URL)) / "ADEChallengeData2016" / "images"

        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={
                    "annotation_file": annotation_file,
                    "base_image_path": image_base_dir / _SPLIT_MAP[split_name],
                },
            )
            for split_name, annotation_file in annotation_files.items()
        ]

    def _generate_examples(self, annotation_file: str, base_image_path: Path):
        with open(annotation_file, "r", encoding="utf-8") as fi:
            for idx, line in enumerate(fi):
                annotation = json.loads(line)
                image_path = base_image_path / f"{annotation['image_id']}.jpg"
                yield idx, {
                    "dataset_id": annotation["dataset_id"],
                    "image_id": annotation["image_id"],
                    "annotator_id": annotation["annotator_id"],
                    "caption": annotation["caption"],
                    "original_caption": annotation["original_caption"],
                    "image": str(image_path.absolute()),
                }
