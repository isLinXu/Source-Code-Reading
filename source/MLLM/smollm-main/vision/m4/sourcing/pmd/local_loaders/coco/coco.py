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
"""COCO - only the subset relevant for PMD"""
import json
import os
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


_IMAGES_URLS = {
    "train": "http://images.cocodataset.org/zips/train2014.zip",
    "validation": "http://images.cocodataset.org/zips/val2014.zip",
}

_KARPATHY_FILES_URL = "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/caption_datasets.zip"

_SPLIT_MAP = {"train": "train2014", "validation": "val2014"}

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "filepath": datasets.Value("string"),
        "sentids": [datasets.Value("int32")],
        "filename": datasets.Value("string"),
        "imgid": datasets.Value("int32"),
        "split": datasets.Value("string"),
        "sentences": {
            "tokens": [datasets.Value("string")],
            "raw": datasets.Value("string"),
            "imgid": datasets.Value("int32"),
            "sentid": datasets.Value("int32"),
        },
        "cocoid": datasets.Value("int32"),
    }
)


class COCO(datasets.GeneratorBasedBuilder):
    """COCO - only the subset relavant for PMD."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        annotation_file = os.path.join(dl_manager.download_and_extract(_KARPATHY_FILES_URL), "dataset_coco.json")
        image_folders = {k: Path(v) for k, v in dl_manager.download_and_extract(_IMAGES_URLS).items()}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotation_file": annotation_file,
                    "image_folders": image_folders,
                    "split_key": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotation_file": annotation_file,
                    "image_folders": image_folders,
                    "split_key": "validation",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "annotation_file": annotation_file,
                    "image_folders": image_folders,
                    "split_key": "test",
                },
            ),
        ]

    def _generate_examples(self, annotation_file, image_folders, split_key):
        counter = 0
        with open(annotation_file, "r", encoding="utf-8") as fi:
            annotations = json.load(fi)

            for image_metadata in annotations["images"]:
                if split_key == "train":
                    if image_metadata["split"] != "train" and image_metadata["split"] != "restval":
                        continue
                elif split_key == "validation":
                    if image_metadata["split"] != "val":
                        continue
                elif split_key == "test":
                    if image_metadata["split"] != "test":
                        continue

                if "val2014" in image_metadata["filename"]:
                    image_path = image_folders["validation"] / _SPLIT_MAP["validation"]
                else:
                    image_path = image_folders["train"] / _SPLIT_MAP["train"]

                image_path = image_path / image_metadata["filename"]

                for caption in image_metadata["sentences"]:
                    yield counter, {
                        "image": str(image_path.absolute()),
                        "filepath": image_metadata["filename"],
                        "sentids": image_metadata["sentids"],
                        "filename": image_metadata["filename"],
                        "imgid": image_metadata["imgid"],
                        "split": image_metadata["split"],
                        "sentences": {
                            "tokens": caption["tokens"],
                            "raw": caption["raw"],
                            "imgid": caption["imgid"],
                            "sentid": caption["sentid"],
                        },
                        "cocoid": image_metadata["cocoid"],
                    }
                    counter += 1
