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
"""Localized narratives - Flickr 30k subset"""
import glob
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

_ANNOTATION_URLs = {
    "train": "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_flickr30k_train_captions_multi.jsonl",
    "validation": (
        "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_flickr30k_val_captions_multi.jsonl"
    ),
    "test": "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_flickr30k_test_captions_multi.jsonl",
}

_LOCAL_IMAGE_FOLDER_NAME = "flickr30k-images"

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


class LocalizedNarrativesFlickr30K(datasets.GeneratorBasedBuilder):
    """Builder for Flickr30k subset of Localized Narratives."""

    @property
    def manual_download_instructions(self):
        return """\
    You need to go to http://shannon.cs.illinois.edu/DenotationGraph/data/index.html,
    and manually download the dataset ("Flickr 30k images."). Once it is completed,
    a file named `flickr30k-images.tar.gz` will appear in your Downloads folder
    or whichever folder your browser chooses to save files to. You then have
    to unzip the file and move `flickr30k-images` under <path/to/folder>.
    The <path/to/folder> can e.g. be "~/manual_data".
    dataset can then be loaded using the following command `datasets.load_dataset("flickr30k", data_dir="<path/to/folder>")`.
    """

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

        path_to_manual_file = Path(dl_manager.manual_dir) / _LOCAL_IMAGE_FOLDER_NAME
        if not path_to_manual_file.exists():
            raise FileNotFoundError(
                f"{path_to_manual_file} does not exist. Make sure you insert a manual dir via"
                " `datasets.load_dataset('flickr30k', data_dir=...)` that includes file name"
                f" {_LOCAL_IMAGE_FOLDER_NAME}. Manual download instructions: {self.manual_download_instructions}"
            )

        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={"annotation_file": annotation_file, "base_image_path": path_to_manual_file},
            )
            for split_name, annotation_file in annotation_files.items()
        ]

    def _generate_examples(self, annotation_file: str, base_image_path: Path):
        annotations = {}
        with open(annotation_file, "r", encoding="utf-8") as fi:
            for line in fi:
                annotation = json.loads(line.strip())
                image_id = str(annotation["image_id"])
                if image_id not in annotations:
                    annotations[image_id] = []
                annotations[image_id].append(annotation)

        counter = 0
        for path in glob.glob(f"{base_image_path}/*.jpg"):
            root, _ = os.path.splitext(path)
            imageid = os.path.basename(root)
            if imageid not in annotations:
                continue
            for ann in annotations[imageid]:
                yield counter, {
                    "dataset_id": ann["dataset_id"],
                    "image_id": ann["image_id"],
                    "annotator_id": ann["annotator_id"],
                    "caption": ann["caption"],
                    "original_caption": ann["original_caption"],
                    "image": path,
                }
                counter += 1
