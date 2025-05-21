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
"""Localized narratives - Open Images subset"""
import json
from copy import deepcopy

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
    "train": (
        "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_open_images_train_v6_captions_multi.jsonl"
    ),
    "validation": (
        "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_open_images_validation_captions_multi.jsonl"
    ),
    "test": "https://huggingface.co/datasets/facebook/pmd/resolve/main/data/ln_open_images_test_captions_multi.jsonl",
}

# # TODO @thomasw21: We download ALL Open-images from a tar file (that also has bounding boxes but oh well), and we filter only those we have annotations for
# _IMAGES = {}

_FEATURES = datasets.Features(
    {
        "dataset_id": datasets.Value("string"),
        "image_id": datasets.Value("string"),
        "annotator_id": datasets.Value("int32"),
        "caption": datasets.Value("string"),
        "original_caption": datasets.Value("string"),
        "image_url": datasets.Value("string"),
    }
)


def _split_to_single_caption(annotations):
    """This function is mainly used in Localized Narratives where a paragraph can contain
    multiple relevant captions to a single image. We split the paragraph into multiple
    captions and then return each as an individual sample.
    """
    extended = []
    for annotation in annotations:
        captions = annotation["caption"].split(".")
        for caption in captions:
            if len(caption.strip()) == 0 or caption.strip() == ".":
                continue
            current = deepcopy(annotation)
            current["caption"] = caption.strip() + "."
            current["original_caption"] = annotation["caption"]
            extended.append(current)
    return extended


class LocalizedNarrativesOpenImages(datasets.GeneratorBasedBuilder):
    """Builder for Open Images subset of Localized Narratives."""

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
        return [
            datasets.SplitGenerator(
                name=split_name,
                gen_kwargs={"annotation_file": annotation_file, "split": split_name},
            )
            for split_name, annotation_file in annotation_files.items()
        ]

    def _generate_examples(self, annotation_file: str, split: str):
        counter = 0
        with open(annotation_file, "r", encoding="utf-8") as fi:
            for idx, line in enumerate(fi):
                annotation = json.loads(line)
                yield counter, {
                    "dataset_id": annotation["dataset_id"],
                    "image_id": annotation["image_id"],
                    "annotator_id": annotation["annotator_id"],
                    "caption": annotation["caption"],
                    "original_caption": annotation["original_caption"],
                    "image_url": f"https://s3.amazonaws.com/open-images-dataset/{split}/{annotation['image_id']}.jpg",
                }
                counter += 1
