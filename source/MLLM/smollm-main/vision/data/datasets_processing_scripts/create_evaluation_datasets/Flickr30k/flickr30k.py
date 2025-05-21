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
"""Flickr 30k"""
import json
import os

import datasets


_CITATION = """
@article{young-etal-2014-image,
    title = "From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions",
    author = "Young, Peter  and
      Lai, Alice  and
      Hodosh, Micah  and
      Hockenmaier, Julia",
    journal = "Transactions of the Association for Computational Linguistics",
    volume = "2",
    year = "2014",
    address = "Cambridge, MA",
    publisher = "MIT Press",
    url = "https://aclanthology.org/Q14-1006",
    doi = "10.1162/tacl_a_00166",
    pages = "67--78",
    abstract = "We propose to use the visual denotations of linguistic expressions (i.e. the set of images they describe) to define novel denotational similarity metrics, which we show to be at least as beneficial as distributional similarities for two tasks that require semantic inference. To compute these denotational similarities, we construct a denotation graph, i.e. a subsumption hierarchy over constituents and their denotations, based on a large corpus of 30K images and 150K descriptive captions.",
}
"""

# TODO: Victor
_DESCRIPTION = """"""

_HOMEPAGE = "https://shannon.cs.illinois.edu/DenotationGraph/"

# TODO: Victor
_LICENSE = ""

_ANNOTATION_URL = "http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "filename": datasets.Value("string"),
        "imgid": datasets.Value("int32"),
        "sentids": [datasets.Value("int32")],
        "sentences": [datasets.Value("string")],
        "sentences_original": [
            {
                "tokens": [datasets.Value("string")],
                "raw": datasets.Value("string"),
                "imgid": datasets.Value("int32"),
                "sentid": datasets.Value("int32"),
            }
        ],
    }
)


class Flickr30k(datasets.GeneratorBasedBuilder):
    """Flick30k."""

    @property
    def manual_download_instructions(self):
        return """\
    You need to go to http://shannon.cs.illinois.edu/DenotationGraph/data/index.html,
    and manually download the dataset ("Flickr 30k images."). Once it is completed,
    a file named `flickr30k-images.tar.gz` will appear in your Downloads folder
    or whichever folder your browser chooses to save files to.
    Then, the dataset can be loaded using the following command `datasets.load_dataset("flickr30k", data_dir="<path/to/folder>")`.
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
        annotations_zip = dl_manager.download_and_extract(_ANNOTATION_URL)
        annotation_path = os.path.join(annotations_zip, "dataset_flickr30k.json")
        images_path = os.path.join(
            dl_manager.extract(os.path.join(dl_manager.manual_dir, "flickr30k-images.tar.gz")), "flickr30k-images"
        )

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={"annotation_path": annotation_path, "images_path": images_path, "split_name": name},
            )
            for (split, name) in [
                (datasets.Split.TRAIN, "train"),
                (datasets.Split.VALIDATION, "val"),
                (datasets.Split.TEST, "test"),
            ]
        ]

    def _generate_examples(self, annotation_path, images_path, split_name):
        counter = 0
        print(annotation_path)
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for elem in data["images"]:
                if elem["split"] != split_name:
                    continue
                assert os.path.exists(os.path.join(images_path, elem["filename"]))
                yield counter, {
                    "image": os.path.join(images_path, elem["filename"]),
                    "filename": elem["filename"],
                    "imgid": elem["imgid"],
                    "sentids": elem["sentids"],
                    "sentences_original": elem["sentences"],
                    "sentences": [ans["raw"] for ans in elem["sentences"]],
                }
                counter += 1
