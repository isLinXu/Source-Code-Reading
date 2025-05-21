# coding=utf-8
# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
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

# Lint as: python3
"""Conceptual Captions dataset. JZ specific version."""

import csv
import json
import os
import textwrap
from functools import partial
from multiprocessing import Pool, cpu_count

import datasets
import PIL.Image
import pyarrow as pa
from datasets.utils import logging

from m4.sourcing.pmd import _FEATURES, get_jz_dataset_dir
from m4.sourcing.pmd.helpers import json_serializer


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

_DESCRIPTION = """\
Google's Conceptual Captions dataset has more than 3 million images, paired with natural-language captions.
In contrast with the curated style of the MS-COCO images, Conceptual Captions images and their raw descriptions are harvested from the web,
and therefore represent a wider variety of styles. The raw descriptions are harvested from the Alt-text HTML attribute associated with web images.
The authors developed an automatic pipeline that extracts, filters, and transforms candidate image/caption pairs, with the goal of achieving a balance of cleanliness,
informativeness, fluency, and learnability of the resulting captions.
"""

_HOMEPAGE = "http://data.statmt.org/cc-100/"

_LICENSE = """\
The dataset may be freely used for any purpose, although acknowledgement ofsq
Google LLC ("Google") as the data source would be appreciated. The dataset is
provided "AS IS" without any warranty, express or implied. Google disclaims all
liability for any damages, direct or indirect, resulting from the use of the
dataset.
"""

_CITATION = """\
@inproceedings{sharma2018conceptual,
  title = {Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning},
  author = {Sharma, Piyush and Ding, Nan and Goodman, Sebastian and Soricut, Radu},
  booktitle = {Proceedings of ACL},
  year = {2018},
}
"""

_ANNOTATION_LOCAL_PATHS = {
    "unlabeled": {
        "train": "Train_GCC-training.tsv",
        "validation": "Validation_GCC-1.1.0-Validation.tsv",
    },
    # "labeled": {
    #     "train": "https://storage.googleapis.com/conceptual-captions-v1-1-labels/Image_Labels_Subset_Train_GCC-Labels-training.tsv?_ga=2.234395421.-20118413.1607637118",
    # },
}

_DESCRIPTIONS = {
    "unlabeled": textwrap.dedent(
        """\
        The basic version of the dataset split into Training, Validation, and Test splits.
        The Training split consists of 3,318,333 image-URL/caption pairs, with a total number of 51,201 total token types in the captions (i.e., total vocabulary).
        The average number of tokens per captions is 10.3 (standard deviation of 4.5), while the median is 9.0 tokens per caption.
        The Validation split consists of 15,840 image-URL/caption pairs, with similar statistics.
        """
    ),
    # "labeled": textwrap.dedent(
    #     """\
    #     A subset of 2,007,090 image-URL/caption pairs from the training set with machine-generated image labels.
    #     The image labels are obtained using the Google Cloud Vision API.
    #     Each image label has a machine-generated identifier (MID) corresponding to the label's Google Knowledge Graph entry and a confidence score for its presence in the image.
    #     Note: 2,007,528 is the number of image-URL/caption pairs specified by the authors, but some rows are missing labels, so they are not included.
    #     """
    # ),
}


class JZConceptualCaptions(datasets.ArrowBasedBuilder):
    """Builder for Conceptual Captions dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig("unlabeled", version=VERSION, description=_DESCRIPTIONS["unlabeled"]),
        # datasets.BuilderConfig("labeled", version=VERSION, description=_DESCRIPTIONS["labeled"]),
    ]
    DEFAULT_CONFIG_NAME = "unlabeled"
    num_proc = cpu_count()
    CHUNK_SIZE = 1_000

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        annotation_data_paths = _ANNOTATION_LOCAL_PATHS[self.config.name]
        splits = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "annotations_file": get_jz_dataset_dir() / "conceptual_captions" / annotation_data_paths["train"]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "annotations_file": (
                        get_jz_dataset_dir() / "conceptual_captions" / annotation_data_paths["validation"]
                    )
                },
            ),
        ]
        return splits

    def _generate_examples(self, annotations_file):
        buffer = []
        with open(annotations_file, encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                # Sanity check
                assert len(row) == 3
                caption, image_url, local_path = row
                if local_path == "N/A":
                    image_path = None
                else:
                    image_path = get_jz_dataset_dir() / "conceptual_captions" / local_path

                buffer.append(
                    {
                        "image_path": image_path,
                        "text": caption,
                        "source": "conceptual_captions",
                        "meta": json.dumps({"image_url": image_url}, default=json_serializer, indent=2),
                    }
                )
                if len(buffer) == self.CHUNK_SIZE:
                    yield buffer
                    buffer = []
            if len(buffer) > 0:
                yield buffer

    def _generate_tables(self, annotations_file):
        rows_iterator = self._generate_examples(annotations_file)

        counter_images_not_found = 0
        count_images_opening_error = 0
        counter_images_not_verified = 0
        with Pool(self.num_proc) as pool:
            tables_iterator = pool.imap(
                partial(self._create_table),
                rows_iterator,
                chunksize=1,
            )
            for idx, (not_found, opening_error, not_verified, table) in enumerate(tables_iterator):
                yield idx, table
                counter_images_not_found += not_found
                count_images_opening_error += opening_error
                counter_images_not_verified += not_verified
        logger.warning(f"Skipped {counter_images_not_found} images (not found).")
        logger.warning(f"Skipped {count_images_opening_error} images (opening error).")
        logger.warning(f"Skipped {counter_images_not_verified} images (not verified).")

    def _create_table(self, examples):
        counter_images_not_found = 0
        count_images_opening_error = 0
        counter_images_not_verified = 0
        result = []

        for ex in examples:
            image_path = ex["image_path"]
            if image_path is not None and os.path.exists(image_path):
                try:
                    image = PIL.Image.open(image_path)
                except PIL.UnidentifiedImageError:
                    count_images_opening_error += 1
                    # logger.warning(f"Loading image with path {image_path} returned an `PIL.UnidentifiedImageError`. Skipping this image.")
                    image = None
                except PIL.Image.DecompressionBombError:
                    count_images_opening_error += 1
                    # logger.warning(f"Loading image with path {image_path} returned an `PIL.Image.DecompressionBombError`. Skipping this image")
                    image = None
                except Exception:
                    count_images_opening_error += 1
                    # logger.warning(f"Loading image with path {image_path} returned an unexpected error: {e}. Skipping this image")
                    image = None

                try:
                    image.verify()
                except Exception:
                    counter_images_not_verified += 1
                    # logger.warning(f"Verifying image with path {image_path} raised an error {e}. Skipping this image.")
                    image = None
            else:
                counter_images_not_found += 1
                # logger.warning(f"Image with path {image_path} could not be found. Skipping this image.")
                image = None
            del ex["image_path"]
            ex["image"] = image
            if image is not None:
                image.close()
            result.append(ex)

        return (
            counter_images_not_found,
            count_images_opening_error,
            counter_images_not_verified,
            pa.table(_FEATURES.encode_batch(self._recast(result))),
        )

    def _recast(self, examples):
        return {
            "image": [ex["image"] for ex in examples],
            "text": [ex["text"] for ex in examples],
            "source": [ex["source"] for ex in examples],
            "meta": [ex["meta"] for ex in examples],
        }
