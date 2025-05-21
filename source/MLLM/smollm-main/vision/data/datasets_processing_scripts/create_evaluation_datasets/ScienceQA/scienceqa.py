# Copyright 2020 The HuggingFace Datasets Authors.
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

"""ScienceQA loading script."""


import json
import os
import string
from pathlib import Path

import datasets


_CITATION = """\
@inproceedings{lu2022learn,
    title={Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering},
    author={Lu, Pan and Mishra, Swaroop and Xia, Tony and Qiu, Liang and Chang, Kai-Wei and Zhu, Song-Chun and Tafjord, Oyvind and Clark, Peter and Ashwin Kalyan},
    booktitle={The 36th Conference on Neural Information Processing Systems (NeurIPS)},
    year={2022}
}
"""

_DESCRIPTION = """\
This is the ScienceQA dataset.
"""

_HOMEPAGE = "https://scienceqa.github.io/"

_LICENSE = "CC BY-NC-SA (Attribution-NonCommercial-ShareAlike)"

_URLS = {
    "pid_splits": "https://drive.google.com/uc?id=1OXlNBuW74dsrwYZIpQMshFqxkjcMPPgV&export=download",
    "problems": "https://drive.google.com/uc?id=1nJ86OLnF2C6eDoi5UOAdTAS5Duc0wuTl&export=download",
    "train": "https://drive.google.com/uc?id=1swX4Eei1ZqrXRvM-JAZxN6QVwcBLPHV8&export=download",
    "val": "https://drive.google.com/uc?id=1ijThWZc1tsoqGrOCWhYYj1HUJ48Hl8Zz&export=download",
    "test": "https://drive.google.com/uc?id=1eyjFaHxbvEJZzdZILn3vnTihBNDmKcIj&export=download",
}

_SUB_FOLDER_OR_FILE_NAME = {
    "pid_splits": "pid_splits.json",
    "problems": "problems.json",
    "train": "train",
    "val": "val",
    "test": "test",
}

# For some reasons I couldn't open these files after downloading them (successfully) with datasets
# so I downloaded these files on JZ and hard coded the paths...
"""
    Commands to run to get problems.json and pid_splits.json in local folders:
        wget "https://drive.google.com/uc?export=download&id=1nJ86OLnF2C6eDoi5UOAdTAS5Duc0wuTl&confirm=yes" -O problems.json
        wget "https://drive.google.com/uc?export=download&id=1OXlNBuW74dsrwYZIpQMshFqxkjcMPPgV" -O pid_splits.json
"""
LOCAL_FOLDER_PATH = {
    "pid_splits": "/Users/leotronchon/Documents/HuggingFace/datasets/pid_splits.json",
    "problems": "/Users/leotronchon/Documents/HuggingFace/datasets/problems.json",
}


class ScienceQADataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            # Previous features
            # {
            #    "question": datasets.Value("string"),
            #    "choices": datasets.Sequence(datasets.Value("string")),
            #    "answer": datasets.Value("int32"),
            #    "hint": datasets.Value("string"),
            #    "image": datasets.Image(),
            #    "task": datasets.Value("string"),
            #    "grade": datasets.Value("string"),
            #    "subject": datasets.Value("string"),
            #    "topic": datasets.Value("string"),
            #    "category": datasets.Value("string"),
            #    "skill": datasets.Value("string"),
            #    "lecture": datasets.Value("string"),
            #    "solution": datasets.Value("string"),
            #    "split": datasets.Value("string"),
            # }
            {
                "image": datasets.Image(),
                "question": datasets.Value("string"),
                "hint": datasets.Value("string"),
                "context": datasets.Value("string"),
                "lecture": datasets.Value("string"),
                "solution": datasets.Value("string"),
                "label": datasets.features.ClassLabel(names=["A", "B", "C", "D", "E"]),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URLS)
        gen_kwargs = {}
        for split_name in ["train", "val", "test"]:
            gen_kwargs_per_split = {}
            gen_kwargs_per_split["pid_splits_path"] = (
                Path(data_dir["pid_splits"]) / _SUB_FOLDER_OR_FILE_NAME["pid_splits"]
            )
            gen_kwargs_per_split["problems_path"] = Path(data_dir["problems"]) / _SUB_FOLDER_OR_FILE_NAME["problems"]
            gen_kwargs_per_split["images_path"] = Path(data_dir[split_name]) / _SUB_FOLDER_OR_FILE_NAME[split_name]
            gen_kwargs_per_split["split_name"] = split_name
            gen_kwargs[split_name] = gen_kwargs_per_split
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=gen_kwargs["train"],
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs=gen_kwargs["val"],
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs=gen_kwargs["test"],
            ),
        ]

    def _generate_examples(self, pid_splits_path, problems_path, images_path, split_name):
        # Ideal solution would be this:
        # pid_splits = json.load(open(pid_splits_path, "r"))
        # problems = json.load(open(problems_path, "r"))
        # But for some reasons I couldn't open these files after downloading them (successfully) with datasets
        # so I downloaded these files on JZ and hard coded the paths...
        pid_splits = json.load(open(LOCAL_FOLDER_PATH["pid_splits"], "r"))
        problems = json.load(open(LOCAL_FOLDER_PATH["problems"], "r"))

        for idx, key in enumerate(pid_splits[split_name]):
            example = problems[key]
            if example["image"]:
                example["image"] = os.path.join(images_path, key, example["image"])

            # Modification from here
            if example["image"]:
                interesting_keys = ["image", "question", "hint", "context", "lecture", "solution", "label"]

                example["label"] = example["answer"]
                example["hint"] = example["hint"].replace("\n\n", "\n")
                example["question"] = example["question"].replace("\n\n", "\n")

                for idx_choice in range(len(example["choices"])):
                    example["choices"][idx_choice] = example["choices"][idx_choice].replace("\n\n", "\n")

                example["question"] = f"Question: {example['question']}\n"

                context = ""
                for idx_choice, choice in enumerate(example["choices"]):
                    context += f"({string.ascii_uppercase[idx_choice]}) {choice} "
                example["context"] = context

                if example["hint"]:
                    example["hint"] = f"Context: {example['hint']}\n"

                if example["lecture"]:
                    example["lecture"] = f"BECAUSE: {example['lecture']}"

                if example["solution"]:
                    example["solution"] = f"{example['solution']}"

                example = {key: val for key, val in example.items() if key in interesting_keys}
                yield idx, example
