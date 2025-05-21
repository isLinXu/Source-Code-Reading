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

"""LAION 2B EN loading script."""


import glob
import json
import os
import tarfile

import datasets


_CITATION = """\
@misc{https://doi.org/10.48550/arxiv.2210.08402,
  doi = {10.48550/ARXIV.2210.08402},
  url = {https://arxiv.org/abs/2210.08402},
  author = {Schuhmann, Christoph and Beaumont, Romain and Vencu, Richard and Gordon, Cade and Wightman, Ross and Cherti, Mehdi and Coombes, Theo and Katta, Aarush and Mullis, Clayton and Wortsman, Mitchell and Schramowski, Patrick and Kundurthy, Srivatsa and Crowson, Katherine and Schmidt, Ludwig and Kaczmarczyk, Robert and Jitsev, Jenia},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {LAION-5B: An open large-scale dataset for training next generation image-text models},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
"""

_DESCRIPTION = """\
This is the 2B subset of English pairs from the LAION 5B dataset.
"""

_HOMEPAGE = "https://laion.ai/blog/laion-5b/"

_LICENSE = "CC BY-NC-SA (Attribution-NonCommercial-ShareAlike)"

DL_DATASET_PATH = "./large_files/laion_tar_files/"  # Path to the downloaded dataset in webdataset format


class LAION2BENDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "text": datasets.Value("string"),
                "source": datasets.Value("string"),
                "meta": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, _):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={},
            ),
        ]

    def _generate_examples(self):
        idx = -1
        tar_paths = glob.glob(os.path.join(DL_DATASET_PATH, "*.tar"))
        for tar_path in tar_paths:
            with tarfile.open(tar_path) as tar_file:
                tar_members = tar_file.getmembers()
                name_to_meta = {}
                name_to_image = {}
                name_to_text = {}
                for tar_member in tar_members:
                    if tar_member.name.endswith(".jpg"):
                        name = tar_member.name.replace(".jpg", "")
                        tar_member_file = tar_file.extractfile(tar_member)
                        img = tar_member_file.read()
                        tar_member_file.close()
                        name_to_image[name] = {
                            "path": None,
                            "bytes": img,
                        }
                    elif tar_member.name.endswith(".json"):
                        name = tar_member.name.replace(".json", "")
                        tar_member_file = tar_file.extractfile(tar_member)
                        json_val = json.loads(tar_member_file.read())
                        tar_member_file.close()
                        status = json_val["status"]
                        if status == "success":
                            name_to_meta[name] = json.dumps(json_val)
                    elif tar_member.name.endswith(".txt"):
                        name = tar_member.name.replace(".txt", "")
                        tar_member_file = tar_file.extractfile(tar_member)
                        captions = tar_member_file.read().decode("utf-8")
                        tar_member_file.close()
                        name_to_text[name] = captions
                for name in name_to_image:
                    if (name in name_to_text) and (name in name_to_meta):
                        example = {
                            "image": name_to_image[name],
                            "text": name_to_text[name],
                            "source": "laion_2b_en",
                            "meta": name_to_meta[name],
                        }
                        idx += 1
                        yield idx, example
