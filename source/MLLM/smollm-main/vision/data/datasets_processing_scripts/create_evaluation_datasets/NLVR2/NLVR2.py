"""NVLR2 loading script."""


import json
import os

import datasets


_CITATION = """\
@article{DBLP:journals/corr/abs-2202-01994,
  author    = {Yamini Bansal and
               Behrooz Ghorbani and
               Ankush Garg and
               Biao Zhang and
               Maxim Krikun and
               Colin Cherry and
               Behnam Neyshabur and
               Orhan Firat},
  title     = {Data Scaling Laws in {NMT:} The Effect of Noise and Architecture},
  journal   = {CoRR},
  volume    = {abs/2202.01994},
  year      = {2022},
  url       = {https://arxiv.org/abs/2202.01994},
  eprinttype = {arXiv},
  eprint    = {2202.01994},
  timestamp = {Mon, 24 Oct 2022 10:21:23 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2202-01994.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
The Natural Language for Visual Reasoning corpora are two language grounding datasets containing natural language sentences grounded in images. The task is to determine whether a sentence is true about a visual input. The data was collected through crowdsourcings, and solving the task requires reasoning about sets of objects, comparisons, and spatial relations. This includes two corpora: NLVR, with synthetically generated images, and NLVR2, which includes natural photographs.
"""

_HOMEPAGE = "https://lil.nlp.cornell.edu/nlvr/"

_LICENSE = "CC BY 4.0"

_URL_JSON = "https://raw.githubusercontent.com/lil-lab/nlvr/master/nlvr2/data/"
_URL_IMG = "https://lil.nlp.cornell.edu/resources/NLVR2/"
_SPLITS = {
    "train": "train",
    "validation": "dev",
    "test": "test",
}


class NLVR2Dataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "identifier": datasets.Value("string"),
                    "sentence": datasets.Value("string"),
                    "left_image": datasets.Image(),
                    "right_image": datasets.Image(),
                    "label": datasets.features.ClassLabel(names=["True", "False"]),
                    "directory": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = {
            "default": {
                "train": os.path.join(_URL_JSON, f'{_SPLITS["train"]}.json'),
                "validation": os.path.join(_URL_JSON, f'{_SPLITS["validation"]}.json'),
                "test1": os.path.join(_URL_JSON, f'{_SPLITS["test"]}1.json'),
                "test2": os.path.join(_URL_JSON, f'{_SPLITS["test"]}2.json'),
            },
        }
        files_path = dl_manager.download_and_extract(urls)

        images_files = {
            "train": os.path.join(_URL_IMG, f'{_SPLITS["train"]}_img.zip'),
            "validation": os.path.join(_URL_IMG, f'{_SPLITS["validation"]}_img.zip'),
            "test1": os.path.join(_URL_IMG, f'{_SPLITS["test"]}1_img.zip'),
            "test2": os.path.join(_URL_IMG, f'{_SPLITS["test"]}2.zip'),
        }
        train_img_path = os.path.join(dl_manager.extract(images_files["train"]), "images", "train")
        validation_img_path = os.path.join(dl_manager.download_and_extract(images_files["validation"]), "dev")
        test1_img_path = os.path.join(dl_manager.download_and_extract(images_files["test1"]), "test1")
        test2_img_path = os.path.join(dl_manager.download_and_extract(images_files["test2"]), "test2")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"files_paths": [files_path[self.config.name]["train"]], "images_paths": [train_img_path]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "files_paths": [files_path[self.config.name]["validation"]],
                    "images_paths": [validation_img_path],
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "files_paths": [files_path[self.config.name]["test1"], files_path[self.config.name]["test2"]],
                    "images_paths": [test1_img_path, test2_img_path],
                },
            ),
        ]

    def _generate_examples(self, files_paths, images_paths):
        idx = 0
        for i, files_path in enumerate(files_paths):
            for line in open(files_path).readlines():
                ex = json.loads(line)
                common_img_identifier = ex["identifier"].split("-")
                left_img_identifier = (
                    f"{common_img_identifier[0]}-{common_img_identifier[1]}-{common_img_identifier[2]}-img0.png"
                )
                right_img_identifier = (
                    f"{common_img_identifier[0]}-{common_img_identifier[1]}-{common_img_identifier[2]}-img1.png"
                )
                if common_img_identifier[0] == "train":
                    directory = str(ex["directory"])
                    left_image_path = str(os.path.join(images_paths[i], directory, left_img_identifier))
                    right_image_path = str(os.path.join(images_paths[i], directory, right_img_identifier))
                else:
                    directory = "0"
                    left_image_path = str(os.path.join(images_paths[i], left_img_identifier))
                    right_image_path = str(os.path.join(images_paths[i], right_img_identifier))
                assert os.path.exists(left_image_path)
                assert os.path.exists(right_image_path)
                record = {
                    "identifier": ex["identifier"],
                    "sentence": ex["sentence"],
                    "left_image": left_image_path,
                    "right_image": right_image_path,
                    "label": ex["label"],
                    "directory": directory,
                }
                idx += 1
                yield idx, record
