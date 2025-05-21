import hashlib
import os
import sys
from typing import List

import datasets
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


IDX_JOB = int(sys.argv[1])
ALL_NAMES_DS = [
    "ai2d",
    "aokvqa",
    "chartqa",
    "clevr",
    "cocoqa",
    "datikz",
    "diagram_image_to_text",
    "docvqa",
    "dvqa",
    "geomverse",
    "hateful_memes",
    "iam",
    "iconqa",
    "infographic_vqa",
    "intergps",
    "mimic_cgd",
    "nlvr2",
    "ocrvqa",
    "okvqa",
    "scienceqa",
    "screen2words",
    "spot_the_diff",
    "st_vqa",
    "tallyqa",
    "textcaps",
    "textvqa",
    "visual7w",
    "visualmrc",
    "vqarad",
    "vqav2",
    "vsr",
]
NAME_DS = ALL_NAMES_DS[IDX_JOB]

COMMON_FOLDER_DS = "/fsx/hugo/fine_tuning_datasets_to_be_merged"
COMMON_FOLDER_SAVE_DS = "/fsx/hugo/fine_tuning_datasets_merge_image_individual"

NUM_PROC = 48

FEATURES = datasets.Features(
    {
        "images": datasets.Sequence(datasets.Image(decode=True)),
        "texts": [
            {
                "user": datasets.Value("string"),
                "assistant": datasets.Value("string"),
                "source": datasets.Value("string"),
            }
        ],
    }
)

TOKENIZER = "HuggingFaceM4/Mistral-7B-v0.1-tokenizer-pad-is-unk"


class MergeOnSameImage:
    def __init__(self, name_ds: str) -> None:
        self.name_ds = name_ds

    def __call__(self) -> None:
        self.load_dataset()
        self.compute_hash_images()
        self.create_image_hash_to_ds_index()
        self.create_new_ds()
        self.save_new_ds()
        self.print_stats()

    def load_dataset(self):
        path_ds = os.path.join(COMMON_FOLDER_DS, self.name_ds)
        self.ds = load_from_disk(path_ds)
        print("Dataset loaded")

    def compute_hash_images(self):
        def map_compute_hash_images(example):
            image_hash = ""
            images = example["images"]
            for image in images:
                md5hash = hashlib.md5(image.tobytes()).hexdigest()
                image_hash += md5hash
            example["image_hash"] = image_hash
            return example

        self.ds = self.ds.map(map_compute_hash_images, num_proc=NUM_PROC)
        print("Finished computing the hash of the images")

    def create_image_hash_to_ds_index(self):
        all_image_hashes = self.ds["image_hash"]
        self.image_hash_to_ds_index = {}
        for idx_image_hash, image_hash in enumerate(all_image_hashes):
            self.image_hash_to_ds_index[image_hash] = self.image_hash_to_ds_index.get(image_hash, []) + [
                idx_image_hash
            ]
        print("Finished creating the mapping to go from an image hash to the associate indices in the dataset")

    def create_new_ds(self):
        dict_new_ds = {"images": [], "texts": []}
        all_texts = self.ds["texts"]
        self.all_answers = [sub_el["assistant"] for el in all_texts for sub_el in el]

        for list_indices in tqdm(list(self.image_hash_to_ds_index.values())):
            common_image = self.ds[list_indices[0]]["images"]
            dict_new_ds["images"].append(common_image)

            texts = [all_texts[idx] for idx in list_indices]
            texts = [sub_el for el in texts for sub_el in el]
            dict_new_ds["texts"].append(texts)

        # self.new_ds = datasets.Dataset.from_dict(dict_new_ds, features=FEATURES)

        def data_generator():
            for images, texts in zip(dict_new_ds["images"], dict_new_ds["texts"]):
                yield {"images": images, "texts": texts}

        self.new_ds = datasets.Dataset.from_generator(data_generator, features=FEATURES, writer_batch_size=100)
        print("Finished creating the new dataset")

    def save_new_ds(self):
        path_save_new_ds = os.path.join(COMMON_FOLDER_SAVE_DS, self.name_ds)
        self.new_ds.save_to_disk(path_save_new_ds, num_proc=NUM_PROC)
        print("Finished saving the new dataset")

    def print_stats(self):
        number_tokens_answers = self.tokenize_and_count_tokens(list_texts=self.all_answers)
        print(f"Dataset: {self.name_ds}")
        print(f"Number of different images: {self.new_ds.num_rows}")
        print(f"Number of question/answer pairs: {len(self.all_answers)}")
        frac_qa_pairs_num_images = len(self.all_answers) / self.new_ds.num_rows
        print(f"Average number of question/answer pairs per image: {frac_qa_pairs_num_images}")
        print(f"Total number of tokens for the answers: {number_tokens_answers}")
        mean_num_tok_per_answer = number_tokens_answers / len(self.all_answers)
        print(f"Average number of tokens per answer: {mean_num_tok_per_answer}")

    def tokenize_and_count_tokens(self, list_texts: List[str]) -> int:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        list_texts_tokenized = [tokenizer(text)["input_ids"] for text in list_texts]
        number_tokens = sum([len(tokens) for tokens in list_texts_tokenized])
        return number_tokens


if __name__ == "__main__":
    merge_on_same_image = MergeOnSameImage(name_ds=NAME_DS)
    merge_on_same_image()
