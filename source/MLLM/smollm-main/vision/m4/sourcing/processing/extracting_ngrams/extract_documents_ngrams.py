import argparse
import logging
import sys
from typing import List

from transformers import AutoTokenizer

from utils import get_ngrams


parser = argparse.ArgumentParser(description="Extract the ngrams")
parser.add_argument("--filepath", type=str, required=True, help="The filepath to the subshard.")
parser.add_argument(
    "--nb_docs_per_subshard",
    type=int,
    default=1_000,
    help="The number of documents per subshard.",
)
args = parser.parse_args()

SHARD_NAME = 0

tokenizer = AutoTokenizer.from_pretrained("gpt2")
vocab_size = tokenizer.vocab_size


def process_document(text: str):
    text = text.strip()
    input_ids = tokenizer(text, return_attention_mask=False, return_tensors="np")["input_ids"][0]

    # Dirty implem. TODO: update
    ngrams = (
        get_ngrams(ids=input_ids, voc_size=vocab_size, n=1)
        + get_ngrams(ids=input_ids, voc_size=vocab_size, n=2)
        + get_ngrams(ids=input_ids, voc_size=vocab_size, n=3)
        + get_ngrams(ids=input_ids, voc_size=vocab_size, n=4)
    )
    return ngrams


def process_docs(documents: List[str], subshard_idx: int):
    doc_counter = subshard_idx * args.nb_docs_per_subshard + 1
    for doc in documents:
        try:
            ngrams = process_document(doc)
            for id in ngrams:
                print(f"{SHARD_NAME},{doc_counter},{id}")
        except ValueError:
            logging.warning(f"Document nb {doc_counter} is invalid. Skipping.\nDoc = {doc}.")
            # TODO: get some way of logging number of skipped documents (not at the process level tough...)
        doc_counter += 1


if __name__ == "__main__":
    input_filepath = args.filepath
    subshard_idx = int(input_filepath.split(".")[-1])
    with open(input_filepath, "r") as file:
        process_docs(file, subshard_idx)
    sys.stdout.flush()
