from datasets import load_dataset
from selectolax.parser import HTMLParser


def load_dataset_html(shuffle=False, buffer_size=10000, seed=42):
    dataset = load_dataset(
        "bs-modeling-metadata/c4-en-html-with-metadata",
        streaming=True,
        split="train",
        use_auth_token=True,
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed)
    dataset = iter(dataset)
    return dataset


def make_selectolax_tree(html_str):
    selectolax_tree = HTMLParser(html_str)
    return selectolax_tree
