import re

import datasets

from m4.training.dataset_utils import get_webdataset
from m4.training.types import DatasetTypes


base_path = "/fsx/leo/fine_tuning_datasets/concat_chatty_tar/shard_{index}.tar"

# Generate paths using a list comprehension
webdataset_paths = [base_path.format(index=i) for i in range(0, 1785)]


FEATURES = datasets.Features(
    {
        "__key__": datasets.Value("string"),
        "__url__": datasets.Value("string"),
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
combined_dataset = get_webdataset(
    urls=webdataset_paths,
    ds_type=DatasetTypes.SFT,
    batch_size=10,
    shuffle_initial_urls_list=False,
    shuffle_before_split_by_node_buffer_size=(None),
    shuffle_before_split_by_worker_buffer_size=(None),
    shuffle_after_tarfile_to_samples_buffer_size=(None),
    shuffle_after_batching_buffer_size=None,
)

# Regex pattern
pattern = r"[?!:]\."
# Find all occurrences
all_matches = []
# Process each text
for batch in combined_dataset:
    for turns in batch["texts"]:
        for turn in turns:
            text = turn["assistant"]
            matches = re.findall(pattern, text)
            if matches:
                all_matches.extend(matches)

print(f"len matches: {len(all_matches)}")
