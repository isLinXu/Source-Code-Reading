"""
srun --pty --cpus-per-task=48 --mem-per-cpu=11G bash -i
conda activate /fsx/m4/conda/shared-m4-2023-03-10
"""


import json
import os

from datasets import Dataset
from tqdm import tqdm


NUM_SHARDS = 200

PATH_LAION_URLS_S3 = "s3://m4-datasets/LAION_data/urls_laion_dataset_filtered_dedup/"
PATH_LAION_URLS_LOCAL = "/scratch/laion_urls"

PATH_SAVE_DISK_DS_LAION_URLS = "/scratch/ds_laion_urls"
PATH_SAVE_S3_DS_LAION_URLS = "s3://m4-datasets/LAION_data/ds_urls_laion_dataset_filtered_dedup/"

NUM_PROC = 48


if __name__ == "__main__":
    command_sync_s3 = f"aws s3 sync {PATH_LAION_URLS_S3} {PATH_LAION_URLS_LOCAL}"
    os.system(command_sync_s3)

    all_urls = []
    for idx_shard in tqdm(range(NUM_SHARDS)):
        if idx_shard not in [184, 189]:
            path_urls_laion_shard = os.path.join(PATH_LAION_URLS_LOCAL, str(idx_shard), "laion_urls.json")
            with open(path_urls_laion_shard) as f:
                all_urls.extend(json.load(f))

    ds_laion_urls = Dataset.from_dict({"url": all_urls})
    ds_laion_urls.save_to_disk(PATH_SAVE_DISK_DS_LAION_URLS, num_proc=NUM_PROC)

    command_sync_s3 = f"aws s3 sync {PATH_SAVE_DISK_DS_LAION_URLS} {PATH_SAVE_S3_DS_LAION_URLS}"
    os.system(command_sync_s3)
