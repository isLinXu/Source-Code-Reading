import os

from datasets import load_from_disk


SUBSET_DIR_PATH = "/scratch/m4/webdocs/web_document_dataset_filtered/"
BIG_SHARD_ID = 0
cm4_valid_path = f"{SUBSET_DIR_PATH}/{BIG_SHARD_ID}"
sync_cmd = (
    "s5cmd sync"
    f" s3://m4-datasets/webdocs/web_document_dataset_filtered_imgurldedup_nsfwfiltered_urldedup_linededup_finalcleaning_setimgurlsdedup/{BIG_SHARD_ID}/*"
    f" {cm4_valid_path}"
)

os.system(sync_cmd)

ds = load_from_disk(cm4_valid_path)

ds_sample = ds.select(range(10000))
repo_id = "HuggingFaceM4/cm4_valid-Sample"
ds_sample.push_to_hub(repo_id, "valid", private=True)

ds.push_to_hub("HuggingFaceM4/cm4_valid", "valid", private=True)
