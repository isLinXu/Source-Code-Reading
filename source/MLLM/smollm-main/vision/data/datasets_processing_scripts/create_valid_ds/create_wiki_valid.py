import os

from datasets import concatenate_datasets, load_from_disk


SUBSET_DIR_PATH = "/scratch/enwiki/"
valid_path = f"{SUBSET_DIR_PATH}/validation"
sync_cmd = f"s5cmd sync s3://m4-datasets/enwiki/enwiki-v2/valid/* {valid_path}"

os.system(sync_cmd)


shard_valid_path = f"{SUBSET_DIR_PATH}/validation/shard_0"
ds = load_from_disk(shard_valid_path)
print(ds)
repo_id = "HuggingFaceM4/enwiki-v2_valid-Sample"
ds.push_to_hub(repo_id, "valid", private=True)


valid_path = [f"{SUBSET_DIR_PATH}/validation/shard_{shard_id}" for shard_id in range(10)]
ds = [load_from_disk(path) for path in valid_path]
ds = concatenate_datasets(ds)

print(ds)
repo_id = "HuggingFaceM4/enwiki-v2_valid"
ds.push_to_hub(repo_id, "valid", private=True)
