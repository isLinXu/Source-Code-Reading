# %%
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk

from m4.sourcing.data_collection.processors.web_document_extractor import save_split_sharded_already_splitted_dataset


NUM_SHARDS = 68
DS_V1_PATH = Path("/home/lucile/local_datasets/enwiki/enwiki-v1")
DS_V2_COMMON_PATH = Path("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION")
EXCLUDE_SHARD_IDS = [34]
DATASET_NAME_COMPLETE_EXAMPLES_V2 = "wikipedia_html_enterprise-with-images-full-v2-v3"
SHARD_SIZE = 20_000

DS_FINAL_DS_PATH = Path("/home/lucile/local_datasets/enwiki/enwiki-v2")
ds_v1 = load_from_disk(DS_V1_PATH)
# %%
ds_v1
# %%
ds_v1_merged = concatenate_datasets([ds_v1["train"], ds_v1["valid"]])
# %%
ds_v1_merged
# %%

ds_list = []
for shard_id in range(0, NUM_SHARDS):
    if shard_id in EXCLUDE_SHARD_IDS:
        continue
    print(f"Processing shard {shard_id}...")
    shard_dir = DS_V2_COMMON_PATH / f"shard_{shard_id}"
    ds_path = shard_dir / DATASET_NAME_COMPLETE_EXAMPLES_V2
    ds = load_from_disk(ds_path)
    ds_list.append(ds)

ds_v2 = concatenate_datasets(ds_list)
# %%
ds_full = concatenate_datasets([ds_v1_merged, ds_v2])
# %%
ds_full = ds_full.remove_columns(["images_urls", "num_found", "num_not_found", "mismatches"])
# %%
ds_full = ds_full.train_test_split(test_size=0.05, shuffle=False)
ds_full["valid"] = ds_full["test"]
ds_full.pop("test")

save_split_sharded_already_splitted_dataset(
    ds_full, Path("/home/lucile/local_datasets/enwiki") / "enwiki-v2-full", SHARD_SIZE
)
# %%
