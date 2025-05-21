from pathlib import Path

from datasets import concatenate_datasets, load_from_disk


NUM_SHARDS = 68
DATA_DIR = Path("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION")
DATASET_NAME_COMPLETE_EXAMPLES = "wikipedia_html_enterprise-with-images-full-v1-v2"
SAVING_DIR = Path("/home/lucile/local_datasets/enwiki/enwiki-v1")
NUM_PROC = 32

EXCLUDE_SHARD_IDS = [34]  # This shard is corrupted since the beginning

ds_list = []
for shard_id in range(NUM_SHARDS):
    if shard_id in EXCLUDE_SHARD_IDS:
        continue
    ds_path = DATA_DIR / f"shard_{shard_id}" / DATASET_NAME_COMPLETE_EXAMPLES
    ds = load_from_disk(ds_path)

    ds = ds.remove_columns(["metadata", "images_urls", "num_found", "num_not_found", "mismatches"])
    ds_list.append(ds)

ds = concatenate_datasets(ds_list)
ds = ds.train_test_split(test_size=0.05, shuffle=False)
ds["valid"] = ds["test"]
ds.pop("test")
ds.save_to_disk(SAVING_DIR)

# data_dir = "/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION/shard_0/wikipedia_html_enterprise_with_images_v0"
# ds = load_from_disk(data_dir)
# sub_ds  = ds.remove_columns(['metadata', 'images_urls', 'num_found', 'num_not_found', 'mismatches'])
# sub_ds
# sub_ds.save_to_disk("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION/shard_0/wikipedia_html_enterprise_with_images_light_v0")
# sub_ds = sub_ds.train_test_split(test_size=0.05, shuffle=True)
# sub_ds["valid"] = sub_ds["test"]
# sub_ds.pop("test")
# sub_ds

# sub_ds.save_to_disk("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION/shard_0/wikipedia_html_enterprise_with_images_light_v0")
# ds = load_from_disk("/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION/shard_0/wikipedia_html_enterprise_with_images_light_v0")
# ds["train"]
