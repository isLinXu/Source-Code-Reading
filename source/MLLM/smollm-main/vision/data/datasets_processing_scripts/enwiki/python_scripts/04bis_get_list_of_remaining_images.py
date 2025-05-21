import glob
import json
import math
import os
import tarfile

from pathos.multiprocessing import ProcessingPool as Pool


NUM_SHARDS = 12
PATH_SAVE_DIR_DOWNLOADED_IMAGES = (
    "/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION/downloaded_images-v2"
)
NUM_PROC = 32


def process_one_tar(args):
    (tar_path, idx_tar) = args
    with tarfile.open(tar_path) as tar_file:
        tar_members = tar_file.getmembers()
        name_to_url = {}
        name_to_img = {}
        url_to_img = {}
        for tar_member in tar_members:
            if tar_member.name.endswith(".jpg"):
                name = tar_member.name.replace(".jpg", "")
                tar_member_file = tar_file.extractfile(tar_member)
                img = tar_member_file.read()
                tar_member_file.close()
                name_to_img[name] = img
            elif tar_member.name.endswith(".json"):
                name = tar_member.name.replace(".json", "")
                tar_member_file = tar_file.extractfile(tar_member)
                json_val = json.loads(tar_member_file.read())
                status = json_val["status"]
                url = json_val["url"]
                tar_member_file.close()
                if status == "success":  # Should always happend with webdataset format, not with parquet
                    name_to_url[name] = url
        for name in name_to_url:
            url_to_img[name_to_url[name]] = name_to_img[name]
        new_urls_indexed = list(url_to_img.keys())
        return new_urls_indexed


print("Starting creating the dataset of all images")
tar_paths = glob.glob(os.path.join(PATH_SAVE_DIR_DOWNLOADED_IMAGES, "*.tar"))
tar_paths.sort()
args_pool = [(tar_path, idx_tar) for idx_tar, tar_path in enumerate(tar_paths) if idx_tar != 82]
pool = Pool(NUM_PROC)
urls_indexed = pool.map(process_one_tar, args_pool)
urls_indexed = [sub_el for el in urls_indexed for sub_el in el]

all_urls_missing_path = (
    "/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION/remaining_urls_v2.txt"
)
with open(all_urls_missing_path, "r") as f:
    all_urls_missing = f.read().splitlines()

urls_indexed_set = set(urls_indexed)
remaining_urls = [url for url in all_urls_missing if url not in urls_indexed_set]

# save the remaining urls in shards
shard_size = math.ceil(len(remaining_urls) / NUM_SHARDS)
for idx_shard in range(NUM_SHARDS):
    last_idx = (idx_shard + 1) * shard_size if idx_shard != NUM_SHARDS - 1 else len(remaining_urls)
    shard = remaining_urls[idx_shard * shard_size : last_idx]
    with open(
        f"/home/lucile/local_datasets/enwiki/enwiki-NS0-20230220-ENTERPRISE-HTML-EXTRACTION/remaining_urls_v2_shard_{idx_shard}.txt",
        "w",
    ) as f:
        f.write("\n".join(shard))
