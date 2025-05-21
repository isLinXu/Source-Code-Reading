import os

from datasets import load_from_disk


SUBSET_DIR_PATH = "/scratch/general_pmd/image/coco/"
valid_path = f"{SUBSET_DIR_PATH}/validation"
sync_cmd = f"s5cmd sync s3://m4-datasets/general_pmd/image/coco/validation/00000-00001/* {valid_path}"

os.system(sync_cmd)

ds = load_from_disk(valid_path)
print(ds)
repo_id = "HuggingFaceM4/coco_valid"
ds.push_to_hub(repo_id, "valid", private=True)
