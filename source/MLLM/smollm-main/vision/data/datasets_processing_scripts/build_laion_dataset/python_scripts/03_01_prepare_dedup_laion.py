# The followings steps were done in different jobs, this is to give an idea of what was done


import pickle

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


laion_dataset = load_dataset("laion/laion2b-en-vit-h-14-embeddings")["train"]  # Takes a long time to download
# The md5 is shorter than the url to identify an image. Moreover, some images in the dataset are the same but under
# different urls. In this case they have the same md5, and we'll be able to have even more compact data
# laion_dataset_md5 is uploaded at s3://m4-datasets/trash/laion_dataset_md5/
laion_dataset_md5 = laion_dataset.remove_columns([c_n for c_n in laion_dataset.column_names if c_n != "md5"])

# Download at https://huggingface.co/datasets/fraisdufour/snip-dedup/resolve/main/is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy
is_dup_all = np.load("/fsx/hugo/prepare_dedup_laion/is_dup_mlp_1024_128_gelu_snn_2layer_notext.npy").ravel()

list_index_dup = [idx for idx, el in enumerate(is_dup_all) if el] + [
    idx for idx in range(len(is_dup_all), len(laion_dataset_md5))
]
set_dup = set()
for idx in tqdm(list_index_dup):
    set_dup.add(laion_dataset_md5[idx]["md5"])

# set_dup_md5.pkl is uploaded at s3://m4-datasets/trash/set_dup_md5.pkl
with open("/fsx/hugo/prepare_dedup_laion/set_dup_md5.pkl", "wb") as f:
    pickle.dump(set_dup, f)
