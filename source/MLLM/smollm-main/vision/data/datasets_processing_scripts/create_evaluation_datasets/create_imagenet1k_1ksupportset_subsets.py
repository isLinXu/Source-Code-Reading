import numpy as np
from datasets import load_dataset
from tqdm import tqdm


NUM_SUBSETS = 6

NAME_DS = "HuggingFaceM4/imagenet1k_support_1k_query_sets"


ds_subsets = [load_dataset(NAME_DS, use_auth_token=True) for _ in range(NUM_SUBSETS)]

num_test_examples = ds_subsets[0]["test_query_set"].num_rows

selected_indices = np.array_split(range(num_test_examples), NUM_SUBSETS)

for idx_ds in range(NUM_SUBSETS):
    ds_subsets[idx_ds]["test_query_set"] = ds_subsets[idx_ds]["test_query_set"].select(selected_indices[idx_ds])

for idx_ds in tqdm(range(NUM_SUBSETS)):
    ds_subsets[idx_ds].push_to_hub(repo_id=NAME_DS + f"_part_{idx_ds}", private=True)
