import glob
import json

from safetensors import safe_open
from safetensors.torch import save_file


model_path = "/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/tr_272_bis_opt_step_15000_merge_and_resize_eou_renamed_lmhead/unwrapped_model"
safetensor_files = glob.glob(f"{model_path}/model*.safetensors")

KEYS_TO_MODIFY_MAPPING = {
    "lm_head.additional_fc": "additional_fc",
}

for file in safetensor_files:
    tensors = {}
    with safe_open(file, framework="pt", device="cpu") as f:
        for old_key in f.keys():
            final_key = old_key
            for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in old_key:
                    final_key = old_key.replace(key_to_modify, new_key)
            tensors[final_key] = f.get_tensor(old_key)
    print(f"{tensors.keys()}")
    save_file(tensors, file, metadata={"format": "pt"})

with open(f"{model_path}/model.safetensors.index.json", "r") as f:
    data = json.load(f)
    keys_to_iterate = list(data["weight_map"].keys())
    new_data_weight_map = {}
    for old_key, v in data["weight_map"].items():
        final_key = old_key
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in old_key:
                final_key = old_key.replace(key_to_modify, new_key)
        new_data_weight_map[final_key] = v
    data["weight_map"] = new_data_weight_map

with open(f"{model_path}/model.safetensors.index.json", "w") as f:
    json_object = json.dumps(data, indent=4)
    f.write(json_object)
