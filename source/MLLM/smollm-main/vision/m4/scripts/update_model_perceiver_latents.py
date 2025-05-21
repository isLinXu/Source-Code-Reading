import json
import os
import shutil

import torch


# Script meant to double the number of perceiver latents of a given idefics model, and save the new model with an updated config
# Source and destination file paths
source_dir = "/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/tr_199_position_embedding_392_tokenizer_fixed/opt_step-65000/"
destination_dir = "/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/tr_199_pos_embed_392_updated_latents_tokenizer_fixed/opt_step-65000"

checkpoint_dir = f"{source_dir}/unwrapped_model"
tokenizer_input_dir = f"{source_dir}/tokenizer"
input_bin_index_file_path = f"{source_dir}/unwrapped_model/pytorch_model.bin.index.json"
config_input_file_path = f"{source_dir}/unwrapped_model/config.json"
generation_config_input_file_path = f"{source_dir}/unwrapped_model/generation_config.json"


checkpoint_out_dir = f"{destination_dir}/unwrapped_model"
tokenizer_out_dir = f"{destination_dir}/tokenizer"
out_bin_index_file_path = f"{destination_dir}/unwrapped_model/pytorch_model.bin.index.json"
config_out_file_path = f"{destination_dir}/unwrapped_model/config.json"
generation_config_out_file_path = f"{destination_dir}/unwrapped_model/generation_config.json"

os.makedirs(checkpoint_out_dir, exist_ok=True)

num_shards = 2
shard_paths = [f"{checkpoint_dir}/pytorch_model-{i+1:05}-of-{num_shards:05}.bin" for i in range(num_shards)]
state_dicts = [torch.load(path, map_location=torch.device("cpu")) for path in shard_paths]

embed_tokens = None
additional_embed = None
additional_lm_head = None
for state_dict in state_dicts:
    for k, v in state_dict.items():
        if "perceiver_resampler.latents" in k:
            state_dict[k] = torch.cat([v, v], dim=0)

# Sanity check
for state_dict in state_dicts:
    for k, v in state_dict.items():
        if "perceiver_resampler.latents" in k:
            print(k)
            print(v.shape)

for i, shard in enumerate(state_dicts):
    torch.save(shard, f"{checkpoint_out_dir}/pytorch_model-{i+1:05}-of-{num_shards:05}.bin")

# Update config accordingly
with open(config_input_file_path, "r") as f:
    model_config = json.loads(f.read())
    model_config["resampler_n_latents"] = 2 * model_config["resampler_n_latents"]
with open(config_out_file_path, "w") as json_file:
    json.dump(model_config, json_file)


# Copy the bin index, generation config and tokenizer
shutil.copy(input_bin_index_file_path, out_bin_index_file_path)
shutil.copy(generation_config_input_file_path, generation_config_out_file_path)
shutil.copytree(tokenizer_input_dir, tokenizer_out_dir)
