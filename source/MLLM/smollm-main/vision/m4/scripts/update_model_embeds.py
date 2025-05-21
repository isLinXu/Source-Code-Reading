import os
import shutil

import torch


source_dir = "/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/tr_refactored_idefics_214_image_size_392/opt_step-9500"
out_dir = "/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/tr_refactored_idefics_214_image_size_392_updated_embeds/opt_step-9500"

checkpoint_dir = f"{source_dir}/unwrapped_model"
tokenizer_input_dir = f"{source_dir}/tokenizer"
input_bin_index_file_path = f"{source_dir}/unwrapped_model/pytorch_model.bin.index.json"
config_input_file_path = f"{source_dir}/unwrapped_model/config.json"
generation_config_input_file_path = f"{source_dir}/unwrapped_model/generation_config.json"


checkpoint_out_dir = f"{out_dir}/unwrapped_model"
tokenizer_out_dir = f"{out_dir}/tokenizer"
out_bin_index_file_path = f"{out_dir}/unwrapped_model/pytorch_model.bin.index.json"
config_out_file_path = f"{out_dir}/unwrapped_model/config.json"
generation_config_out_file_path = f"{out_dir}/unwrapped_model/generation_config.json"

os.makedirs(checkpoint_out_dir, exist_ok=True)
num_shards = 2
shard_paths = [f"{checkpoint_dir}/pytorch_model-{i+1:05}-of-{num_shards:05}.bin" for i in range(num_shards)]
state_dicts = [torch.load(path, map_location=torch.device("cpu")) for path in shard_paths]

embed_tokens = None
additional_embed = None
additional_lm_head = None
for state_dict in state_dicts:
    for k, v in state_dict.items():
        if "embed_tokens.weight" in k:
            embed_tokens = v
        if "embed_tokens.additional_embedding.weight" in k:
            additional_embed = v
        if "lm_head.additional_fc.weight" in k:
            additional_lm_head = v

init_tensor = embed_tokens.mean(dim=0, keepdim=True)
additional_embed = torch.cat([additional_embed, init_tensor], dim=0)
additional_lm_head = torch.cat([additional_lm_head, init_tensor], dim=0)

for state_dict in state_dicts:
    for k, v in state_dict.items():
        if "embed_tokens.additional_embedding.weight" in k:
            state_dict[k] = additional_embed
        if "lm_head.additional_fc.weight" in k:
            state_dict[k] = additional_lm_head

# Sanity check
for state_dict in state_dicts:
    for k, v in state_dict.items():
        if "embed_tokens.additional_embedding.weight" in k:
            print(k)
            print(v.shape)
        if "lm_head.additional_fc.weight" in k:
            print(k)
            print(v.shape)

for i, shard in enumerate(state_dicts):
    torch.save(shard, f"{checkpoint_out_dir}/pytorch_model-{i+1:05}-of-{num_shards:05}.bin")

# Copy the bin index, config, generation config and tokenizer
shutil.copy(input_bin_index_file_path, out_bin_index_file_path)
shutil.copy(config_input_file_path, config_out_file_path)
shutil.copy(generation_config_input_file_path, generation_config_out_file_path)
shutil.copytree(tokenizer_input_dir, tokenizer_out_dir)
