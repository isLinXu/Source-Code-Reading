import json
import math
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file


# Source and destination file paths
source_dir = (
    "/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/siglip-so400m-14-384-flash-attn2"
)
out_dir = (
    "/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/siglip-so400m-14-700-flash-attn2"
)
config_input_file_path = f"{source_dir}/config.json"
config_out_file_path = f"{out_dir}/config.json"


os.makedirs(out_dir, exist_ok=True)


state_dict = load_file(f"{source_dir}/model.safetensors")
new_size = 700

with open(config_input_file_path, "r") as f:
    model_config = json.loads(f.read())
    vision_model_config = model_config["vision_config"]

k = "vision_model.embeddings.position_embedding.weight"
v = state_dict[k]
print(f"Shape before interpolation: {v.shape}")
height = new_size
width = new_size
patch_pos_embed = state_dict[k].unsqueeze(0)
num_positions = patch_pos_embed.shape[1]

embed_dim = patch_pos_embed.shape[-1]
num_h_patches = height // vision_model_config["patch_size"]
num_w_patches = width // vision_model_config["patch_size"]
# we add a small number to avoid floating point error in the interpolation
# see discussion at https://github.com/facebookresearch/dino/issues/8
num_h_patches, num_w_patches = num_h_patches + 0.1, num_w_patches + 0.1
sqrt_num_positions = math.sqrt(num_positions)
patch_pos_embed = patch_pos_embed.reshape(1, int(sqrt_num_positions), int(sqrt_num_positions), embed_dim)
patch_pos_embed_dtype = patch_pos_embed.dtype
patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2).to(torch.float)
patch_pos_embed = nn.functional.interpolate(
    patch_pos_embed,
    scale_factor=(num_h_patches / sqrt_num_positions, num_w_patches / sqrt_num_positions),
    mode="bicubic",
    align_corners=False,
).to(patch_pos_embed_dtype)
if int(num_h_patches) != patch_pos_embed.shape[-2] or int(num_w_patches) != patch_pos_embed.shape[-1]:
    raise ValueError(
        f"Number of patches for images ({int(num_h_patches), int(num_w_patches)}) don't match the "
        f"shape of position embedding ({patch_pos_embed.shape[-2], patch_pos_embed.shape[-1]})"
    )
patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, embed_dim)
patch_pos_embed = patch_pos_embed.squeeze(0)
state_dict[k] = patch_pos_embed

# Sanity check
print(k)
print(f"Shape after interpolation: {state_dict[k].shape}")

save_file(state_dict, f"{out_dir}/model.safetensors", metadata={"format": "pt"})
# Update config accordingly
with open(config_input_file_path, "r") as f:
    model_config = json.loads(f.read())
    model_config["vision_config"]["image_size"] = new_size

with open(config_out_file_path, "w") as json_file:
    json.dump(model_config, json_file)
