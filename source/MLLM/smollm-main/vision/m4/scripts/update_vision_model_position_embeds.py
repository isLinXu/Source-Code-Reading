import math
import os
import shutil

import torch
import torch.nn as nn

from m4.models.vmistral.modeling_vmistral import VMistralForCausalLM


# Source and destination file paths
source_dir = (
    "/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/tr_266_265_stage_1/opt_step-80000"
)
out_dir = "/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/tr_266_265_stage_1/opt_step-80000_siglip_res_980"

checkpoint_dir = f"{source_dir}/unwrapped_model"
tokenizer_input_dir = f"{source_dir}/tokenizer"


checkpoint_out_dir = f"{out_dir}/unwrapped_model"
tokenizer_out_dir = f"{out_dir}/tokenizer"

os.makedirs(checkpoint_out_dir, exist_ok=True)
model = VMistralForCausalLM.from_pretrained(
    checkpoint_dir,
    is_resume=False,
    new_model=False,
)

# args to change
new_size = 980
new_vision_model_name = "HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit"

new_state_dict = model.state_dict()
vision_model_config = model.config.vision_config

embedding_key = "model.vision_model.embeddings.position_embedding.weight"
print(f"Shape before interpolation: {model.state_dict()[embedding_key].shape}")
new_state_dict["model.vision_model.embeddings.position_embedding.weight"].shape
height = new_size
width = new_size
patch_pos_embed = new_state_dict[embedding_key].unsqueeze(0)
num_positions = patch_pos_embed.shape[1]

embed_dim = patch_pos_embed.shape[-1]
num_h_patches = height // vision_model_config.patch_size
num_w_patches = width // vision_model_config.patch_size
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
new_state_dict[embedding_key] = patch_pos_embed

# Update config accordingly, and load new state dict in the new model
new_config = model.config
new_config.vision_config.image_size = new_size
new_config.vision_config.vision_model_name = new_vision_model_name
new_model = VMistralForCausalLM.from_config(new_config)
new_model.load_state_dict(new_state_dict)
print(f"Shape after interpolation:: {new_model.state_dict()[embedding_key].shape}")


new_model.save_pretrained(checkpoint_out_dir)
# copy paste tokenizer
shutil.copytree(tokenizer_input_dir, tokenizer_out_dir)
