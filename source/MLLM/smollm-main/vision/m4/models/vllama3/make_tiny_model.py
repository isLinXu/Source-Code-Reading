#!/usr/bin/env python

# This script creates a super tiny model that is useful inside tests, when we just want to test that
# the machinery works, without needing to check the quality of the outcomes.
#
# usage: adjust the configs if wanted, but otherwise just run the script

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from m4.models.vllama3.configuration_vllama3 import VLlama3Config, VLlama3PerceiverConfig, VLlama3VisionConfig
from m4.models.vllama3.modeling_vllama3 import VLlama3ForCausalLM
from m4.training.utils import get_tokenizer


mname_tiny = "tiny-random-vllama3-m4"

path = Path(mname_tiny)
path.mkdir(parents=True, exist_ok=True)

# from the hardcoded https://github.com/huggingface/m4/blob/adf102f0000cb2632cd8a3ebb87398c65e448a97/m4/training/main.py#L80
additional_vocab_size = 2

config = VLlama3Config()
config.update(
    dict(
        hidden_size=16,
        intermediate_size=16 * 4,
        max_position_embeddings=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_hidden_layers=2,
        max_new_tokens=100,
        vision_config=VLlama3VisionConfig(vision_model_name="HuggingFaceM4/tiny-random-siglip"),
        use_resampler=True,
        perceiver_config=VLlama3PerceiverConfig(
            resampler_depth=2,
            resampler_head_dim=8,
            resampler_n_heads=2,
            resampler_n_latents=16,
        ),
        vocab_size=128_256,
        additional_vocab_size=additional_vocab_size,
        _flash_attn_2_enabled=True,
    )
)
# print(config)
# can now modify config to say tiny values

model = VLlama3ForCausalLM.from_config(config)
# print(model.config)
# print(model)

tokenizer_config = dict(
    tokenizer_add_special_tokens="{}",
    tokenizer_add_tokens=(
        '[AddedToken("<fake_token_around_image>", rstrip=False, lstrip=False, normalized=False), AddedToken("<image>",'
        " rstrip=False, lstrip=False, normalized=False)]"
    ),
    tokenizer_name="HuggingFaceM4/Meta-Llama-3-8B-tokenizer-pad-is-reserved_special_token_0",
    tokenizer_params='{"use_fast": True}',
)
tokenizer_config = SimpleNamespace(**tokenizer_config)
# print(tokenizer_config)

tokenizer = get_tokenizer(
    tokenizer_name=tokenizer_config.tokenizer_name,
    tokenizer_add_tokens=tokenizer_config.tokenizer_add_tokens,
    tokenizer_add_special_tokens=tokenizer_config.tokenizer_add_special_tokens,
    tokenizer_params=tokenizer_config.tokenizer_params,
    additional_vocab_size=model.config.additional_vocab_size,
    model_vocab_size=model.config.vocab_size,
)
assert "<image>" in tokenizer.get_vocab()

single_image_seq_len = (
    model.config.perceiver_config.resampler_n_latents
    if model.config.use_resampler
    else (model.config.vision_config.image_size // model.config.vision_config.patch_size)
    ** 2  # + 1 # That +1 is only valid for vit/clip
)
image_tokens = "<image>" * single_image_seq_len
# Test w/ one image and one text
query = f"<fake_token_around_image>{image_tokens}<fake_token_around_image>This is a picture of a cat."
query_tokens = tokenizer(query, return_tensors="pt")

num_images_per_ex = 1
img = Image.fromarray(np.uint8(np.random.rand(30, 30, 3) * 255))
pixel_values = transforms.ToTensor()(img).repeat(1, 1, 1, 1).unsqueeze(0).to(dtype=torch.bfloat16).to("cuda:0")

model.to(torch.bfloat16)
model.to("cuda:0")
input = {
    "input_ids": query_tokens["input_ids"].to("cuda:0"),
    "attention_mask": query_tokens["attention_mask"].to("cuda:0"),
    "pixel_values": pixel_values,
}
# debug shapes
# print(query_tokens["input_ids"].shape)
# print(query_tokens["attention_mask"].shape)
# print(pixel_values.shape)
out_gen = model.generate(**input)
text = tokenizer.batch_decode(out_gen)
# print(text)

# Save model + config + tokenizer
# model.half()  # makes it smaller
model.save_pretrained(path)
tokenizer.save_pretrained(path)

# test we can load it back
model = VLlama3ForCausalLM.from_pretrained(path)

print(f"Generated {mname_tiny} - Upload the generated folder to the hub")
# Pushed to HuggingFaceM4/tiny-random-vllama3-m4
