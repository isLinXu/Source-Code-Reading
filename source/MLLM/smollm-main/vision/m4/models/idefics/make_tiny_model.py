#!/usr/bin/env python

# This script creates a super tiny model that is useful inside tests, when we just want to test that
# the machinery works, without needing to check the quality of the outcomes.
#
# usage: adjust the configs if wanted, but otherwise just run the script

from pathlib import Path
from types import SimpleNamespace

import torchvision.transforms as transforms
from PIL import Image

from m4.models.idefics.modeling_idefics import IdeficsConfig, IdeficsForCausalLM
from m4.training.utils import get_tokenizer


mname_tiny = "tiny-random-idefics-m4"

path = Path(mname_tiny)
path.mkdir(parents=True, exist_ok=True)

# from the hardcoded https://github.com/huggingface/m4/blob/adf102f0000cb2632cd8a3ebb87398c65e448a97/m4/training/main.py#L80
additional_vocab_size = 2

config = IdeficsConfig()
config.update(
    dict(
        ffn_dim=64,
        hidden_size=16,
        max_position_embeddings=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        word_embed_proj_dim=16,
        max_new_tokens=100,
        use_resampler=True,
        resampler_depth=2,
        resampler_head_dim=8,
        resampler_n_heads=2,
        resampler_n_latents=16,
        vision_embed_dim=32,
        vision_image_size=30,
        vision_model_name="hf-internal-testing/tiny-random-clip",
        vision_model_params="{}",
        vocab_size=32000,
        additional_vocab_size=additional_vocab_size,
    )
)

# print(config)
# can now modify config to say tiny values

model = IdeficsForCausalLM.from_config(config)
# print(model.config)
# print(model)

tokenizer_config = dict(
    tokenizer_add_special_tokens="{}",
    tokenizer_add_tokens=(
        '[AddedToken("<fake_token_around_image>", rstrip=False, lstrip=False, normalized=False), AddedToken("<image>",'
        " rstrip=False, lstrip=False, normalized=False)]"
    ),
    tokenizer_name="HuggingFaceM4/huggy-llama-tokenizer-7b",
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

# Test w/ one image and one text
query = "<fake_token_around_image><image><fake_token_around_image>This is a picture of a cat."
query_tokens = tokenizer(query, return_tensors="pt")

num_images_per_ex = 1
pixel_values = transforms.ToTensor()(Image.new("RGB", (30, 30))).repeat(1, 1, 1, 1).unsqueeze(0)

input = {
    "input_ids": query_tokens["input_ids"],
    "attention_mask": query_tokens["attention_mask"],
    "pixel_values": pixel_values,
    "pixel_values": pixel_values,
}
# debug shapes
# print(query_tokens["input_ids"].shape)
# print(query_tokens["attention_mask"].shape)
# print(pixel_values.shape)
# print(image_attention_mask.shape)

out_gen = model.generate(**input)
text = tokenizer.batch_decode(out_gen)
# print(text)

# Save model + config + tokenizer
model.half()  # makes it smaller
model.save_pretrained(path)
tokenizer.save_pretrained(path)

# test we can load it back
model = IdeficsForCausalLM.from_pretrained(path)

print(f"Generated {mname_tiny} - Upload the generated folder to the hub")
