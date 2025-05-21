#!/usr/bin/env python

# This script creates a super tiny model that is useful inside tests, when we just want to test that
# the machinery works, without needing to check the quality of the outcomes.
#
# usage: adjust the configs if wanted, but otherwise just run the script

from pathlib import Path

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


mname_tiny = "tiny-random-Llama3ForCausalLM"

path = Path(mname_tiny)
path.mkdir(parents=True, exist_ok=True)

config = LlamaConfig()
config.update(
    dict(
        vocab_size=128_256,
        hidden_size=16,
        intermediate_size=16 * 4,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
    )
)
model = LlamaForCausalLM(config)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Test w/ one text
query = "This is a test"
query_tokens = tokenizer(query, return_tensors="pt")

input = {
    "input_ids": query_tokens["input_ids"],
    "attention_mask": query_tokens["attention_mask"],
}

out_gen = model.generate(**input)
text = tokenizer.batch_decode(out_gen)

# Save model + config + tokenizer
model.half()  # makes it smaller
model.save_pretrained(path)
tokenizer.save_pretrained(path)

# test we can load it back
model = LlamaForCausalLM.from_pretrained(path)

print(f"Generated {mname_tiny} - Upload the generated folder to the hub")
# Pushed to HuggingFaceM4/tiny-random-Llama3ForCausalLM
