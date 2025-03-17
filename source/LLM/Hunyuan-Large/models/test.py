from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer

# Step 1: Initialize ByteLevelBPETokenizer
#tokenizer = ByteLevelBPETokenizer(
#    "vocab.json",
#    "merges.txt"
#)

# Step 2: Save the tokenizer configuration
#tokenizer.save_model("auto_model")

# Step 3: Load the tokenizer using AutoTokenizer
auto_tokenizer = AutoTokenizer.from_pretrained("./", use_fast=False, trust_remote_code=True)

# Test the tokenizer
text = "Hello, world!"
encoded = auto_tokenizer.encode(text)
decoded = auto_tokenizer.decode(encoded)

print("Encoded:", encoded)
print("Decoded:", decoded)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm good, thank you! How can I help you today?"},
    {"role": "user", "content": "Nothing"},
]

print('messages:', messages)
ids = auto_tokenizer.apply_chat_template(messages)
print(f"input_ids:\t{ids}")
text = auto_tokenizer.decode(ids)
print(f"input_text:\t[{text}]")
