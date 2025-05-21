import json
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


NAME_FILE = sys.argv[1]

PATH_IDEAS = f"/fsx/hugo/ideas_mistral/ideas/{NAME_FILE}.txt"
with open(PATH_IDEAS, "r") as f:
    IDEAS = f.read().split("\n")

BATCH_SIZE = 10
PROMPT = """Create a very SIMPLE and SHORT website with the following elements: {idea}
Be creative with the design, size, position of the elements, columns, etc...
Don't give any explanation, just the content of the HTML code `index.html` starting with `<!DOCTYPE html>`, followed by the CSS code `styles.css` starting with `/* Global Styles */`.
Write real and short sentences for the paragraphs, don't use Lorem ipsum. When you want to display an image, don't use <img> in the HTML, always display a colored rectangle instead. """
PAD_TOKEN_ID = 32014
ALL_GEN = []

PATH_SAVE_GEN = f"/fsx/hugo/deepseek_html_css/gen/{NAME_FILE}.json"


model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-coder-33b-instruct",
    # device_map="auto",
    use_flash_attention_2=True,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
).cuda()
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-33b-instruct")


for idx in tqdm(range(0, len(IDEAS), BATCH_SIZE)):
    current_ideas = IDEAS[idx : idx + BATCH_SIZE]
    all_messages = [
        [
            {"role": "user", "content": PROMPT.format(idea=idea)},
        ]
        for idea in current_ideas
    ]
    all_encodeds = [tokenizer.apply_chat_template(messages, return_tensors="pt") for messages in all_messages]
    len_prompts = [encodeds.shape[1] for encodeds in all_encodeds]
    max_len_prompt = max(len_prompts)
    padded_all_encodeds = torch.full((len(current_ideas), max_len_prompt), PAD_TOKEN_ID)
    for idx_enc, encodeds in enumerate(all_encodeds):
        padded_all_encodeds[idx_enc, max_len_prompt - len_prompts[idx_enc] :] = encodeds[0]
    model_inputs = padded_all_encodeds.to(model.device)

    generated_ids = model.generate(model_inputs, max_new_tokens=2048, do_sample=True, eos_token_id=32021)
    generated_ids = generated_ids[:, max_len_prompt:]
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    decoded = [el.replace("<|EOT|>", "") for el in decoded]
    ALL_GEN.extend(decoded)


with open(PATH_SAVE_GEN, "w") as f:
    json.dump(ALL_GEN, f)
print("Generations successfully saved")
