import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


BATCH_SIZE = 10
NUM_ITER = 100
PROMPT = (
    "Generate diverse website layout ideas for different companies, each with a unique design element. Examples"
    " include: a car company site with a left column, a webpage footer with a centered logo. Explore variations in"
    " colors, positions, and company fields. Don't give any explanations or recognition that you have understood the"
    " request, just give the list of 10 ideas, with a line break between each."
)
ALL_IDEAS = []
NAME_FILE = sys.argv[1]  # 300 jobs for NUM_JOBS * NUM_ITER * BATCH_SIZE * NUM_IDEAS_PER_EX = 3M ideas


device = "cuda"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer.padding_side = "left"

model.to(device)


messages = [
    {"role": "user", "content": PROMPT},
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
encodeds = encodeds.repeat(BATCH_SIZE, 1)
len_prompt = encodeds.shape[1]
model_inputs = encodeds.to(device)


for _ in tqdm(range(NUM_ITER)):
    generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
    generated_ids = generated_ids[:, len_prompt:]
    decoded = tokenizer.batch_decode(generated_ids)

    for dec_ in decoded:
        try:
            dec = dec_
            dec = dec.replace("</s>", "").replace("\n\n", "\n")
            dec = dec.split("\n")
            dec = [el for el in dec if el[0].isdigit()]
            if len(dec) != 10:
                continue
            dec = [
                el[2:].strip() if el[:2] == f"{idx + 1}." else el[3:].strip() if el[:3] == f"{idx + 1}." else ""
                for idx, el in enumerate(dec)
            ]
            dec = [el for el in dec if el]
            if len(dec) != 10:
                continue
            ALL_IDEAS.extend(dec)
        except Exception:
            pass


print(f"Number of ideas: {len(ALL_IDEAS)}")

with open(f"/fsx/hugo/ideas_mistral_websight_v02/ideas/{NAME_FILE}.txt", "w") as f:
    f.write("\n".join(ALL_IDEAS))
