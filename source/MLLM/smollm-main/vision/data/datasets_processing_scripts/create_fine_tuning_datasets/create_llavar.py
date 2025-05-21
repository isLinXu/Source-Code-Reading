# DL finetuning images: https://drive.google.com/file/d/1Ms7OCjcFQ18Whmujszpc9bTp0Jy0Dye4/view?usp=sharing
# DL finetuning instructions: https://drive.google.com/file/d/1ISdKOV1wwVkLHf5FNutctpOBa-CmNRFv/view?usp=sharing


import json
import os

from datasets import Dataset
from PIL import Image


PATH_CONV = "/Users/hugolaurencon/Desktop/llava_instruct_150k_llavar_16k.json"
PATH_DIR_IMAGES = "/Users/hugolaurencon/Desktop/finetune"


with open(PATH_CONV) as f:
    data_conv = json.load(f)
data_conv = data_conv[-15500:]  # Before it's only the regular LLaVA instructions


all_image = []
all_user_texts = []
all_bot_texts = []

for conv in data_conv:
    image_path = os.path.join(PATH_DIR_IMAGES, conv["image"])
    image = Image.open(image_path)
    all_image.append(image)
    user_texts = []
    bot_texts = []
    for turn in conv["conversations"]:
        if turn["from"] == "human":
            user_texts.append(turn["value"].replace("<image>", "").strip())
        elif turn["from"] == "gpt":
            bot_texts.append(turn["value"])
    assert len(user_texts) == len(bot_texts)
    all_user_texts.append(user_texts)
    all_bot_texts.append(bot_texts)

assert len(all_image) == len(all_user_texts) == len(all_bot_texts)


ds = Dataset.from_dict({"image": all_image, "user_texts": all_user_texts, "bot_texts": all_bot_texts})
ds.push_to_hub("HuggingFaceM4/LLaVAR-Instruct-16K")
