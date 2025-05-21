import logging
import os
from pathlib import Path

import torch
from PIL import Image

from m4.models.vgpt2.modeling_vgpt2 import VGPT2LMHeadModel
from m4.training.utils import build_image_transform, get_tokenizer


logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_path):
    opt_step_dir = Path(model_path)
    model_file = opt_step_dir / "unwrapped_model/pytorch_model.bin"
    model_config_file = opt_step_dir / "unwrapped_model/config.json"
    model = VGPT2LMHeadModel.from_pretrained(model_file, config=model_config_file)
    return model


def generate_image_conditioned_text_from_pretrained(
    model_paths,
    image_paths,
    prompt,
    tokenizer_infos,
    max_len=50,
    num_beams=None,
    no_repeat_ngram_size=None,
    image_size=224,
    num_images_per_ex=1,
):
    logger.info(f"DEVICE: {DEVICE}")
    # Model names are gathered based on the paths fed. Therefore your model path needs to have
    # the following structure for this to make sense: .../model_name/opt_step-xxxx
    model_names = [
        f"{str(model_path).split(os.path.sep)[-2]}_{str(model_path).split(os.path.sep)[-1]}"
        for model_path in model_paths
    ]
    wandb_name = "gen_"
    wandb_name += f"beam_{num_beams}" if num_beams else "greedy"
    wandb_name += f"_ngram_{no_repeat_ngram_size}" if no_repeat_ngram_size else ""
    for model_name in model_names:
        wandb_name += f"___{model_name}"
    wandb_config = {
        "prompt": prompt,
        "greedy": False if num_beams else True,
        "beams": num_beams,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "max_len": max_len,
        "models compared": model_names,
    }

    image_transform = build_image_transform(image_size, eval=True)

    list_images = []
    for i, image_path in enumerate(image_paths):
        list_images.append(image_transform(Image.open(image_path)).repeat(num_images_per_ex, 1, 1, 1).unsqueeze(0))
    # TODO: allow for different prompts depending on image
    queries = [[prompt] for _ in range(len(image_paths))]

    all_texts = {}
    for model_path, model_name in zip(model_paths, model_names):
        model = get_model(model_path).to(DEVICE)
        model.eval()

        model_texts = []

        tokenizer = get_tokenizer(
            tokenizer_name=tokenizer_infos["tokenizer_name"],
            tokenizer_add_tokens=tokenizer_infos["tokenizer_add_tokens"],
            tokenizer_add_special_tokens=tokenizer_infos["tokenizer_add_special_tokens"],
            tokenizer_params=tokenizer_infos["tokenizer_params"],
            additional_vocab_size=model.config.additional_vocab_size,
            model_vocab_size=model.config.vocab_size,
        )
        bad_words_ids = [[tokenizer.eos_token_id]]

        for query, pixel_values in zip(queries, list_images):
            query_tokens = tokenizer(query, return_tensors="pt")
            input = {
                "input_ids": query_tokens["input_ids"].to(DEVICE),
                "attention_mask": query_tokens["attention_mask"].to(DEVICE),
                "pixel_values": pixel_values.to(DEVICE),
            }
            out_gen = model.generate(
                **input,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_length=max_len,
                eos_token_id=tokenizer.additional_special_tokens_ids[-1],
                bad_words_ids=bad_words_ids,
            )
            text = tokenizer.batch_decode(out_gen)
            model_texts.append(text)
        all_texts[model_name] = model_texts
        image_paths = [str(image_path) for image_path in image_paths]

    return wandb_config, wandb_name, all_texts, image_paths, model_names
