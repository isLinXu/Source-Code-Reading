import json
import logging
import os
from datetime import timedelta
from pathlib import Path

import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType, extract_model_from_parallel
from deepspeed.runtime.engine import DeepSpeedEngine
from deepspeed.utils.z3_leaf_module import set_z3_leaf_modules
from PIL import Image
from transformers import AutoTokenizer

from m4.evaluation.generation.config import get_config
from m4.models.__init__ import model_type_to_modeling_class
from m4.training.utils import build_image_transform


logger = logging.getLogger(__name__)


def load_tokenizer_model(opt_step_dir):
    model_name = Path(f"{opt_step_dir}/unwrapped_model")
    tokenizer_path = Path(f"{opt_step_dir}/tokenizer")
    adapter_path = Path(f"{opt_step_dir}/unwrapped_adapter")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = "left"

    # Get model
    with open(f"{model_name}/config.json", "r") as f:
        model_config = json.loads(f.read())
        model_class = model_type_to_modeling_class.get(model_config["model_type"], None)
        model = model_class.from_pretrained(model_name, token=os.getenv("HF_TOKEN", True))
    if adapter_path.exists():
        model.load_adapter(adapter_path)
        model.enable_adapters()
        logger.info("Loaded adapter for model")

    logger.info(f"Loaded model {model.__class__.__name__} from checkpoint {model_name}")
    model.eval()
    set_z3_leaf_modules(model, [model_class])
    return tokenizer, model


def fetch_images(images_paths):
    images = []
    for image_path in images_paths:
        images.append(Image.open(image_path))
    return images


def prepare_model(model, accelerator):
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        dummy_dataloader = torch.utils.data.DataLoader([0 for _ in range(10)], batch_size=1)
        # Deepspeed doesn't allow custom device placement
        _, model = accelerator.prepare(dummy_dataloader, model)
    else:
        model = accelerator.prepare(model)

    if isinstance(model, DeepSpeedEngine):
        if model.zero_optimization_partition_weights():
            # Enable automated discovery of external parameters by indicating that
            # we are in a forward pass.
            for module in model.module.modules():
                module._parameters._in_forward = True
                pass
    unwrapped_model = extract_model_from_parallel(model)
    return unwrapped_model


def model_generation(
    prompts,
    all_images,
    all_image_paths,
    tokenizer,
    model,
    accelerator,
    image_size,
    vision_encoder_type,
    num_beams,
    no_repeat_ngram_size,
    max_new_tokens,
    min_length,
    ban_tokens,
    hide_special_tokens,
    length_penalty,
    repetition_penalty,
    add_special_tokens,
    output_file,
):
    all_texts_gen = []
    all_image_paths_gen = []
    single_image_seq_len = (
        model.config.perceiver_config.resampler_n_latents
        if model.config.use_resampler
        else int(((model.config.vision_config.image_size // model.config.vision_config.patch_size) ** 2) / 9)
        # else (model.config.vision_config.image_size // model.config.vision_config.patch_size) ** 2
    )
    image_transform = build_image_transform(
        max_image_size=model.config.vision_config.image_size if image_size is None else None,
        min_image_size=378 if image_size is None else None,
        image_size=image_size,
        vision_encoder_type=vision_encoder_type,
        eval=True,
    )

    prompts = [prompt.replace("<image>", "<image>" * single_image_seq_len) for prompt in prompts]
    bad_words_ids = [tokenizer(banned_token, add_special_tokens=False).input_ids for banned_token in ban_tokens]
    tokens = tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
        add_special_tokens=add_special_tokens,
    )
    input_ids = torch.stack([tokens.input_ids[idx] for idx in range(len(prompts))]).to(model.device)
    attention_mask = torch.stack([tokens.attention_mask[idx] for idx in range(len(prompts))]).to(model.device)
    for image, image_path in zip(all_images, all_image_paths):
        example_images = [image] * len(prompts)
        pixel_values = [torch.stack([image_transform(img) for img in example_images])]
        pixel_values = torch.stack(pixel_values, dim=1).to(model.device)
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                bad_words_ids=bad_words_ids,
                length_penalty=length_penalty,
                use_cache=True,
                early_stopping=True,
                synced_gpus=True,
                repetition_penalty=repetition_penalty,
            )
        if accelerator.is_main_process:
            tokens = [tokenizer.convert_ids_to_tokens(ex) for ex in generated_tokens]
            decoded_skip_special_tokens = repr(
                tokenizer.batch_decode(generated_tokens, skip_special_tokens=hide_special_tokens)[0]
            )
            decoded = repr(tokenizer.batch_decode(generated_tokens))

            logger.info(
                "Result: \n"
                f"Ids from prompt + generation: {generated_tokens}\n"
                f"Tokens from prompt + generation: {tokens}\n"
                f"Tokens decoded: {decoded_skip_special_tokens}\n"
                f"String decoded: {decoded}\n"
                f"generated with prompts: {prompts}\n"
                f"               num_beams: {num_beams}\n"
                f"               no_repeat_ngram_size: {no_repeat_ngram_size}\n"
                f"               max_new_tokens: {max_new_tokens}\n"
                f"               min_length: {min_length}\n"
                f"               length_penalty: {length_penalty}\n"
                f"               repetition_penalty: {repetition_penalty}\n"
            )

            original_prompt = generated_tokens[:, : input_ids.shape[-1]]
            actual_generated_tokens = generated_tokens[:, input_ids.shape[-1] :]
            displayed_tokens = torch.cat([original_prompt, actual_generated_tokens], dim=-1)
            generated_texts = tokenizer.batch_decode(displayed_tokens, skip_special_tokens=hide_special_tokens)
            generated_texts = [generated_text for generated_text in generated_texts]
            image_paths = [image_path for _ in range(len(generated_texts))]
            all_texts_gen.extend(generated_texts)
            all_image_paths_gen.extend(image_paths)
            jsonl_generations = []
            for text, image_path in zip(generated_texts, image_paths):
                jsonl_generations.append({"image_path": f"{image_path}", "text": f"{text}"})
            with open(output_file, "a+") as file:
                for r in jsonl_generations:
                    file.write(json.dumps(r) + "\n")

    return (all_texts_gen, all_image_paths_gen)


def main():
    config = get_config()
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=1800))]
    rngs_types = ["torch", "cuda", "generator"] if torch.cuda.is_available() else ["torch", "generator"]
    accelerator = Accelerator(rng_types=rngs_types, kwargs_handlers=kwargs_handlers)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    images_paths = [img_path for img_path in Path(config.hparams.image_dir).iterdir() if img_path.is_file()]
    all_images = fetch_images(images_paths)
    tokenizer, model = load_tokenizer_model(
        opt_step_dir=config.hparams.opt_step_dir,
    )
    model = prepare_model(model, accelerator)
    model.eval()
    generated_texts, corresponding_image_paths = model_generation(
        prompts=config.hparams.prompts,
        all_images=all_images,
        all_image_paths=images_paths,
        tokenizer=tokenizer,
        model=model,
        accelerator=accelerator,
        image_size=config.hparams.image_size,
        vision_encoder_type=config.hparams.vision_encoder_type,
        num_beams=config.hparams.num_beams,
        no_repeat_ngram_size=config.hparams.no_repeat_ngram_size,
        max_new_tokens=config.hparams.max_new_tokens,
        min_length=config.hparams.min_length,
        ban_tokens=config.hparams.ban_tokens,
        hide_special_tokens=config.hparams.hide_special_tokens,
        length_penalty=config.hparams.length_penalty,
        repetition_penalty=config.hparams.repetition_penalty,
        add_special_tokens=config.hparams.add_special_tokens,
        output_file=config.hparams.generation_output_file,
    )
    if accelerator.is_main_process:
        for text, image_path in zip(generated_texts, corresponding_image_paths):
            print(f"image name: {str(image_path).split('/')[-1]} | Generated text: {text}")


if __name__ == "__main__":
    main()
