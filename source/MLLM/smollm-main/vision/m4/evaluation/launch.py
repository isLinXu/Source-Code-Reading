# TODO LAUNCH EVALUATTION
# evaluate --model clip --predictor linearprobe

import json
import logging
import os
from datetime import timedelta
from time import time

import accelerate
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from rich.logging import RichHandler

import m4.evaluation.tasks as tasks
from m4.evaluation.config import get_config
from m4.evaluation.evaluators import in_contexter, linear_prober
from m4.evaluation.tasks.base import Predictor
from m4.evaluation.utils import get_model_from_config_file, get_prompt_template_id
from m4.utils.check_valid_tokenizer import check_valid_tokenizer


logging.basicConfig(
    level=logging.INFO,
    format=" - %(process)d - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[RichHandler(level=logging.INFO)],
)
logger = logging.getLogger(__name__)
ALL_TASKS = [
    *tasks.VGPT2_TASKS[Predictor.in_contexter],
]
ALL_TASKS = [task_class.__name__ for task_class in ALL_TASKS]


EVALUATOR_MAPPING = {
    Predictor.in_contexter: in_contexter,
    Predictor.linear_prober: linear_prober,
}


def main(args):
    accelerate.utils.set_seed(args.hparams.seed)
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=args.hparams.timeout))]
    rngs_types = ["torch", "cuda", "generator"] if torch.cuda.is_available() else ["torch", "generator"]
    accelerator = Accelerator(rng_types=rngs_types, kwargs_handlers=kwargs_handlers)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args.hparams.save_to_jsonl.parent.mkdir(parents=True, exist_ok=True)

    start = time()
    args.hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.tasks.model_name
    tokenizer_name = args.tasks.tokenizer_name if args.tasks.tokenizer_name is not None else args.tasks.model_name
    evaluation_version = args.tasks.evaluation_version
    tokenizer_use_fast = args.tasks.tokenizer_use_fast
    vision_encoder_type = args.tasks.vision_encoder_type
    scale_up_images = args.tasks.scale_up_images
    image_size_after_scaling = args.tasks.image_size_after_scaling

    do_tasks = args.tasks.do_tasks if args.tasks.do_tasks != ["all"] else ALL_TASKS
    model = get_model_from_config_file(args, is_deepspeed=accelerator.distributed_type == DistributedType.DEEPSPEED)
    model_config = model.config
    logger.info("Model loaded.")

    vision_encoder_max_image_size = model_config.vision_config.image_size
    image_seq_len = (
        model_config.perceiver_config.resampler_n_latents
        if model_config.use_resampler
        else int(((vision_encoder_max_image_size // model_config.vision_config.patch_size) ** 2) / 9)
    )
    dummy_dataloader = torch.utils.data.DataLoader([0 for _ in range(20)], batch_size=args.hparams.batch_size_per_gpu)
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        # Deepspeed doesn't allow custom device placement
        _, model = accelerator.prepare(dummy_dataloader, model)
    else:
        _, model = accelerator.prepare(
            dummy_dataloader,
            model,
            device_placement=[
                False,
                True,
            ],  # Only letting accelerate handle the device placement for the model. For the dataloader, as it loads very big batches to then mini split them, we handle the device placement mini batch by mini batch.
        )

    for current_task in do_tasks:
        task_class = getattr(tasks, current_task)
        task = task_class(
            model_name=model_name,
            tokenizer_name=tokenizer_name,
            tokenizer_use_fast=tokenizer_use_fast,
            evaluation_version=evaluation_version,
            vision_encoder_max_image_size=vision_encoder_max_image_size,
            vision_encoder_type=vision_encoder_type,
            image_seq_len=image_seq_len,
            scale_up_images=scale_up_images,
            image_size_after_scaling=image_size_after_scaling,
        )
        check_valid_tokenizer(task.tokenizer)

        logger.info(f" *** Running {current_task} *** ")
        evaluator = EVALUATOR_MAPPING[task.predictor_class]
        score = evaluator(task, accelerator, model, args)
        logger.info(f"{current_task} {score}")

        is_main_process = True

        try:
            from accelerate.state import AcceleratorState

            state = AcceleratorState()
            is_main_process = state.process_index == 0
        except ValueError:
            # We are not using Accelerate
            pass

        if args.hparams.save_to_jsonl is not None and is_main_process:
            prompt_template_id = get_prompt_template_id(args, task)
            with open(args.hparams.save_to_jsonl, "a", newline="") as f_object:
                data_dict = {
                    "model_name_or_path": model_name,
                    "task": current_task,
                    "score": str(score),
                    "evaluator": evaluator.__name__,
                    "in_context_params": {
                        "num_shots": args.tasks.in_context_params.num_shots,
                        "shot_selection_mode": args.tasks.in_context_params.shot_selection_mode.name,
                        "vision_encoder": args.tasks.in_context_params.vision_encoder_name,
                    },
                    "text_generation_params": {
                        "num_beams": args.tasks.text_generation_params.num_beams,
                        "no_repeat_ngram_size": args.tasks.text_generation_params.no_repeat_ngram_size,
                        "max_new_tokens": args.tasks.text_generation_params.max_new_tokens,
                    },
                    "evaluation_version": evaluation_version.name,
                    "commit_hash": args.hparams.commit_hash,
                    "prompt_template_id": prompt_template_id,
                    "dataset_split": args.tasks.dataset_split.value,
                    "scale_up_images": args.tasks.scale_up_images,
                    "image_size_after_scaling": args.tasks.image_size_after_scaling,
                }
                json.dump(data_dict, f_object)
                f_object.write(os.linesep)

    end = time()
    logger.info(f"Took {end-start} seconds.")


if __name__ == "__main__":
    args = get_config()
    main(args)
