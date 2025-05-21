import json
import logging
from pathlib import Path

import torch
from repos.m4_1.m4.evaluation.generation.deprecated_generation.generate import (
    generate_image_conditioned_text_from_pretrained,
)

from m4.evaluation.generation.config import get_config


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    config = get_config()
    config.hparams.image_paths = [
        img_path for img_path in Path(config.hparams.image_dir).iterdir() if img_path.is_file()
    ]
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {DEVICE}")
    logger.info(f"Config: {config}")
    if config.hparams.wandb_dump_dir is None:
        logger.warning(
            "The generations produced cannot be logged/retrieved since no directory was specified to dump the data"
        )
    wandb_config, wandb_name, all_texts, image_paths, model_names = generate_image_conditioned_text_from_pretrained(
        config.hparams.model_paths,
        config.hparams.image_paths,
        config.hparams.prompt,
        config.hparams.tokenizer_infos,
        max_len=config.hparams.max_len,
        num_beams=config.hparams.num_beams,
        no_repeat_ngram_size=config.hparams.no_repeat_ngram_size,
        image_size=config.hparams.image_size,
        num_images_per_ex=config.hparams.num_images_per_ex,
    )
    if config.hparams.wandb_dump_dir is not None:
        gen_output = {
            "wandb_config": wandb_config,
            "wandb_name": wandb_name,
            "wandb_project": config.hparams.wandb_project,
            "wandb_entity": config.hparams.wandb_entity,
            "all_texts": all_texts,
            "image_paths": image_paths,
            "model_names": model_names,
        }
        id = config.hparams.job_id if config.hparams.job_id is not None else "0"
        dump_file = config.hparams.wandb_dump_dir / f"gen_{id}.json"
        with open(dump_file, "w") as f_object:
            json.dump(gen_output, f_object)
        logger.info(f"Generation infos for wandb dumped in {dump_file}")


if __name__ == "__main__":
    main()
