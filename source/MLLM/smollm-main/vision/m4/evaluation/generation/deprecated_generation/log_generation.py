import json
import logging
import os
from dataclasses import dataclass

import torchvision.transforms as transforms
import wandb
from PIL import Image
from simple_parsing import ArgumentParser, Serializable

from m4.training.utils import _convert_to_rgb


logger = logging.getLogger(__name__)


@dataclass
class Parameters(Serializable):
    """base options."""

    gen_file: str = ""

    @classmethod
    def parse(cls):
        parser = ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance


def log_generation(
    wandb_config, wandb_name, wandb_project, wandb_entity, all_texts, image_paths, model_names, image_size=224
):
    image_size = image_size
    image_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=Image.BICUBIC),
        ]
    )
    list_images = [image_transform(_convert_to_rgb(Image.open(image_path))) for image_path in image_paths]
    columns = ["image"]
    for model_name in model_names:
        columns.append(f"{model_name}_text")

    wandb.init(
        config=wandb_config,
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name,
    )
    my_table = wandb.Table(columns=columns)
    for i, image in enumerate(list_images):
        table_inputs = (wandb.Image(image),)
        for j, model_name in enumerate(model_names):
            table_inputs += (repr(all_texts[model_name][i]),)
        my_table.add_data(*table_inputs)

    wandb.log({"Image conditionned text generation table": my_table})


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = Parameters.parse()
    with open(args.gen_file, "r") as j:
        gen_file = json.loads(j.read())
    log_generation(
        gen_file["wandb_config"],
        gen_file["wandb_name"],
        gen_file["wandb_project"],
        gen_file["wandb_entity"],
        gen_file["all_texts"],
        gen_file["image_paths"],
        gen_file["model_names"],
    )

    os.remove(gen_file)
    logger.info("wandb sync completed and temp_file {gen_file} removed")


if __name__ == "__main__":
    main()
