import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from simple_parsing import ArgumentParser, Serializable, list_field

from m4.training.utils import VisionEncoderTypes


logger = logging.getLogger(__name__)


@dataclass
class Hparams:
    """General Hyperparameters"""

    # Model checkpoint path. Should look like this: "/gpfsscratch/rech/cnw/commun/experiments/local_experiment_dir/tr_47_cm4_filtered_35/opt_step-43499"
    opt_step_dir: Optional[Path] = None
    # File in which generations should be dumped
    generation_output_file: Optional[Path] = None
    # Directory of sample images used to condition the generations. ex: /gpfsscratch/rech/cnw/commun/local_datasets/sample_images
    image_dir: Optional[Path] = None
    # Directory used to dump information necessary for wandb to log all the data in another job (using prepost, otherwise impossible to log this data offline)
    wandb_dump_dir: Optional[Path] = None
    # Use fast tokenizer. Defaults to True
    tokenizer_fast: bool = True
    # Number of beams used in the model.generate() function. Set to None for greedy generation
    num_beams: int = 1
    # Prevents repetition of ngrams. size of the ngrams
    no_repeat_ngram_size: int = 0
    # Max length of the generations
    max_new_tokens: int = 50
    # Min length of the generations
    min_length: int = 0
    # Tokens to prevent from being generated (separated by ";")
    ban_tokens: List[str] = list_field("<image>", "<fake_token_around_image>")
    # add special tokens
    add_special_tokens: bool = False
    # Hide special tokens in the text
    hide_special_tokens: bool = False
    # length_penalty
    length_penalty: float = 1.0
    # repetition_penalty
    repetition_penalty: float = 1.0

    # General prompt kickstarting generation for all images and all models. Should start by an image token
    prompts: List[str] = list_field(
        "<fake_token_around_image><image><fake_token_around_image>Question: What's in the image? Answer:"
    )

    model_class: str = "IdeficsForCausalLM"
    config_class: str = "IdeficsConfig"
    # tokenizer used for all models. Should change to use the one stored with models
    job_id: Optional[str] = None
    image_size: Optional[int] = None
    vision_encoder_type: VisionEncoderTypes = VisionEncoderTypes.siglip
    # Number of images used per cationning examples. For now it's always attendoing to the samme image.
    # Added this to the config instead of hardcoding because we may want to use multiple images
    wandb_entity: str = "huggingfacem4"
    wandb_project: str = "VLOOM_generations"


@dataclass
class CfgFileConfig:
    """Config file args"""

    config: Optional[Path] = None  # load config file
    save_config: Optional[Path] = None  # save config to specified file


@dataclass
class Parameters(Serializable):
    """base options."""

    hparams: Hparams = Hparams()

    @classmethod
    def parse(cls):
        parser = ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance


def get_config(args=None):
    # taken from https://github.com/lebrice/SimpleParsing/issues/45#issuecomment-1035224416
    cfgfile_parser = ArgumentParser(add_help=False)
    cfgfile_parser.add_arguments(CfgFileConfig, dest="cfgfile")
    cfgfile_args, _ = cfgfile_parser.parse_known_args()

    cfgfile: CfgFileConfig = cfgfile_args.cfgfile

    file_config: Optional[Parameters] = None
    if cfgfile.config is not None:
        file_config = Parameters.load(cfgfile.config)

    parser = ArgumentParser()

    # add cfgfile args so they appear in the help message
    parser.add_arguments(CfgFileConfig, dest="cfgfile")
    parser.add_arguments(Parameters, dest="my_preferences", default=file_config)

    args = parser.parse_args()

    prefs: Parameters = args.my_preferences
    print(prefs)

    if cfgfile.save_config is not None:
        prefs.save(cfgfile.save_config, indent=4)
    return prefs


if __name__ == "__main__":
    get_config()
