import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch
from simple_parsing import ArgumentParser
from simple_parsing.helpers import Serializable, list_field

from m4.evaluation.utils import EvaluationVersion
from m4.training.utils import VisionEncoderTypes


logger = logging.getLogger(__name__)


@dataclass
class CfgFileConfig:
    """Config file args"""

    load_config: Optional[Path] = None  # load config file
    save_config: Optional[Path] = None  # save config to specified file


@dataclass
class Hparams:
    """General Hyperparameters"""

    commit_hash: Optional[str] = None
    # deprecated in favor of batch_size_per_gpu
    batch_size: Optional[int] = None
    batch_size_per_gpu: int = 1000
    batch_size_per_gpu_dl: Optional[int] = None
    # deprecated in favor of batch_size_per_gpu
    mini_batch_size: Optional[int] = None
    select_n_examples: Optional[int] = None
    device: Optional[torch.device] = None
    only_load_datasets: bool = False
    save_to_jsonl: Optional[Path] = None
    dir_path_load_from_disk: Optional[Path] = None
    timeout: int = 1800 * 12  # 6h
    seed: int = 42
    show_gpu_mem_util: bool = False
    is_test: bool = False
    use_selected_prompt_template_ids: bool = False


class ShotSelectionMode(Enum):
    random = "random"
    rices = "rices"
    first_without_image = "first_without_image"


@dataclass
class InContextParams:
    """In context learning parameters"""

    num_shots: int = 0
    shot_selection_mode: ShotSelectionMode = ShotSelectionMode.rices
    vision_encoder_name: str = "openai/clip-vit-base-patch32"


@dataclass
class TextGenerationParams:
    """Text generation parameters"""

    num_beams: int = 3
    no_repeat_ngram_size: int = 0
    max_new_tokens: int = 15


class ModelPrecision(Enum):
    fp32 = torch.float32
    bf16 = torch.bfloat16
    fp16 = torch.float16


class DatasetSplit(Enum):
    default = "default"
    validation = "validation"
    test = "test"
    server_check = "server_check"


@dataclass
class TaskParams:
    """Task parameters"""

    evaluation_version: EvaluationVersion = EvaluationVersion.v2
    model_name: str = "openai/clip-vit-base-patch32"
    model_precision: ModelPrecision = ModelPrecision.fp32
    tokenizer_name: Optional[str] = None
    vision_encoder_max_image_size: int = 384
    vision_encoder_type: VisionEncoderTypes = VisionEncoderTypes.siglip
    do_tasks: List[str] = list_field("Food101ClipLinearProberAcc")
    in_context_params: InContextParams = InContextParams()
    tokenizer_use_fast: bool = True
    text_generation_params: TextGenerationParams = TextGenerationParams()
    dataset_split: DatasetSplit = DatasetSplit.default
    prompt_template_id: int = 0
    save_generations: bool = False
    scale_up_images: bool = False
    image_size_after_scaling: int = 4000


@dataclass
class Parameters(Serializable):
    """base options."""

    hparams: Hparams = Hparams()
    tasks: TaskParams = TaskParams()

    def __post_init__(self, should_verify: bool = True):
        """Post-initialization code"""
        # # Used to set different hyperparameters depending on the model
        # if self.hparams.arch == "FlamingoBase":
        #     self.model_param: FlamingoBase = FlamingoBase()

        # deprecation
        if self.hparams.batch_size is not None:
            if self.hparams.batch_size_per_gpu > 1:
                raise ValueError(
                    "as hparams.batch_size is deprecated - don't know how to proceed with both hparams.batch_size>1"
                    " and hparams.batch_size_per_gpu > 1"
                )
            else:
                logger.warn(
                    "will use the deprecated hparams.batch_size, but transition to hparams.batch_size_per_gpu instead"
                )
                self.hparams.batch_size_per_gpu = self.hparams.batch_size
        self.hparams.batch_size = None

        if self.hparams.batch_size_per_gpu_dl is None:
            self.hparams.batch_size_per_gpu_dl = self.hparams.batch_size_per_gpu

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
    if cfgfile.load_config is not None:
        file_config = Parameters.load(cfgfile.load_config)

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
    config = get_config()
    print(config)
