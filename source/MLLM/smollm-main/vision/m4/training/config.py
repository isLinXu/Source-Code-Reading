import json
import logging
import time
from dataclasses import InitVar, asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import git
import yaml
from simple_parsing import ArgumentParser, Serializable
from simple_parsing.helpers import dict_field, list_field

from m4.training.types import DatasetNames, DatasetTypes
from m4.training.utils import LoggingTypes


logger = logging.getLogger(__name__)


@dataclass
class CfgFileConfig:
    """Config file args"""

    # path to config file
    config: Optional[Path] = None
    # set to false if you don't want to save config automatically
    save_config: bool = True


@dataclass
class GlobalBatchSizeRampUp:
    """These are init variables that are used to set up the GBS ramp up protocol"""

    # global batch size ramp up protocol:
    #
    # 1. start with global batch size `start`
    # 2. every time the number of `samples` is consumed increment global batch size by `increment`
    # 3. repeat step 2 until global batch size reaches `finish`
    start: Optional[int] = None
    finish: Optional[int] = None
    increment: Optional[int] = None
    samples: Optional[int] = None


@dataclass
class GlobalBatchSizeRampUpRunningParams:
    """The are running variables that are used to tell when to increment GBS and when to stop doing
    that, they are never set directly in the config file, but are calculated when the training starts.
    """

    global_seen_samples: int = 0
    global_batch_size_current: int = 0
    next_goal_samples: int = 0
    grad_acc_size_current: int = 1


@dataclass
class Hparams:
    """General Hyperparameters"""

    # --------------------
    # General parameters
    # --------------------

    seed: int = 13
    # If set to True, the sole purpose of the job is to pre-process the dataset (i.e. the map
    # operations). The job will exit as soon as the dataset is pre-processed.
    just_preprocess: bool = False
    jz_job_time_sec: Optional[float] = None
    jz_start_time: float = time.time()
    job_id: Optional[int] = None
    timeout: int = 1800  # 30 min
    # set to False to ignore the optimizer states when loading from a checkpoint
    load_optimizer_states: Optional[bool] = True
    # set to False to disable this gpu memory saving method
    gradient_checkpointing: Optional[bool] = True

    # --------------------
    # Model-related hparams
    # --------------------
    tokenizer_name: str = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    # The value of the string will evaluated (i.e. interpreted) and must be a dict
    tokenizer_params: str = '{"use_fast":True}'
    tokenizer_add_tokens: str = "[]"
    # The value of the string will evaluated (i.e. interpreted). Unnecessary if tokenizer has a `pad_token`.
    tokenizer_add_special_tokens: str = "{}"
    model_name: str = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
    lora_name: Optional[str] = None
    revision: str = "main"
    model_config: Dict[str, Any] = dict_field(
        dict(
            vision_config=dict(
                vision_model_name=None,
            ),
            perceiver_config=dict(
                resampler_n_latents=64,
                resampler_depth=6,
                resampler_n_heads=16,
                resampler_head_dim=96,
            ),
            tie_word_embeddings=False,
            # Freeze different parts of the model
            freeze_lm_head=False,
            freeze_text_layers=True,
            freeze_text_module_exceptions=[],
            freeze_vision_layers=True,
            freeze_vision_module_exceptions=[],
        )
    )
    lora_config: Dict[str, Any] = dict_field(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
    )
    use_lora: bool = False
    # --------------------
    # Training parameters
    # --------------------
    resume_run: Optional[bool] = None
    do_validation: bool = True
    is_fine_tuning: bool = False
    training_image_size: int = 224

    # deprecated in favor of batch_size_per_gpu
    batch_size: Optional[int] = None
    batch_size_per_gpu: int = 1
    global_batch_size: Optional[int] = None

    global_batch_size_ramp_up: GlobalBatchSizeRampUp = GlobalBatchSizeRampUp()
    grad_acc_size: Optional[int] = 1

    grad_clip: float = 1.0

    # weights by which to multiply the loss of each dataset when accumulating gradients over datasets
    loss_weights_per_dataset: Optional[List[float]] = None
    # int(max_num_tokens / (batch_size * max_seq_len * grad_acc_size * num_processes))
    max_num_opt_steps: Optional[int] = 500_000
    max_num_opt_steps_this_run: Optional[int] = None
    max_num_epochs: Optional[int] = None

    # If the path appears the program will stop after finishing the current training step
    kill_switch_path: Optional[Path] = None

    # If the path appears the program will save a checkpoint and immediately delete this flag
    save_switch_path: Optional[Path] = None

    # --------------------
    # Logging parameters
    # --------------------
    train_logging_opt_steps: int = 50
    train_logging_per_dataset_suffix: str = ""

    # If a specific logging type is specified, per dataset information will be inserted inside
    # those logs.
    train_logging_per_dataset_info: List[LoggingTypes] = list_field(LoggingTypes.JSONL, LoggingTypes.WANDB)

    # If `train_logging_activations` is not empty, hooks will be inserted to the model to track
    # the min/max/std/norm of the activations and weights. This will slow down training.
    # See https://huggingface.co/docs/transformers/main/en/debugging#underflow-and-overflow-detection
    train_logging_activations: List[LoggingTypes] = list_field()
    train_logging_activations_opt_steps: Optional[int] = 25
    train_logging_grad_param_deepspeed: List[LoggingTypes] = list_field()
    train_logging_grad_param_deepspeed_opt_steps: int = 50
    val_logging_opt_steps: int = train_logging_opt_steps * 5
    val_inline_logging_opt_steps: int = train_logging_opt_steps
    train_saving_opt_steps: int = train_logging_opt_steps * 5
    save_dir: Optional[Path] = None
    upload_to_s3: bool = False
    train_log_mem_usage: bool = False
    timing_break_down: bool = False

    save_batch_max_idx: Optional[int] = None
    save_batch_min_idx: Optional[int] = None

    # ----------------------
    # Wandb Parameters
    # ----------------------
    wandb_enable: bool = False
    # name of the project
    wandb_project: str = "VLOOM"
    wandb_entity: str = "huggingfacem4"
    # name of the wandb entity
    wandb_log_freq: int = 50
    wandb_run_id: str = ""
    wandb_tags: Optional[List[str]] = None

    repo_commit_id: Optional[str] = None

    # ----------------------
    # Debug Parameters
    # ----------------------
    use_torch_profiler: bool = False

    # ----------------------
    # Lora + Freezing/Unfreezing Parameters
    # ----------------------
    patterns_to_loraify: Optional[List[List[str]]] = list_field(["model.layers", "proj"])
    patterns_to_unfreeze: Optional[List[List[str]]] = list_field(["perceiver"], ["modality"], ["additional"])


@dataclass
class ResumeParams:
    # ----------------------
    # Resume run Parameters
    # ----------------------
    # Need to make sure that resume_run is True to give an input here
    opt_step_dir: Optional[Path] = None
    accelerator_state_dir: Optional[Path] = None
    model_file: Optional[Path] = None
    lora_file: Optional[Path] = None
    model_config_file: Optional[Path] = None
    # Automatically resumes last run of the save_dir. Set to False to choose a specific run
    resume_last: bool = True
    train_logs: Dict = dict_field()
    resume_opt_step: int = 0
    resume_epoch: int = 0
    resume_dataset_state: List = list_field()

    gbs_running: GlobalBatchSizeRampUpRunningParams = GlobalBatchSizeRampUpRunningParams()


@dataclass
class DatasetParams:
    # This always need to be specified as it is needed by dataset utils down the line
    dataset_name: DatasetNames
    # max number of images per sample
    max_num_images: int = 5
    # maximum sequence length
    max_seq_len: int = 256
    training_datasets_paths: List[Path] = list_field()
    validation_datasets_paths: List[Path] = list_field()
    # if True, instead of split and pack, each instance in sample will be
    # either truncated or padded to the same length.
    pad_dataset: bool = True
    map_batch_size: int = 64
    # Preprocessing number of processes in map (not useful for processing on the fly)
    map_num_proc: Optional[int] = None
    # Decides how many number of samples/subsequence should be extracted from the
    # CM4 corpus when the dataset is to be padded irrelavent otherwise as full packing
    # is used
    max_num_samples_per_document: int = 1

    add_begin_of_doc_token: bool = True
    add_end_of_doc_token: bool = True

    shuffle_after_packing: bool = False

    # Parameters for T5 MLM
    t5_mlm_noise_density: float = 0.15
    t5_mlm_mean_noise_span_length: int = 3

    dataset_type: Optional[DatasetTypes] = None

    # Parameters for webdataset pipeline
    shuffle_initial_urls_list: bool = False
    shuffle_before_split_by_node_buffer_size: Optional[int] = None
    shuffle_before_split_by_worker_buffer_size: Optional[int] = None
    shuffle_after_tarfile_to_samples_buffer_size: Optional[int] = None
    shuffle_after_batching_buffer_size: Optional[int] = None

    # Parameters for random scale up of images during training
    pre_split_scale_up_max: Optional[float] = 0.0
    pre_split_scale_up_frequency: Optional[float] = 0.0
    scale_up_max: Optional[float] = None
    scale_up_frequency: Optional[float] = None
    min_image_size: int = 378
    max_image_size: int = 384

    # Parameter has to be set later once the vision_config is known.
    vision_encoder_max_image_size: int = 0


@dataclass
class ImageCaptionPairedDatasetParams(DatasetParams):
    dataset_type: DatasetTypes = DatasetTypes.IMAGE_CAPTION_PAIRS


@dataclass
class OCRDatasetParams(DatasetParams):
    dataset_type: DatasetTypes = DatasetTypes.OCR


@dataclass
class DOCVQADatasetParams(DatasetParams):
    dataset_type: DatasetTypes = DatasetTypes.DOCVQA


@dataclass
class VQAv2TaskFineTuningPairedDatasetParams(DatasetParams):
    dataset_type: DatasetTypes = DatasetTypes.VQAV2_TASK_FINETUNING


@dataclass
class WebDocumentsDatasetParams(DatasetParams):
    dataset_type: DatasetTypes = DatasetTypes.WEB_DOCUMENTS


@dataclass
class SFTDatasetParams(DatasetParams):
    dataset_type: DatasetTypes = DatasetTypes.SFT


@dataclass
class DataParams(Serializable):
    """Data Parameters"""

    # what software to use for the dataset
    use_webdataset: bool = False

    # number of workers for dataloaders int
    num_workers: int = 1
    # allow async faster data transfer to GPUs (only make sense when CUDA GPUs are available)
    # known to cause memory issues
    pin_memory: bool = False
    # Whether to use persistent workers for the dataloaders
    persistent_workers: bool = True
    realtime_processing: bool = False

    train_seed: int = 1
    val_seed: int = 2

    # can use one config for both train + validation or specific ones if need to be different
    select_n_examples: Optional[int] = None
    select_n_examples_train: Optional[int] = None
    select_n_examples_validation: Optional[int] = None

    # TODO: Move to per dataset params as it makes more sense there
    proba_interleaving_dataset: Optional[List[float]] = None

    pmd: ImageCaptionPairedDatasetParams = ImageCaptionPairedDatasetParams(dataset_name=DatasetNames.PMD)
    laion: ImageCaptionPairedDatasetParams = ImageCaptionPairedDatasetParams(dataset_name=DatasetNames.LAION)
    laion_coco: ImageCaptionPairedDatasetParams = ImageCaptionPairedDatasetParams(dataset_name=DatasetNames.LAION_COCO)
    tikz: ImageCaptionPairedDatasetParams = ImageCaptionPairedDatasetParams(dataset_name=DatasetNames.TIKZ)
    image_website_code: ImageCaptionPairedDatasetParams = ImageCaptionPairedDatasetParams(
        dataset_name=DatasetNames.IMAGE_WEBSITE_CODE
    )
    ocr: OCRDatasetParams = OCRDatasetParams(dataset_name=DatasetNames.OCR)
    docvqa: DOCVQADatasetParams = DOCVQADatasetParams(dataset_name=DatasetNames.DOCVQA)
    cm4: WebDocumentsDatasetParams = WebDocumentsDatasetParams(dataset_name=DatasetNames.CM4)
    wiki: WebDocumentsDatasetParams = WebDocumentsDatasetParams(dataset_name=DatasetNames.WIKI)
    vqav2_task_finetuning: VQAv2TaskFineTuningPairedDatasetParams = VQAv2TaskFineTuningPairedDatasetParams(
        dataset_name=DatasetNames.VQAV2_TASK_FINETUNING
    )
    sft: SFTDatasetParams = SFTDatasetParams(dataset_name=DatasetNames.SFT)


@dataclass
class OptimizerParams:
    """Optimization parameters"""

    # --------------------
    # vl optim parameters
    # --------------------
    vl_optim: str = "AdamW"
    vl_optim_params: Dict[str, Any] = dict_field(
        dict(
            # learning rate
            lr=1e-4,
            # betas for adam
            betas=(0.9, 0.999),
            weight_decay=0.1,
            no_decay=["bias", "alpha", "layernorm", "ln", "layer_norm", "perceiver_resampler"],
        )
    )

    vl_lr_scheduler: str = "get_constant_schedule_with_warmup"
    # number of warmup steps for the learning rate
    vl_lr_scheduler_params: Dict[str, Any] = dict_field(dict(num_warmup_steps=5_000, last_epoch=-1))
    z_loss: float = 0.0


@dataclass
class Parameters(Serializable):
    """base options."""

    hparams: Hparams = Hparams()
    optim_param: OptimizerParams = OptimizerParams()
    data_param: DataParams = DataParams()
    resume_param: ResumeParams = ResumeParams()
    should_verify: InitVar[bool] = True

    def verify(self, should_verify: bool):
        if not should_verify:
            return

        dict_rep = vars(self)
        expected = vars(self.__class__(should_verify=False))
        for key, value in dict_rep.items():
            if isinstance(value, dict):
                diff = set(value.keys()) - set(asdict(expected[key]).keys())
                raise TypeError(
                    f"{key} in {self.__class__.__name__} has extra keys: {diff}. Please fix your config if you are"
                    " using one."
                )
            if key not in expected:
                raise ValueError(f"{key} is not a valid parameter for {self.__class__.__name__}")

    def __post_init__(self, should_verify: bool = True):
        """Post-initialization code"""
        self.verify(should_verify=should_verify)

        # copy select_n_examples to the more specific ones if the latter haven't been preset
        if self.data_param.select_n_examples is not None:
            if self.data_param.select_n_examples_train is None:
                self.data_param.select_n_examples_train = self.data_param.select_n_examples
            if self.data_param.select_n_examples_validation is None:
                self.data_param.select_n_examples_validation = self.data_param.select_n_examples

        # Get commit id
        if self.hparams.repo_commit_id is None:
            self.hparams.repo_commit_id = git.Repo(search_parent_directories=True).head.object.hexsha

        if self.hparams.lora_name is not None and not self.hparams.use_lora:
            raise ValueError("Can't have a lora_name if use_lora is False")

        # If processing on the fly, with the current implementation, we can't have `num_workers=0`
        if self.data_param.realtime_processing and self.data_param.num_workers == 0:
            raise ValueError(
                "If doing processing on the fly (and thus using the `IterableDataset`), you can't have `num_workers`"
                " equal to 0."
            )

        # batch_size deprecation
        if self.hparams.batch_size is not None:
            if self.hparams.batch_size_per_gpu > 1:
                raise ValueError(
                    "as hparams.batch_size is deprecated - don't know how to proceed with both hparams.batch_size>1"
                    " and hparams.batch_size_per_gpu > 1"
                )
            else:
                logger.warning(
                    "will use the deprecated hparams.batch_size, but transition to hparams.batch_size_per_gpu instead"
                )
                self.hparams.batch_size_per_gpu = self.hparams.batch_size
        self.hparams.batch_size = None

        # Assign batch size to data_param as well for dataloaders
        self.data_param.batch_size = self.hparams.batch_size_per_gpu

        # note: all global batch_size-related configs including hparams.grad_acc_size will be
        # checked/set in trainer's setup_batch_size_related_configs since we need to know the value
        # of num_processes

        # Assign loggingtypes given values
        self.hparams.train_logging_activations = [LoggingTypes(val) for val in self.hparams.train_logging_activations]

        # Check that proba_interleaving_dataset is mutually exclusive to loss_weights_per_dataset
        if self.data_param.proba_interleaving_dataset and self.hparams.loss_weights_per_dataset:
            raise ValueError(
                "Can't have hparams.loss_weights_per_dataset and proba_interleaving_dataset. If we have"
                " loss_weights_per_dataset, it means the gradients are accumulated over datasets. Therefore a batch of"
                " each given at each update and there is no use of proba_interleaving_dataset"
            )

        if (
            self.data_param.proba_interleaving_dataset is not None
            and sum(self.data_param.proba_interleaving_dataset) != 1
        ):
            if abs(sum(self.data_param.proba_interleaving_dataset) - 1) > 0.001:
                # Allow a tolerance for floating points rounding errors.
                raise ValueError("proba_interleaving_dataset must sum to 1")

        if self.hparams.use_lora:
            has_vision_lora = any(["vision" in pattern for pattern in self.hparams.patterns_to_loraify])
            has_text_lora = any(["model.layers" in pattern for pattern in self.hparams.patterns_to_loraify])
            if has_vision_lora and not self.hparams.model_config["freeze_vision_layers"]:
                raise ValueError(
                    "hparams.patterns_to_loraify suggests Lora is applied on the vision backbone, so"
                    " model_config.freeze_vision_layers should be True, but it is set to False"
                )
            if has_text_lora and not self.hparams.model_config["freeze_text_layers"]:
                raise ValueError(
                    "hparams.patterns_to_loraify,suggests Lora is applied on the text backbone, so"
                    " model_config.freeze_text_layers should be True, but it is set to False"
                )

        self.hparams.train_logging_grad_param_deepspeed = [
            LoggingTypes(val) for val in self.hparams.train_logging_grad_param_deepspeed
        ]

        # Resume run if there is already an existing folder for this run
        if self.hparams.save_dir is not None and self.hparams.save_dir.exists():
            save_dir_has_checkpoints = (
                len([dir for dir in self.hparams.save_dir.iterdir() if (dir.is_dir() and "opt_step" in str(dir))]) > 0
            )
            if self.hparams.resume_run is not None and not self.hparams.resume_run and save_dir_has_checkpoints:
                logger.warning(
                    "`resume_run` was explicitely set to False (i.e. starting from scratch), but the experiment"
                    " folder already has been populated with previous runs.\nAlready saved checkpoints will be"
                    " overwritten (at best, when `train_saving_opt_steps` is the same) or will be mixed with the new"
                    " checkpoints of a potentially brand new experiment. Would it make sense to create a new"
                    " `save_dir`?"
                )
            self.hparams.resume_run = save_dir_has_checkpoints

        # Setup all args needed to resume a run
        if self.hparams.resume_run:
            # Get last step directory
            if self.resume_param.opt_step_dir is None and not self.resume_param.resume_last:
                raise ValueError(
                    "`opt_step_dir` cannot be None while `resume_last` is False. Choose which dir you want to resume"
                    " from..."
                )
            if self.resume_param.resume_last:
                if self.resume_param.opt_step_dir is not None:
                    raise ValueError(
                        "`resume_last` cannot be True while `opt_step_dir` is not None. Choose which dir you want to"
                        " resume from..."
                    )
                latest_path = self.hparams.save_dir / "latest_opt_step_dir"
                with open(latest_path, "r") as fd:
                    self.resume_param.opt_step_dir = Path(fd.read().strip())
                if not (self.resume_param.opt_step_dir.exists() and self.resume_param.opt_step_dir.is_dir()):
                    raise ValueError(
                        f"It appears that the path in the `latest_opt_step_dir` file {latest_path} is invalid. It's"
                        " either does not exist or is not a directory. Please fix that."
                    )

            with open(self.resume_param.opt_step_dir / "resume_run_infos.json", "r") as f:
                resume_infos = json.load(f)
            logger.info(f"Resuming from {self.resume_param.opt_step_dir}")
            self.resume_param.accelerator_state_dir = self.resume_param.opt_step_dir / "accelerator_state"
            self.resume_param.model_file = self.resume_param.opt_step_dir / "unwrapped_model"
            self.resume_param.lora_file = self.resume_param.opt_step_dir / "unwrapped_adapter"
            self.resume_param.model_config_file = self.resume_param.opt_step_dir / "unwrapped_model/config.json"
            self.resume_param.tokenizer = self.resume_param.opt_step_dir / "tokenizer"

            self.resume_param.train_logs = resume_infos["train_logs"]
            self.resume_param.resume_opt_step = resume_infos["resume_opt_step"]
            self.resume_param.resume_epoch = resume_infos["resume_epoch"]
            self.resume_param.resume_dataset_state = resume_infos.get("resume_dataset_state", list())

            gbs_running = resume_infos["gbs_running"]
            self.resume_param.gbs_running.global_batch_size_current = gbs_running["global_batch_size_current"]
            self.resume_param.gbs_running.global_seen_samples = gbs_running["global_seen_samples"]
            self.resume_param.gbs_running.next_goal_samples = gbs_running["next_goal_samples"]
            self.resume_param.gbs_running.grad_acc_size_current = gbs_running["grad_acc_size_current"]

            self.hparams.wandb_run_id = resume_infos["wandb_run_id"]
            self.hparams.seed = resume_infos["seed"]

            if not self.hparams.wandb_enable:
                self.hparams.wandb_run_id = ""

    @classmethod
    def parse(cls):
        cfgfile_parser = ArgumentParser(add_help=False)
        cfgfile_parser.add_arguments(CfgFileConfig, dest="cfgfile")
        cfgfile_args, rest = cfgfile_parser.parse_known_args()

        cfgfile: CfgFileConfig = cfgfile_args.cfgfile

        file_config: Optional[Parameters] = None
        if cfgfile.config is not None:
            file_config = Parameters.load(cfgfile.config, load_fn=yaml.safe_load)

        parser = ArgumentParser()

        # add cfgfile args so they appear in the help message
        parser.add_arguments(CfgFileConfig, dest="cfgfile")
        parser.add_arguments(Parameters, dest="parameters", default=file_config)

        # XXX: currently when called from tests we don't want to parse pytest arguments, so either
        # this whole logic needs to be rewritten to not always call `parser.parse_args` but only
        # when needed, for now as a workaround using `parse_known_args` and ignoring the args which
        # don't belong to this program
        args, unknown = parser.parse_known_args()

        parameters: Parameters = args.parameters
        parameters.save_config = cfgfile.save_config

        return parameters

    def save_config_state(self):
        if self.save_config:
            self.hparams.save_dir.mkdir(parents=True, exist_ok=True)
            if self.hparams.job_id is not None:
                config_file_name = f"{self.hparams.job_id}_config.yaml"
            else:
                config_file_name = "config.yaml"
            self.save(self.hparams.save_dir / config_file_name)


def get_config(print_config: bool = True):
    parameters: Parameters = Parameters.parse()
    if print_config:
        print(parameters)
    return parameters


if __name__ == "__main__":
    config = get_config()
    config.save_config_state()
