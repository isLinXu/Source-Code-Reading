import copy
import gc
import json
import logging
import os
import pickle
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

import accelerate
import psutil
import torch
import torch.optim as torch_optim
import transformers.optimization as transformers_optim
import wandb
from packaging import version

from m4.training.config import (
    DataParams,
    GlobalBatchSizeRampUpRunningParams,
    Hparams,
    OptimizerParams,
    Parameters,
    ResumeParams,
)
from m4.training.dataset import DatasetNames
from m4.training.debug_utils import validate_optim_states_are_reset
from m4.training.utils import (  # deepspeed_gathered_parameters_context_manager,
    IMAGE_TOKEN,
    JSONEncoderForDataclasses,
    LoggingTypes,
    SigtermListener,
    get_deepspeed_engine,
    get_stats,
    get_stats_format,
    is_deepspeed_used,
    is_deepspeed_zero3_used,
    is_deepspeed_zero_init_enabled,
    lora_unload,
    mem_usage_formatted,
    pynmvl_handle,
    pynvml_get_total_energy_in_joules,
)
from m4.utils.activation_tracker import ActivationTracker
from m4.utils.debug import printflock as print
from m4.utils.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TimeElapsedColumn
from m4.utils.training.timer import DeviceAgnosticTimer, Timer, format_secs_to_sec_fractions, format_secs_to_time


logger = logging.getLogger(__name__)

fqdn = socket.getfqdn()
if "compute.internal" in fqdn:
    # hfc: 1.1TB RAM
    _MEMORY_EXPLOSION_THRESHOLD = 93.0
elif "idris.fr" in fqdn or "idrsrv" in fqdn:
    # jz: 0.5TB RAM
    _MEMORY_EXPLOSION_THRESHOLD = 90.0
else:
    _MEMORY_EXPLOSION_THRESHOLD = 90.0


METRICS_TO_DEFAULT_VALUE_FN = {
    "lr": lambda: None,
    "num_opt_steps": lambda: 0,
    "num_epochs": lambda: 0,
    "per_token_loss": lambda: defaultdict(lambda: None),
    "z_loss": lambda: defaultdict(lambda: None),
    "watt/s": lambda: defaultdict(lambda: None),
    "tflops": lambda: defaultdict(lambda: None),
    "tflop_counter": lambda: defaultdict(lambda: None),
    "fwd_bwd_time": lambda: defaultdict(lambda: None),
    "tflops_acc": lambda: defaultdict(lambda: None),
    "num_per_device_batches": lambda: defaultdict(lambda: None),
    "num_images": lambda: defaultdict(lambda: None),
    "num_image_tokens": lambda: defaultdict(lambda: None),
    "num_tokens": lambda: defaultdict(lambda: None),
    "image_to_text_ratio": lambda: defaultdict(lambda: None),
    "pixel_values_sum": lambda: defaultdict(lambda: None),
    "num_padding": lambda: defaultdict(lambda: None),
    "num_per_device_batches_in_curr_epoch": lambda: defaultdict(lambda: None),
    "num_batches": lambda: defaultdict(int),
    "num_batches_in_curr_epoch": lambda: defaultdict(int),
    "per_token_loss_acc": lambda: defaultdict(lambda: None),
    "z_loss_acc": lambda: defaultdict(lambda: None),
    "num_batches_since_training_logged": lambda: defaultdict(int),
    "num_per_device_batches_since_training_logged": lambda: defaultdict(lambda: None),
    "tflop_counter_since_training_logged": lambda: defaultdict(lambda: None),
    "total_energy_delta_since_training_logged": lambda: defaultdict(lambda: None),
    "fwd_bwd_time_since_training_logged": lambda: defaultdict(lambda: None),
}

METRICS_TO_RESET_AFTER_LOGGING = [
    "per_token_loss_acc",
    "z_loss_acc",
    "num_batches_since_training_logged",
    "num_per_device_batches_since_training_logged",
    "tflop_counter_since_training_logged",
    "total_energy_delta_since_training_logged",
    "fwd_bwd_time_since_training_logged",
]


class Trainer(object):
    """
    Trainer object to monitor training and validation
    :- config -: json config file
    """

    def __init__(self, accelerator, vl_model, tokenizer, train_loader, val_loader, config):
        # Initialize params
        self.config: Parameters = config
        self.optim_param: OptimizerParams = config.optim_param
        self.hparams: Hparams = config.hparams
        self.resume_param: ResumeParams = config.resume_param
        self.data_param: DataParams = config.data_param

        # Initialize last step directory
        self.last_opt_step_dir = ""

        # Initialize the model
        self.vl_model = vl_model

        # Gradient checkpointing
        if self.hparams.gradient_checkpointing:
            self.vl_model.gradient_checkpointing_enable()

        # Debug
        if accelerator.is_main_process and self.hparams.train_logging_activations:
            self.activation_tracker = ActivationTracker(self.vl_model)
        else:
            self.activation_tracker = None

        # Initialize tokenizer
        self.tokenizer = tokenizer
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        # Initialize accelerator
        self.accelerator = accelerator

        # Initialize loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Checks
        self._compatibility_checks()

        # Initialize everything related to distributed training
        self._configure_optimizer_and_scheduler()

        # Prepare and/or register model, optimizer, dataloaders and scheduler
        self._prepare_register()

        # now that we have num_processes, figure out batch_size-related variables
        self.setup_batch_size_related_configs()

        # Compute useful variables
        self.optim_param.opt_batch_size = self.hparams.global_batch_size

        if self.hparams.max_num_opt_steps is None and self.hparams.max_num_epochs is None:
            if hasattr(self.train_loader, "__len__") and self.hparams.global_batch_size_ramp_up.start is not None:
                raise ValueError("Currently global batch size ramp up doesn't work with MappedDataset")

            try:
                self.hparams.max_num_opt_steps = int(len(self.train_loader) // self.hparams.grad_acc_size)
            except TypeError:
                raise ValueError("max_num_opt_steps or max_num_epochs must be defined if you use IterableDataset")
        # self._set_model_tflops_per_batch_per_gpu()

        # Init trackers
        self._init_trackers()

        # Handle jz timing and memory
        self.jz_training_time_over = [False]
        self.memory_explosion = False

        # Stopping on demand
        self.kill_switch_activated = False

        # Sigterm signal listener
        self.sigterm_signal_received = False
        self.sigterm_listener = SigtermListener()

        sizes = defaultdict(int)
        trainable_params = []
        numel_fn = lambda p: p.ds_numel if is_deepspeed_zero_init_enabled() else p.numel()  # noqa
        for name, param in self.accelerator.unwrap_model(self.vl_model).named_parameters():
            numel = numel_fn(param)
            sizes["total"] += numel
            sizes["total_lora"] += numel if "lora_" in name else 0
            if "vision_model" in name:
                sizes["vision_model"] += numel
                sizes["vision_model_lora"] += numel if "lora_" in name else 0
            if "perceiver_resampler" in name:
                sizes["perceiver_resampler"] += numel
            if "modality_projection" in name:
                sizes["modality_projection"] += numel
            if param.requires_grad:
                sizes["trainable"] += numel
                sizes["trainable_lora"] += numel if "lora_" in name else 0
                trainable_params += [name]

        if self.accelerator.is_main_process:
            logger.info(f"""
-------------------------------------
Model:
    - Total size: {sizes["total"]}
    ---- Lora size: {sizes["total_lora"]}
    - Vision encoder size: {sizes["vision_model"]}
    ---- Lora size: {sizes["vision_model_lora"]}
    - Perceiver resampler size: {sizes["perceiver_resampler"]}
    - Modality projection: {sizes["modality_projection"]}
    - Number of trainable parameters: {sizes["trainable"]}
    ---- Lora trainable parameters: {sizes["trainable_lora"]}
Maximum number of optimizer steps: {self.hparams.max_num_opt_steps}
Maximum number of epochs: {self.hparams.max_num_epochs if self.hparams.max_num_epochs else "N/A"}
Number of gradient accumulation steps: {self.hparams.grad_acc_size}
Number of processes: {self.accelerator.num_processes}
Batch sizes:
    - Per device batch size: {self.hparams.batch_size_per_gpu}
    - Optimizer batch size: {self.optim_param.opt_batch_size}
-------------------------------------
    """)
        logger.info("Trainable/non-trainable parameters:")
        for name, param in vl_model.named_parameters():
            logger.info(f"    Name: {name} | Trainable: {param.requires_grad}")

        if len(self.hparams.train_logging_grad_param_deepspeed) > 0:
            from deepspeed.utils import safe_get_full_fp32_param, safe_get_full_grad, safe_get_full_optimizer_state

            self.safe_get_full_fp32_param = safe_get_full_fp32_param
            self.safe_get_full_grad = safe_get_full_grad
            self.safe_get_full_optimizer_state = safe_get_full_optimizer_state

        self.float_placeholder_tensor = torch.tensor(-1.0, device=self.accelerator.device, dtype=torch.float)
        self.long_placeholder_tensor = torch.tensor(-1, device=self.accelerator.device, dtype=torch.long)

    def setup_batch_size_related_configs(self):
        """
        batch_size-related configs are processed here.

        All this work is done here because it requires knowing the value of num_processes
        """
        hparams = self.hparams

        if hparams.global_batch_size_ramp_up.start is not None:
            # case 1. global batch size ramp up

            # 1a. ramp up constraints
            if (
                hparams.global_batch_size_ramp_up.finish is None
                or hparams.global_batch_size_ramp_up.increment is None
                or hparams.global_batch_size_ramp_up.samples is None
            ):
                raise ValueError(
                    "When using batch size ramp up hparam config entries global_batch_size_ramp_up.start,"
                    " global_batch_size_ramp_up.finish, global_batch_size_ramp_up.increment and"
                    " global_batch_size_ramp_up.samples have to be defined."
                )

            # range checks
            ramp_up_range = hparams.global_batch_size_ramp_up.finish - hparams.global_batch_size_ramp_up.start
            if ramp_up_range < hparams.global_batch_size_ramp_up.increment:
                raise ValueError(
                    f"{hparams.global_batch_size_ramp_up.start=} has to be smaller than"
                    f" {hparams.global_batch_size_ramp_up.finish=}."
                )

            if not (ramp_up_range / hparams.global_batch_size_ramp_up.increment).is_integer():
                raise ValueError(
                    f"({hparams.global_batch_size_ramp_up.finish=} -"
                    f" {hparams.global_batch_size_ramp_up.start=}) /"
                    f" {hparams.global_batch_size_ramp_up.increment=} has to be a whole number"
                )

            if not (
                hparams.global_batch_size_ramp_up.increment
                / (hparams.batch_size_per_gpu * self.accelerator.num_processes)
            ).is_integer():
                raise ValueError(
                    f"{hparams.global_batch_size_ramp_up.increment=} has to be a multiple of"
                    f" {hparams.batch_size_per_gpu * self.accelerator.num_processes=}"
                )

            if self.accelerator.is_main_process:
                logger.info(
                    "Will ramp up global batch size from"
                    f" {hparams.global_batch_size_ramp_up.start} to"
                    f" {hparams.global_batch_size_ramp_up.finish}, in increments of"
                    f" {hparams.global_batch_size_ramp_up.increment} every"
                    f" {hparams.global_batch_size_ramp_up.samples} samples."
                )

            # 1b. hparam.grad_acc_size constraints and derivations
            if hparams.grad_acc_size > 1:
                raise ValueError("When using batch size ramp up hparam.grad_acc_size must be None or 1.")

            hparams.grad_acc_size = hparams.global_batch_size_ramp_up.start / (
                hparams.batch_size_per_gpu * self.accelerator.num_processes
            )
            if not hparams.grad_acc_size.is_integer():
                raise ValueError(
                    f"{hparams.global_batch_size_ramp_up.start=} has to be a multiple of"
                    f" {hparams.batch_size_per_gpu * self.accelerator.num_processes}"
                )
            hparams.grad_acc_size = int(hparams.grad_acc_size)
            logger.info(f"Derived {hparams.grad_acc_size=}")

            # 1c. hparams.global_batch_size constraints and derivation
            if hparams.global_batch_size is not None:
                raise ValueError("When using batch size ramp up hparam.global_batch_size must be None.")
            # in the first run we start at global_batch_size == global_batch_size_ramp_up.start
            hparams.global_batch_size = hparams.global_batch_size_ramp_up.start

        else:
            # case 2. fixed global batch size or fixed grad_acc_size

            # 2a. constraints
            if hparams.grad_acc_size > 1:
                # when global_batch_size is used grad_acc_size will be derived automatically from global_batch_size and n_gpus
                if hparams.global_batch_size is not None and hparams.global_batch_size > 1:
                    raise ValueError("set either hparams.grad_acc_size>1 or hparams.global_batch_size>1, but not both")

            if hparams.global_batch_size is not None and hparams.global_batch_size > 1:
                # 2b. have global_batch_size need to derive grad_acc_size

                hparams.grad_acc_size = hparams.global_batch_size / (
                    hparams.batch_size_per_gpu * self.accelerator.num_processes
                )
                if not hparams.grad_acc_size.is_integer():
                    raise ValueError(
                        f"The derived {hparams.grad_acc_size=} is not an integer,"
                        f" {hparams.global_batch_size=} / ({hparams.batch_size_per_gpu=} *"
                        f" {self.accelerator.num_processes=})"
                    )
                hparams.grad_acc_size = int(hparams.grad_acc_size)
                logger.info(f"Derived {hparams.grad_acc_size=}")
            else:
                # 2c. have grad_acc_size need to derive global_batch_size

                hparams.global_batch_size = (
                    hparams.batch_size_per_gpu * hparams.grad_acc_size * self.accelerator.num_processes
                )
                logger.info(f"Derived {hparams.global_batch_size=}")

    def update_gas_and_gbs(self, grad_acc_size_current, global_batch_size_current):
        """
        Update m4, deepspeed and accelerate with the derived global_batch_size and grad_acc_size
        """
        self.hparams.grad_acc_size = grad_acc_size_current
        self.hparams.global_batch_size = global_batch_size_current
        self.accelerator.gradient_accumulation_steps = grad_acc_size_current
        if is_deepspeed_used():
            get_deepspeed_engine(self.accelerator).set_train_batch_size(global_batch_size_current)

    def _compatibility_checks(self):
        # BF16 requires cuda>=11 and nccl>=2.10.3
        if self.accelerator.mixed_precision == "bf16":
            if version.parse(torch.version.cuda) < version.parse("11.0"):
                raise ValueError(f"mixed precision dtype BF16 requires cuda>=11, but got {torch.version.cuda}")
            if torch.cuda.nccl.version() < (2, 10, 3):
                raise ValueError(
                    f"mixed precision dtype BF16 requires NCCL>=2.10.3, but got {'.'.join(torch.cuda.nccl.version())}"
                )

    def _init_trackers(self):
        if self.hparams.wandb_enable and self.accelerator.is_main_process:
            # Initialize wandb_run_id
            if self.hparams.resume_run and self.hparams.wandb_run_id != "":
                pass
            else:
                if self.hparams.resume_run and not self.hparams.wandb_run_id != "":
                    logger.warning(
                        "** The run you are resuming was not logging into wandb. Therefore a new wandb_run_id has"
                        " been generated **"
                    )
                self.hparams.wandb_run_id = wandb.util.generate_id()
                logger.info(f"** `wandb_run_id`: {self.hparams.wandb_run_id} **")

            # Initialize all trackers
            run_name = self.hparams.save_dir.name
            wdb_config = {}
            for k, v in vars(self.config).items():
                if not hasattr(v, "__dict__"):
                    wdb_config[k] = v
                    continue
                for key, value in vars(v).items():
                    wdb_config[f"{k}-{key}"] = str(value)
            wandb_logger = wandb.init(
                config=wdb_config,
                id=self.hparams.wandb_run_id,
                resume="allow",
                project=self.hparams.wandb_project,
                entity=self.hparams.wandb_entity,
                name=run_name,
                allow_val_change=True,
                tags=self.hparams.wandb_tags,
            )

            self.dummy_module = torch.nn.LayerNorm(1)
            wandb.watch(
                self.dummy_module,
                log="all",
                log_freq=self.hparams.wandb_log_freq * self.hparams.grad_acc_size,
                idx=0,  # To avoid creating a new panel each we un-watch and then re-watch
            )
            tb_run_name = "tb_run_" + self.hparams.wandb_run_id
            tensorboard_tracker = accelerate.tracking.TensorBoardTracker(
                run_name=tb_run_name, logging_dir=self.hparams.save_dir
            )
            logger.info(f"** TensorBoardTracker logging into { self.hparams.save_dir / tb_run_name } **")
            self.accelerator.trackers = [tensorboard_tracker, wandb_logger]

            # Alert WB in replacement of slurm notifications and emails
            wandb.alert(
                title="Training has either started or resumed",
                text=(
                    f"Run name = {run_name}, Jobid = {self.hparams.job_id}, Resuming = {self.hparams.resume_run},"
                    f" Experiment folder = {str(self.hparams.save_dir)}"
                ),
            )

    def _configure_optimizer_and_scheduler(self):
        """defines model optimizer and lr scheduler"""

        vl_optim = getattr(torch_optim, self.optim_param.vl_optim)
        if issubclass(vl_optim, torch_optim.AdamW):
            no_decay = self.optim_param.vl_optim_params.pop("no_decay", [])
            weight_decay = self.optim_param.vl_optim_params.pop("weight_decay", 0.0)
            optim_grouped_params = [
                # Group: tunable parameters with weight decay
                {
                    "params": [
                        p
                        for n, p in self.vl_model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": weight_decay,
                },
                # Group: tunable parameters without weight decay at all
                {
                    "params": [
                        p
                        for n, p in self.vl_model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]

            vl_optim = vl_optim(
                optim_grouped_params,
                **self.optim_param.vl_optim_params,
            )
        else:
            vl_optim = vl_optim(
                self.vl_model.parameters(),
                **self.optim_param.vl_optim_params,
            )

        try:
            vl_scheduler_class = getattr(torch_optim.lr_scheduler, self.optim_param.vl_lr_scheduler)
        except AttributeError:
            vl_scheduler_class = getattr(transformers_optim, self.optim_param.vl_lr_scheduler)
        else:
            raise ValueError(
                f"Could not find {self.optim_param.vl_lr_scheduler} type of LR Scheduler in neither `torch.optim` nor"
                " `transformers.optimization`"
            )
        vl_scheduler = vl_scheduler_class(
            optimizer=vl_optim,
            **self.optim_param.vl_lr_scheduler_params,
        )
        self.vl_optim = vl_optim
        self.vl_scheduler = vl_scheduler

    def _prepare_register(self):
        """
        Prepare model, optimizer and dataloader if necessary.
        Register the scheduler for checkpointing.
        """
        if isinstance(self.train_loader.dataset, torch.utils.data.IterableDataset):
            # `dummy_dataloader`: trick as suggested here: https://discuss.huggingface.co/t/when-using-deepspeed-why-do-i-need-to-pass-dataloaders-to-the-accelerator-prepare/22432/2?u=victorsanh
            # In our IterableDataset, *WE* handle dispatch (instead of `Accelerate`) for each process ourselves as we need
            # better shuffling support.
            #   =>> See `DATA_PROCESSING.md` for more information!
            dummy_dataloader = torch.utils.data.DataLoader(
                [0 for _ in range(20)], batch_size=self.hparams.batch_size_per_gpu
            )

            # important: do note add lr scheduler in `prepare`. We are doing something non canonical
            # with our custom data loaders that leads to not being able to use the standard
            # arguments in the accelerator (typically split_batches which would have ensure the LR
            # was increased every x steps, where x>1 and would be the correct value for grad acc).
            # context: https://github.com/huggingface/m4/pull/386
            self.vl_model, self.vl_optim, dummy_dataloader = self.accelerator.prepare(
                self.vl_model, self.vl_optim, dummy_dataloader
            )
        else:
            (
                self.vl_model,
                self.vl_optim,
                self.train_loader,
                self.val_loader,
            ) = self.accelerator.prepare(self.vl_model, self.vl_optim, self.train_loader, self.val_loader)

        self.accelerator.register_for_checkpointing(self.vl_scheduler)

    def _set_up_training(self):
        """
        Prepare variables for trainings.
        """

        if self.hparams.resume_run:
            # 1. resume
            train_logs = self.resume_param.train_logs

            curr_opt_step = self.resume_param.resume_opt_step
            curr_epoch = self.resume_param.resume_epoch
            gbs_running = self.resume_param.gbs_running

            # This check is necessary because the info is saved as json in the checkpoint
            # and when it is loaded back it is converted to a normal dictionary which can
            # fail downstream in case one of the dataset keys were missing in the saved info
            train_logs = self._check_default_dict_in_train_logs(train_logs)
            self.train_loader.load_state(self.resume_param.opt_step_dir / "resumable_states")

            if self.hparams.load_optimizer_states:
                self.accelerator.load_state(self.resume_param.accelerator_state_dir)
            else:
                # don't load the optimizer states and start with a fresh optimizer
                self.accelerator.load_state(self.resume_param.accelerator_state_dir, load_optimizer_states=False)
                validate_optim_states_are_reset(self)

            self.accelerator.wait_for_everyone()

            opt_step_is_saved = True
            eval_is_done = True

        else:
            # 2. non-resume (first run)
            train_logs = self._reset_train_logs(None)
            curr_opt_step = 0
            curr_epoch = 0
            opt_step_is_saved = False
            eval_is_done = False

            gbs_running = GlobalBatchSizeRampUpRunningParams(
                global_seen_samples=0,
                global_batch_size_current=self.hparams.global_batch_size,
                next_goal_samples=self.hparams.global_batch_size_ramp_up.samples,
                grad_acc_size_current=self.hparams.grad_acc_size,
            )

            self.train_loader.reset_state()

            # rng = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(self.hparams.seed)))
            # self.main_rng_seed = rng.get_state()
            # self.main_rng_seed = rng.RandomState.get_state()

        self.update_gas_and_gbs(gbs_running.grad_acc_size_current, gbs_running.global_batch_size_current)

        max_num_epochs = self.hparams.max_num_epochs
        try:
            num_batches = int(len(self.train_loader) // self.hparams.grad_acc_size)
            max_num_updates = min(self.hparams.max_num_opt_steps, num_batches)
            if max_num_epochs is not None:
                logger.info(
                    "** Setting `max_num_updates` to `max_num_epochs * num_batches` since `max_num_epochs` "
                    "was specified and `max_num_epochs * num_batches` is smaller than `max_num_updates`. **"
                )
                max_num_updates = min(max_num_updates, max_num_epochs * num_batches)
        except TypeError:
            # For iterable datasets len(dataset) is not defined
            max_num_updates = self.hparams.max_num_opt_steps

        if self.hparams.max_num_opt_steps_this_run is not None:
            self.max_num_updates_this_run = min(
                max_num_updates, curr_opt_step + self.hparams.max_num_opt_steps_this_run
            )
        else:
            self.max_num_updates_this_run = max_num_updates

        progress_columns = (
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            "Time Elapsed:",
            TimeElapsedColumn(),
            "Steps Completed",
            MofNCompleteColumn(),
        )
        return (
            progress_columns,
            train_logs,
            max_num_epochs,
            max_num_updates,
            curr_opt_step,
            curr_epoch,
            opt_step_is_saved,
            eval_is_done,
            gbs_running,
        )

    def _do_batch(self, batch, curr_opt_step, dataset_name=None, dataset_idx=None, validation=False):
        # Use effective max_num_images per batch. ie: if the max_num_images of this batch is 3, the pixel_values and image mask are truncated accordingly.
        # Same for max_height and max_width
        effective_max_num_images = max(batch["num_images"])
        if effective_max_num_images > 0:
            images_heights = batch["pixel_attention_mask"][:, :, :, 0].sum(dim=-1)
            images_widths = batch["pixel_attention_mask"][:, :, 0].sum(dim=-1)
            effective_max_height = images_heights.max().int()
            effective_max_width = images_widths.max().int()
            batch["pixel_values"] = batch["pixel_values"][
                :, :effective_max_num_images, :, :effective_max_height, :effective_max_width
            ]
            batch["pixel_attention_mask"] = batch["pixel_attention_mask"][
                :, :effective_max_num_images, :effective_max_height, :effective_max_width
            ]
        else:
            # This case is a security check: if there are no images, then it should not appear in `batch` in the first place
            batch.pop("pixel_values", None)
            batch.pop("pixel_attention_mask", None)

        effective_max_num_tokens = max(batch["attention_mask"].sum(dim=-1))
        batch["input_ids"] = batch["input_ids"][:, :effective_max_num_tokens]
        if "labels" in batch:
            batch["labels"] = batch["labels"][:, :effective_max_num_tokens]
        batch["attention_mask"] = batch["attention_mask"][:, :effective_max_num_tokens]

        batch = accelerate.utils.operations.send_to_device(batch, self.accelerator.device)

        num_images = batch["num_images"].sum()
        num_image_tokens = (batch["input_ids"] == self.image_token_id).sum()
        num_text_tokens = batch["num_text_tokens"].sum()
        total_tokens = batch["attention_mask"].numel()
        num_padding = total_tokens - batch["attention_mask"].sum()
        if "pixel_values" in batch:
            pixel_values_sum = batch["pixel_values"].sum()
        else:
            pixel_values_sum = torch.tensor(0.0, device=self.accelerator.device)
        image_to_text_ratio = torch.div(num_images, num_text_tokens)

        vl_output = self.vl_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"] if "pixel_values" in batch else None,
            pixel_attention_mask=batch["pixel_attention_mask"] if "pixel_attention_mask" in batch else None,
            labels=batch["labels"] if "labels" in batch else batch["input_ids"],
        )
        per_token_loss = vl_output.loss

        if validation:
            return (
                per_token_loss,
                num_images,
                num_text_tokens,
                image_to_text_ratio,
                num_padding,
                pixel_values_sum,
            )
        else:
            if self.hparams.loss_weights_per_dataset is not None:
                per_token_loss *= self.hparams.loss_weights_per_dataset[dataset_idx]
            if self.optim_param.z_loss > 0.0:
                logits = vl_output.logits
                attention_mask = batch["attention_mask"] * (1 - (batch["input_ids"] == self.image_token_id).long())
                log_z = torch.logsumexp(logits, dim=-1) * attention_mask
                z_loss = log_z**2
                z_loss = z_loss.sum() / attention_mask.sum()
                combined_loss = per_token_loss + self.optim_param.z_loss * z_loss
            else:
                z_loss = torch.tensor(0.0, device=self.accelerator.device)
                combined_loss = per_token_loss
            sync_gradients = self.accelerator.sync_gradients
            deepspeed = hasattr(self.accelerator, "deepspeed_engine_wrapped")

            # accelerate's deepspeed `backward` calls `engine.step`, which is a problem if we want
            # to investigate things before step, so override with just a backward call and then call
            # `engine.step` along with `optim.step` a bit lower
            if deepspeed:
                self.accelerator.deepspeed_engine_wrapped.engine.backward(combined_loss)
            else:
                self.accelerator.backward(combined_loss)

            if sync_gradients:
                self.accelerator.clip_grad_norm_(self.vl_model.parameters(), self.hparams.grad_clip)

                if (
                    len(self.hparams.train_logging_grad_param_deepspeed) != 0
                    and (curr_opt_step + 1) % self.hparams.train_logging_grad_param_deepspeed_opt_steps == 0
                ):
                    self._log_deepspeed_training_stats(curr_opt_step=curr_opt_step)

            if deepspeed:
                self.accelerator.deepspeed_engine_wrapped.engine.step()

            self.vl_optim.step()

            # 1. sync_gradients is used for this dirty trick: https://github.com/huggingface/m4/pull/386
            # 2. since we don't accelerate prepare the lr scheduler we need to manually skip it if
            # optimizer skipped (otherwise accelerate would do that for us)
            if sync_gradients and not self.accelerator.optimizer_step_was_skipped:
                self.vl_scheduler.step()

            self.vl_optim.zero_grad(set_to_none=True)
            tflops_per_batch_per_gpu = self.vl_model.get_model_tflops_per_batch_per_gpu(
                hparams=self.hparams,
                data_param=getattr(self.data_param, dataset_name),
                tokenizer=self.tokenizer,
                max_num_images=effective_max_num_images,
                max_num_tokens=effective_max_num_tokens,
            ).to(self.accelerator.device)

        # Reset batch
        return (
            per_token_loss,
            z_loss,
            num_images,
            num_image_tokens,
            num_text_tokens,
            image_to_text_ratio,
            num_padding,
            pixel_values_sum,
            tflops_per_batch_per_gpu,
        )

    def _log_deepspeed_training_stats(self, curr_opt_step):
        if self.hparams.job_id is not None:
            log_stats_file = self.hparams.save_dir / "logs" / f"{self.hparams.job_id}_logs_params_grads_stats.jsonl"
        else:
            log_stats_file = self.hparams.save_dir / "logs" / "logs_params_grads_stats.jsonl"

        beta1 = self.optim_param.vl_optim_params["betas"][0]
        beta2 = self.optim_param.vl_optim_params["betas"][1]
        eps = self.optim_param.vl_optim_params.get("eps", 1e-8)

        step = self.vl_optim.optimizer.state[list(self.vl_optim.optimizer.state.keys())[0]]["step"]

        bias_correction_1 = 1 / (1 - beta1**step)
        bias_correction_2 = 1 / (1 - beta2**step)

        for n, lp in self.vl_model.named_parameters():
            self.accelerator.wait_for_everyone()
            if not lp.requires_grad:
                continue

            hp = self.safe_get_full_fp32_param(lp)
            exp_avg = self.safe_get_full_optimizer_state(lp, "exp_avg")
            exp_avg_sq = self.safe_get_full_optimizer_state(lp, "exp_avg_sq")
            hp_grad = self.safe_get_full_grad(lp)

            if not self.accelerator.is_main_process:
                continue

            if exp_avg_sq is not None and exp_avg is not None:
                effective_update = exp_avg * bias_correction_1 / (torch.sqrt(exp_avg_sq * bias_correction_2) + eps)
            else:
                effective_update = None

            grad_param_logs = {
                "name": n,
                "step": step.item(),
                **get_stats(hp, "hp"),
                **get_stats(exp_avg, "exp_avg"),
                **get_stats(exp_avg_sq, "exp_avg_sq"),
                **get_stats(hp_grad, "hp_grad"),
                **get_stats(effective_update, "effective_update"),
            }
            grad_param_format = {
                "step": "",
                "name": "",
                **get_stats_format("hp"),
                **get_stats_format("exp_avg"),
                **get_stats_format("exp_avg_sq"),
                **get_stats_format("hp_grad"),
                **get_stats_format("effective_update"),
            }

            if LoggingTypes.JSONL in self.hparams.train_logging_grad_param_deepspeed:
                with open(log_stats_file, "a") as f:
                    f.write(json.dumps(grad_param_logs) + "\n")

            if LoggingTypes.WANDB in self.hparams.train_logging_grad_param_deepspeed and self.hparams.wandb_enable:
                self.accelerator.log({**grad_param_logs, **self._get_additional_step_logs()}, step=curr_opt_step + 1)

            if LoggingTypes.PRINT in self.hparams.train_logging_grad_param_deepspeed:
                log = "Grads and params stats: "
                log += self.format_print_logs(grad_param_logs, grad_param_format)
                print(log)

    def gather_metrics(
        self,
        local_metric_list: List[Dict[str, torch.Tensor]],
        placeholder_tensor: torch.Tensor,
        reduce_op_list,
        ds_name_suffix: str,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Collating all metrics to gather into ONE call to `torch.distributed.all_gather` instead of doing one per metric x dataset_name.
        """
        if self.accelerator.num_processes == 1:
            for local_metric in local_metric_list:
                for ds_name, tensor in local_metric.items():
                    if tensor is not None:
                        local_metric[ds_name] = tensor.item()
            return local_metric_list

        dataset_names = sorted([f"{e.value}{ds_name_suffix}" for e in DatasetNames] + ["all"])

        for local_metric in local_metric_list:
            for ds_name in dataset_names:
                if local_metric[ds_name] is None:
                    local_metric[ds_name] = placeholder_tensor.clone()

        collated_local_metrics = torch.stack(
            [torch.stack([local_metric[ds_name] for ds_name in dataset_names]) for local_metric in local_metric_list]
        )  # Size len(local_metric_list) x len(dataset_names)
        broadcasted_placeholder_tensor = torch.stack(
            [torch.stack([placeholder_tensor for _ in dataset_names]) for _ in local_metric_list]
        )  # Size len(local_metric_list) x len(dataset_names)

        output_objects = [broadcasted_placeholder_tensor.clone() for _ in range(self.accelerator.num_processes)]
        torch.distributed.all_gather(output_objects, collated_local_metrics)
        gathered_metrics = torch.stack(
            output_objects
        )  # Size num_processes x len(local_metric_list) x len(dataset_names)

        gathered_metric_list = []
        for metric_idx, (_, reduce_op) in enumerate(zip(local_metric_list, reduce_op_list)):
            result = {}
            for ds_idx, ds_name in enumerate(dataset_names):
                metrics = gathered_metrics[:, metric_idx, ds_idx]
                metrics = metrics[metrics != placeholder_tensor]
                if metrics.numel() == 0:
                    result[ds_name] = None
                else:
                    result[ds_name] = reduce_op(metrics).item()
            gathered_metric_list.append(result)

        return gathered_metric_list

    def _update_logs(
        self,
        curr_opt_step,
        curr_epoch,
        global_batch_size_current,
        train_logs,
        per_token_loss,
        z_loss,
        num_tokens,
        num_images,
        num_image_tokens,
        image_to_text_ratio,
        num_padding,
        fwd_bwd_time,
        pixel_values,
        tflops_per_batch_per_gpu,
        total_energy_delta_per_gpu,
        dataset_name,
        ds_name_suffix="",
    ):
        def init_dict():
            return {f"{e.value}{ds_name_suffix}": None for e in DatasetNames}

        local_per_token_loss = init_dict()
        local_z_loss = init_dict()
        local_num_tokens = init_dict()
        local_num_images = init_dict()
        local_num_image_tokens = init_dict()
        local_image_to_text_ratio = init_dict()
        local_fwd_bwd_time = init_dict()
        local_pixel_values = init_dict()
        local_tflops_per_batch_per_gpu = init_dict()
        local_total_energy_delta_per_gpu = init_dict()
        local_num_padding = init_dict()
        local_num_batches = init_dict()

        for key_name in [f"{dataset_name}{ds_name_suffix}", "all"]:
            local_per_token_loss[key_name] = per_token_loss
            local_z_loss[key_name] = z_loss
            local_num_tokens[key_name] = num_tokens
            local_num_images[key_name] = num_images
            local_num_image_tokens[key_name] = num_image_tokens
            local_image_to_text_ratio[key_name] = image_to_text_ratio
            local_fwd_bwd_time[key_name] = fwd_bwd_time
            local_pixel_values[key_name] = pixel_values
            local_tflops_per_batch_per_gpu[key_name] = tflops_per_batch_per_gpu
            local_total_energy_delta_per_gpu[key_name] = total_energy_delta_per_gpu
            local_num_padding[key_name] = num_padding
            local_num_batches[key_name] = torch.tensor(1.0, device=self.accelerator.device, dtype=torch.long)

        [
            gathered_per_token_loss,
            gathered_z_loss,
            gathered_image_to_text_ratio,
            gathered_fwd_bwd_time,
            gathered_pixel_values,
            gathered_tflops_per_batch_per_gpu,
            gathered_total_energy_delta_per_gpu,
        ] = self.gather_metrics(
            local_metric_list=[
                local_per_token_loss,
                local_z_loss,
                local_image_to_text_ratio,
                local_fwd_bwd_time,
                local_pixel_values,
                local_tflops_per_batch_per_gpu,
                local_total_energy_delta_per_gpu,
            ],
            reduce_op_list=[torch.sum, torch.sum, torch.mean, torch.sum, torch.sum, torch.sum, torch.sum],
            placeholder_tensor=self.float_placeholder_tensor,
            ds_name_suffix=ds_name_suffix,
        )

        [
            gathered_num_padding,
            gathered_num_tokens,
            gathered_num_batches,
            gathered_num_images,
            gathered_num_image_tokens,
        ] = self.gather_metrics(
            local_metric_list=[
                local_num_padding,
                local_num_tokens,
                local_num_batches,
                local_num_images,
                local_num_image_tokens,
            ],
            reduce_op_list=[torch.sum, torch.sum, torch.sum, torch.sum, torch.sum],
            placeholder_tensor=self.long_placeholder_tensor,
            ds_name_suffix=ds_name_suffix,
        )

        for ds_name in local_per_token_loss.keys():
            for metric_name, new_value in [
                ("per_token_loss_acc", gathered_per_token_loss[ds_name]),
                ("z_loss_acc", gathered_z_loss[ds_name]),
                ("num_images", gathered_num_images[ds_name]),
                ("num_image_tokens", gathered_num_image_tokens[ds_name]),
                ("num_tokens", gathered_num_tokens[ds_name]),
                ("num_padding", gathered_num_padding[ds_name]),
                ("pixel_values_sum", gathered_pixel_values[ds_name]),
                ("tflop_counter_since_training_logged", gathered_tflops_per_batch_per_gpu[ds_name]),
                ("fwd_bwd_time_since_training_logged", gathered_fwd_bwd_time[ds_name]),
                ("total_energy_delta_since_training_logged", gathered_total_energy_delta_per_gpu[ds_name]),
                ("fwd_bwd_time", gathered_fwd_bwd_time[ds_name]),
                ("tflop_counter", gathered_tflops_per_batch_per_gpu[ds_name]),
                ("num_per_device_batches_since_training_logged", gathered_num_batches[ds_name]),
                ("num_per_device_batches", gathered_num_batches[ds_name]),
                ("num_per_device_batches_in_curr_epoch", gathered_num_batches[ds_name]),
            ]:
                if new_value is None:
                    continue

                if train_logs[metric_name][ds_name] is None:
                    train_logs[metric_name][ds_name] = new_value
                else:
                    train_logs[metric_name][ds_name] += new_value

            if gathered_image_to_text_ratio[ds_name] is not None:
                train_logs["image_to_text_ratio"][ds_name] = gathered_image_to_text_ratio[ds_name]

            if gathered_fwd_bwd_time[ds_name] is not None:
                train_logs["tflops_acc"][ds_name] = (
                    train_logs["tflop_counter"][ds_name] / train_logs["fwd_bwd_time"][ds_name]
                )

        train_logs["num_batches_since_training_logged"]["all"] += 1
        train_logs["num_batches"]["all"] += 1
        train_logs["num_batches_in_curr_epoch"]["all"] += 1

        train_logs["lr"] = self.vl_scheduler.get_last_lr()[0]

        train_logs["num_opt_steps"] = curr_opt_step
        train_logs["num_epochs"] = curr_epoch
        train_logs["global_batch_size_current"] = global_batch_size_current

        return train_logs

    def _update_datasets_states(self, dataset_idx, dataset_state):
        # TODO: This step will go away in future PRs. The dataloader already knows the state when it
        # sends it to the trainer. There is no need to send it to trainer and send it back. Let's
        # simplify this as well in the future
        self.train_loader.update_state(dataset_idx, dataset_state)

    def _get_additional_step_logs(self):
        if self.config.hparams.job_id is not None:
            return {"job_id": self.config.hparams.job_id, "commit": self.config.hparams.repo_commit_id}
        else:
            return {"commit": self.config.hparams.repo_commit_id}

    def format_print_logs(self, dict_logs, keys_known_formats, skip_keys=[]):
        """
        compact formatting of the logs with pre-specified formatter for each log entry, plus a
        catch-all if new log entries are added but forgotten to be added in keys_known_formats

        the keys order is the one that controls how the logs are printed (py37+).
        even if there is no formatter there is still an empty value entry here as it tells use the order of keys.
        """
        log = ""
        for key in keys_known_formats.keys():
            if key in dict_logs:
                if isinstance(dict_logs[key], dict):
                    for sub_key in dict_logs[key].keys():
                        prefix = f"{key}"
                        if sub_key != "all":
                            if LoggingTypes.PRINT not in self.hparams.train_logging_per_dataset_info:
                                continue
                            prefix += f"/{sub_key}"
                        log += f"{prefix}: {dict_logs[key][sub_key]:{keys_known_formats[key]}} | "
                else:
                    log += f"{key}: {dict_logs[key]:{keys_known_formats[key]}} | "

        # in case some new log entries were added that don't yet have the formatter string we dump them as is
        for key in set(dict_logs.keys() - set(skip_keys) - set(keys_known_formats.keys())):
            if key in dict_logs:
                log += f"{key}: {dict_logs[key]} | "

        return log

    def format_jsonl_logs(self, dict_logs):
        """
        Similar to format_print_logs but for jsonl logs
        """
        log = {}
        for key in dict_logs:
            # We don't want to log the accumulated values
            if "_acc" in key:
                continue
            elif isinstance(dict_logs[key], dict):
                for sub_key in dict_logs[key].keys():
                    prefix = f"{key}"
                    if sub_key != "all":
                        if LoggingTypes.JSONL not in self.hparams.train_logging_per_dataset_info:
                            continue
                        prefix += f"/{sub_key}"
                    log[prefix] = dict_logs[key][sub_key]
            else:
                log[key] = dict_logs[key]

        return log

    def format_val_logs(self, val_logs, logger_type=LoggingTypes.PRINT):
        keys_known_formats = {
            "val_per_token_loss": ".4f",
            "val_num_images": "",
            "val_num_tokens": "",
            "val_num_padding": "",
            "val_image_to_text_ratio": ".4f",
        }
        if logger_type == LoggingTypes.PRINT:
            return self.format_print_logs(val_logs, keys_known_formats)
        elif logger_type == LoggingTypes.JSONL:
            return self.format_jsonl_logs(val_logs)
        else:
            raise ValueError(f"Unknown logger type: {logger_type}")

    def format_train_logs(self, train_logs, logger_type=LoggingTypes.PRINT) -> Union[str, Dict]:
        keys_known_formats = {
            "per_token_loss": ".4f",
            "lr": ".3E",
            "global_batch_size": "",
            "num_tokens": "",
            "num_images": "",
            "num_image_tokens": "",
            "num_padding": "",
            "fwd_bwd_time": ".1f",
            "image_to_text_ratio": ".4f",
            "num_batches": "",
            "num_batches_in_curr_epoch": "",
            "num_batches_since_training_logged": "",
            "num_per_device_batches": "",
            "num_per_device_batches_in_curr_epoch": "",
            "num_per_device_batches_since_training_logged": "",
            "tflops": ".1f",
            "watt/s": ".1f",
            "fwd_bwd_time_since_training_logged": ".1f",
            "num_epochs": "",
            "num_opt_steps": "",
            "z_loss": ".4f",
            "pixel_values_sum": ".5E",
            "tflop_counter": ".3E",
            "tflops_acc": ".1f",
        }

        # intermediary state accumulate keys
        skip_keys = [
            "per_token_loss_acc",
            "z_loss_acc",
            "tflop_counter_since_training_logged",
            "num_batches_since_training_logged",
            "num_per_device_batches_since_training_logged",
            "total_energy_delta_since_training_logged",
        ]

        if logger_type == LoggingTypes.PRINT:
            return self.format_print_logs(train_logs, keys_known_formats, skip_keys)
        elif logger_type == LoggingTypes.JSONL:
            return self.format_jsonl_logs(train_logs)
        else:
            raise ValueError(f"Unknown logger type: {logger_type}")

    def _log_training(self, curr_opt_step, train_task, train_logs):
        for key in train_logs["per_token_loss_acc"].keys():
            if train_logs["num_per_device_batches_since_training_logged"][key] is not None:
                train_logs["per_token_loss"][key] = (
                    train_logs["per_token_loss_acc"][key]
                    / train_logs["num_per_device_batches_since_training_logged"][key]
                )
                train_logs["z_loss"][key] = (
                    train_logs["z_loss_acc"][key] / train_logs["num_per_device_batches_since_training_logged"][key]
                )
            else:
                train_logs["per_token_loss"][key] = None
                train_logs["z_loss"][key] = None

            if train_logs["fwd_bwd_time_since_training_logged"][key] is not None:
                train_logs["tflops"][key] = (
                    train_logs["tflop_counter_since_training_logged"][key]
                    / train_logs["fwd_bwd_time_since_training_logged"][key]
                )
                train_logs["watt/s"][key] = (
                    train_logs["total_energy_delta_since_training_logged"][key]
                    / train_logs["fwd_bwd_time_since_training_logged"][key]
                )
            else:
                train_logs["tflops"][key] = None
                train_logs["watt/s"][key] = None

        if self.accelerator.is_main_process:
            print_log = ""
            progress = f"{str(MofNCompleteColumn().render(train_task)):>12} {TaskProgressColumn().render(train_task)}"
            print_log += f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] iteration: {progress} | "
            elapsed_time = TimeElapsedColumn().render(train_task)
            print_log += f"elapsed time: {elapsed_time} | "

            print_log += self.format_train_logs(train_logs, logger_type=LoggingTypes.PRINT)

            # TODO: Allow mem usage to be logged according to LogginTypes passed in hparams
            if self.hparams.train_log_mem_usage:
                print_log += mem_usage_formatted(LoggingTypes.PRINT)

            print(print_log)

            jsonl_logs = {
                "iteration": progress.strip(),
                "elapsed_time": str(elapsed_time),
                "set": "train",
            }
            jsonl_logs.update(self.format_train_logs(train_logs, logger_type=LoggingTypes.JSONL))
            if self.hparams.train_log_mem_usage:
                jsonl_logs.update(mem_usage_formatted(LoggingTypes.JSONL))

            if self.hparams.job_id is not None:
                log_jsonl_file = self.hparams.save_dir / "logs" / f"{self.hparams.job_id}_logs.jsonl"
            else:
                log_jsonl_file = self.hparams.save_dir / "logs" / "logs.jsonl"

            log_jsonl_file.parent.mkdir(parents=True, exist_ok=True)

            with open(log_jsonl_file, "a") as f:
                f.write(json.dumps(jsonl_logs) + "\n")

            if self.hparams.wandb_enable:
                filtered_train_logs = train_logs
                if LoggingTypes.WANDB not in self.hparams.train_logging_per_dataset_info:
                    filtered_train_logs = {}
                    for key in train_logs.keys():
                        if isinstance(train_logs[key], dict):
                            filtered_train_logs[key] = train_logs[key]["all"]
                        else:
                            filtered_train_logs[key] = train_logs[key]
                # remove nested None values as wandb doesn't support them
                filtered_train_logs = {k: v for k, v in filtered_train_logs.items() if v is not None}
                for k, v in filtered_train_logs.items():
                    if isinstance(v, dict):
                        filtered_train_logs[k] = {k2: v2 for k2, v2 in v.items() if v2 is not None}
                self.accelerator.log({**filtered_train_logs, **self._get_additional_step_logs()}, step=curr_opt_step)

        train_logs = self._reset_train_logs(train_logs)
        return train_logs

    def _log_activations(self, curr_opt_step):
        if not self.activation_tracker.jsonl_stats:
            return

        if LoggingTypes.JSONL in self.hparams.train_logging_activations:
            if self.hparams.job_id is not None:
                log_activations_filename = (
                    self.hparams.save_dir / "logs" / f"{self.hparams.job_id}_logs_activations.jsonl"
                )
            else:
                log_activations_filename = self.hparams.save_dir / "logs" / "logs_activations.jsonl"

            self.activation_tracker.dump_stats(log_activations_filename, curr_opt_step=curr_opt_step)

        # if LoggingTypes.WANDB in self.hparams.train_logging_activations and self.hparams.wandb_enable:
        #     for stats in self.activation_tracker.jsonl_stats:
        #         self.accelerator.log({**stats, **self._get_additional_step_logs()}, step=curr_opt_step)

        if LoggingTypes.PRINT in self.hparams.train_logging_activations:
            for stats in self.activation_tracker.jsonl_stats:
                stats["step"] = curr_opt_step
                activation_format = {
                    k: "" if ("nonzero" in k or "step" in k or "name" in k or "type" in k or "batches" in k) else "e"
                    for k in stats.keys()
                }
                log = "Activation stats: "
                log += self.format_print_logs(stats, activation_format)
                print(log)

        self.activation_tracker.reset_jsonl_stats()

    def _check_kill_switch(self):
        if self.hparams.kill_switch_path is not None and self.hparams.kill_switch_path.exists():
            self.kill_switch_activated = True

    def _check_jz_time_and_memory(self, curr_opt_step):
        # From https://github.com/wandb/wandb/blob/9c777265f8cea1eaeb0407dd37ab889aeea81114/wandb/sdk/internal/stats.py#L263
        if self.accelerator.is_local_main_process:
            self.memory_value = torch.tensor(psutil.virtual_memory().percent).to(self.accelerator.device)
        else:
            self.memory_value = torch.tensor(0.0).to(self.accelerator.device)
        self.accelerator.wait_for_everyone()
        memory_value_max = self.accelerator.gather(self.memory_value)
        memory_value_max = memory_value_max.max().item()
        self.memory_explosion = memory_value_max >= _MEMORY_EXPLOSION_THRESHOLD

        if self.hparams.jz_job_time_sec is not None:
            if self.accelerator.is_main_process:
                overall_time = time.time() - self.hparams.jz_start_time
                self.jz_training_time_over[0] = overall_time >= self.hparams.jz_job_time_sec
            self.accelerator.wait_for_everyone()
            accelerate.utils.broadcast_object_list(self.jz_training_time_over)

        if self.accelerator.is_main_process and self.hparams.wandb_enable:
            system_metrics_logs = self._get_system_metrics_logs(memory_value_max)
            self.accelerator.log({**system_metrics_logs, **self._get_additional_step_logs()}, step=curr_opt_step)

    def _check_sigterm_signal(self):
        if self.sigterm_listener.kill_now:
            self.sigterm_signal_received = True

    def _save(
        self,
        train_logs,
        curr_opt_step,
        curr_epoch,
        gbs_running,
    ):
        self.accelerator.wait_for_everyone()

        # create directory and file names
        self.last_opt_step_dir = self.hparams.save_dir / f"opt_step-{curr_opt_step}"
        # Make directory for this step
        self.last_opt_step_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader.save_state(self.last_opt_step_dir / "resumable_states")
        # XXX: why is there a hardcoded accelerator_state path? should be coming from config, no?
        self.accelerator.save_state(self.last_opt_step_dir / "accelerator_state")

        if self.accelerator.is_main_process:
            # Save model and accelerator state
            unwrapped_model = self.accelerator.unwrap_model(self.vl_model)

            # fix the model class name to be of VLOOOM type the first time it's saved
            unwrapped_model.config.architectures = [unwrapped_model.__class__.__name__]

            # deepspeed doesn't need the overhead of gathering the model from all gpus
            if not is_deepspeed_zero3_used():
                if self.hparams.use_lora:
                    unwrapped_model.save_pretrained(self.last_opt_step_dir / "unwrapped_adapter")
                    # Manual unloading with a simple PeftMixin to avoid having to deal with PeftModel state dict
                    base_model = lora_unload(copy.deepcopy(unwrapped_model))
                    # Save pretrained with _hf_peft_config_loaded=True will save the adapters only. So we set it manually to False
                    base_model._hf_peft_config_loaded = False
                    base_model.save_pretrained(self.last_opt_step_dir / "unwrapped_model")
                    del base_model
                else:
                    unwrapped_model.save_pretrained(self.last_opt_step_dir / "unwrapped_model")
            else:
                # For deepspeed, `save_checkpoint` done by accelerate takes care of saving the model in a
                # special format per gpu which on resume will be used to load the model - so we don't need to
                # save pytorch state_dict separately, which can be costly or impossible if there not enough CPU RAM.
                # We only need to save the config
                unwrapped_model.config.save_pretrained(
                    self.last_opt_step_dir / "unwrapped_model",
                )
                if self.hparams.use_lora:
                    unwrapped_model.peft_config["default"].save_pretrained(
                        self.last_opt_step_dir / "unwrapped_adapter",
                    )

            # Save tokenizer directly into the same dir
            self.tokenizer.save_pretrained(
                self.last_opt_step_dir / "tokenizer",
            )

            # infos to resume run at this step
            data = {
                "train_logs": train_logs,
                "wandb_run_id": self.hparams.wandb_run_id,
                "seed": self.hparams.seed,
                "resume_opt_step": curr_opt_step,
                "resume_epoch": curr_epoch,
                "gbs_running": gbs_running,
            }

            with open(self.last_opt_step_dir / "resume_run_infos.json", "w") as fp:
                # json.dump(data, fp, indent=2)
                json.dump(data, fp, indent=2, cls=JSONEncoderForDataclasses)

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # mark this checkpoint as finished - needed for async slurm jobs like s3 uploader
            latest_path_finished = self.last_opt_step_dir / "finished-saving"
            latest_path_finished.touch()

            # mark which is latest saved checkpoint for correct resume
            latest_path = self.hparams.save_dir / "latest_opt_step_dir"
            with open(latest_path, "w") as fd:
                fd.write(str(self.last_opt_step_dir))
            logger.info(f"** Saving finished at `{self.last_opt_step_dir}` **")

        if self.accelerator.is_main_process and self.hparams.upload_to_s3:
            # We keep around the last checkpoint (which was saved just above) locally, and delete the previous to last one.
            locally_present_saved_steps_inds = [
                int(os.path.split(dir)[-1].split("opt_step-")[-1])
                for dir in self.hparams.save_dir.iterdir()
                if (dir.is_dir() and "opt_step" in str(dir))
            ]
            if len(locally_present_saved_steps_inds) >= 2:
                previous_to_last_saved_step = sorted(locally_present_saved_steps_inds)[-2]
                previous_to_last_folder = f"opt_step-{previous_to_last_saved_step}"
            else:
                previous_to_last_folder = ""

            # Subprocess command is inspired from https://stackoverflow.com/questions/5772873/python-spawn-off-a-child-subprocess-detach-and-exit/64145368#64145368
            # `stdout` and `stderr` are supressed to avoid polluting the logs with the output of the command.
            cmd = (
                str(Path(__file__).resolve().parents[2]) + "/experiments/pretraining/vloom/common/sync_and_upload.sh",
                os.path.split(self.hparams.save_dir)[0],
                os.path.split(self.hparams.save_dir)[1],
                previous_to_last_folder,
                "opt_step-" + str(curr_opt_step),
            )
            subprocess.Popen(cmd, start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    def _save_batch(self, batch, curr_idx):
        dir_path = self.hparams.save_dir / "batches"
        dir_path.mkdir(parents=True, exist_ok=True)

        with open(dir_path / f"batch_idx_{curr_idx}_proc_{self.accelerator.process_index}.pkl", "wb") as file:
            pickle.dump(batch, file)

    def _check_if_training_is_over(self, curr_opt_step, max_num_updates):
        self._check_kill_switch()
        self._check_jz_time_and_memory(curr_opt_step=curr_opt_step)
        self._check_sigterm_signal()

        finished_training = True

        if self.kill_switch_activated:
            logger.info(f"** Kill switch activated (Don't forget to remove {self.hparams.kill_switch_path}) **")
        elif self.sigterm_signal_received:
            logger.info("** SIGTERM signal received. Please restart training **")
        elif self.jz_training_time_over[0]:
            logger.info("** Training time is over **")
        elif self.memory_explosion:
            logger.info("** CPU memory is close to explosion. Please restart training **")
        elif curr_opt_step >= max_num_updates:
            logger.info("** Maximum number of steps has been reached for this training **")
        elif curr_opt_step >= self.max_num_updates_this_run:
            logger.info("** Maximum number of steps has been reached for this run **")
        else:
            finished_training = False

        return finished_training

    def _check_if_training_is_over_and_maybe_save_model(
        self,
        curr_opt_step,
        curr_epoch,
        gbs_running,
        max_num_updates,
        train_logs,
        opt_step_is_saved,
    ):
        # check if should finish the training
        finished_training = self._check_if_training_is_over(curr_opt_step, max_num_updates)

        # save the model
        # 1. either because it's a scheduled saving
        is_opt_to_save_model = curr_opt_step != 0 and curr_opt_step % self.hparams.train_saving_opt_steps == 0
        # 2. it was requested via the save switch
        if self.hparams.save_switch_path is not None and self.hparams.save_switch_path.exists():
            is_opt_to_save_model = True
            logger.info("** Save switch activated - forcing checkpoint save **")
            self.hparams.save_switch_path.unlink()

        if not opt_step_is_saved and (finished_training or is_opt_to_save_model):
            self._save(
                train_logs,
                curr_opt_step,
                curr_epoch,
                gbs_running,
            )
            opt_step_is_saved = True

        return finished_training, opt_step_is_saved

    def _reset_train_logs(self, train_logs):
        if train_logs is None:
            train_logs = {}

            for metric_name, default_value_fn in METRICS_TO_DEFAULT_VALUE_FN.items():
                if metric_name in METRICS_TO_RESET_AFTER_LOGGING:
                    continue
                train_logs[metric_name] = default_value_fn()

        # reset counters that get zeroed after every log event
        train_logs.update(
            {metric_name: METRICS_TO_DEFAULT_VALUE_FN[metric_name]() for metric_name in METRICS_TO_RESET_AFTER_LOGGING}
        )
        return train_logs

    def _check_default_dict_in_train_logs(self, train_logs):
        for metric_name, default_value_fn in METRICS_TO_DEFAULT_VALUE_FN.items():
            value = default_value_fn()
            # if the metric was initialized as a defaultdict, we need to convert it again to a defaultdict
            if not isinstance(value, defaultdict):
                continue
            value.update(train_logs.get(metric_name, {}))
            train_logs[metric_name] = value
        return train_logs

    def _end_of_epoch_reset_train_logs(self, train_logs):
        if train_logs is None:
            raise ValueError("`train_logs` should not be `None` at the end of an epoch.")
        train_logs.update(
            {
                metric_name: METRICS_TO_DEFAULT_VALUE_FN[metric_name]()
                for metric_name in ["num_batches_in_curr_epoch", "num_per_device_batches_in_curr_epoch"]
            }
        )
        return train_logs

    def _do_validation(self, progress, curr_opt_step):
        try:
            val_len = len(self.val_loader)
        except TypeError:
            val_len = None
        self.vl_model.eval()
        val_loop = progress.add_task(
            f"[cyan]Validation step-{curr_opt_step}",
            disable=not self.accelerator.is_main_process,
            total=val_len,
            visible=False,
        )

        curr_val_task = progress.tasks[-1]
        starter_dict = {e.value: 0 for e in DatasetNames}
        starter_dict["all"] = 0
        val_per_token_loss_acc = copy.deepcopy(starter_dict)
        val_steps = copy.deepcopy(starter_dict)
        val_num_images = copy.deepcopy(starter_dict)
        val_num_tokens = copy.deepcopy(starter_dict)
        val_num_padding = copy.deepcopy(starter_dict)
        val_image_to_text_ratio = {e.value: [] for e in DatasetNames}
        val_image_to_text_ratio["all"] = []

        for _, dataset_name, _, batch in self.val_loader:
            with torch.no_grad():
                (
                    curr_val_per_token_loss,
                    curr_val_num_images,
                    curr_val_num_tokens,
                    curr_val_image_to_text_ratio,
                    curr_val_num_padding,
                    _,
                ) = self._do_batch(batch, curr_opt_step, validation=True)

            val_per_token_loss_acc["all"] += curr_val_per_token_loss
            val_num_images["all"] += curr_val_num_images
            val_num_tokens["all"] += curr_val_num_tokens
            val_num_padding["all"] += curr_val_num_padding
            val_image_to_text_ratio["all"].append(curr_val_image_to_text_ratio)
            val_steps["all"] += 1

            val_per_token_loss_acc[dataset_name] += curr_val_per_token_loss
            val_num_images[dataset_name] += curr_val_num_images
            val_num_tokens[dataset_name] += curr_val_num_tokens
            val_num_padding[dataset_name] += curr_val_num_padding
            val_image_to_text_ratio[dataset_name].append(curr_val_image_to_text_ratio)
            val_steps[dataset_name] += 1

            progress.update(val_loop, advance=1)
            if (
                curr_val_task.completed % self.hparams.val_inline_logging_opt_steps == 0
                and self.accelerator.is_main_process
            ):
                logger.info(
                    "Validation"
                    f" step-{curr_opt_step} state:{TaskProgressColumn().render(curr_val_task)} Time"
                    f" Elapsed: {TimeElapsedColumn().render(curr_val_task)} Steps"
                    f" Completed:{MofNCompleteColumn().render(curr_val_task)}"
                )
        self.vl_model.train()

        return (
            val_steps,
            val_per_token_loss_acc,
            val_num_images,
            val_num_tokens,
            val_image_to_text_ratio,
            val_num_padding,
        )

    def _log_validation(
        self,
        val_steps,
        curr_opt_step,
        val_per_token_loss_acc,
        val_num_images,
        val_num_tokens,
        val_image_to_text_ratio,
        val_num_padding,
    ):
        def convert_to_tensor(x):
            if not torch.is_tensor(x):
                return torch.tensor(x, device=self.accelerator.device)
            return x

        for key, value in val_image_to_text_ratio.items():
            if len(value) != 0:
                val_image_to_text_ratio[key] = sum(value) / len(value)
            else:
                val_image_to_text_ratio[key] = 0.0

        gathered_val_per_token_loss = {}
        gathered_val_num_images = {}
        gathered_val_num_tokens = {}
        gathered_val_num_padding = {}
        gathered_val_image_to_text_ratio = {}
        gathered_val_steps = {}

        for key in val_image_to_text_ratio.keys():
            (
                gathered_val_per_token_loss[key],
                gathered_val_num_images[key],
                gathered_val_num_tokens[key],
                gathered_val_num_padding[key],
                gathered_val_image_to_text_ratio[key],
                gathered_val_steps[key],
            ) = self.accelerator.gather(
                (
                    convert_to_tensor(val_per_token_loss_acc[key]),
                    convert_to_tensor(val_num_images[key]),
                    convert_to_tensor(val_num_tokens[key]),
                    convert_to_tensor(val_num_padding[key]),
                    convert_to_tensor(val_image_to_text_ratio[key]),
                    convert_to_tensor(val_steps[key]),
                )
            )

            # No overall steps so we should skip this
            if gathered_val_steps[key].sum() == 0:
                gathered_val_per_token_loss.pop(key)
                gathered_val_num_images.pop(key)
                gathered_val_num_tokens.pop(key)
                gathered_val_num_padding.pop(key)
                gathered_val_image_to_text_ratio.pop(key)
                continue

            gathered_val_per_token_loss[key] = (
                gathered_val_per_token_loss[key].sum().item() / gathered_val_steps[key].sum().item()
            )
            gathered_val_num_images[key] = gathered_val_num_images[key].sum().item()
            gathered_val_num_tokens[key] = gathered_val_num_tokens[key].sum().item()
            gathered_val_num_padding[key] = gathered_val_num_padding[key].sum().item()
            gathered_val_image_to_text_ratio[key] = (
                gathered_val_image_to_text_ratio[key][gathered_val_steps[key] != 0.0].mean().item()
            )

        val_logs = {
            "val_per_token_loss": gathered_val_per_token_loss,
            "val_num_images": gathered_val_num_images,
            "val_num_tokens": gathered_val_num_tokens,
            "val_num_padding": gathered_val_num_padding,
            "val_image_to_text_ratio": gathered_val_image_to_text_ratio,
        }
        if self.accelerator.is_main_process:
            print(f"Validation logs: {self.format_val_logs(val_logs, LoggingTypes.PRINT)}")
            jsonl_logs = {"current step": curr_opt_step, "set": "validation"}
            jsonl_logs.update(self.format_val_logs(val_logs, LoggingTypes.JSONL))

            if self.hparams.job_id is not None:
                log_jsonl_file = self.hparams.save_dir / "logs" / f"{self.hparams.job_id}_logs.jsonl"
            else:
                log_jsonl_file = self.hparams.save_dir / "logs" / "logs.jsonl"

            with open(log_jsonl_file, "a") as f:
                f.write(json.dumps(jsonl_logs) + "\n")

            if self.hparams.wandb_enable:
                self.accelerator.log({**val_logs, **self._get_additional_step_logs()}, step=curr_opt_step)

    def train(self, maybe_torch_profile_scheduler=None):
        # timing_break_down = self.accelerator.is_main_process and self.hparams.timing_break_down

        if self.accelerator.is_main_process:
            logger.info(f"** Global main process pid={os.getpid()} **")
        elif self.accelerator.is_local_main_process:
            logger.info(f"** Local main process pid={os.getpid()} **")
        # --------------------
        # Set-up everything needed for training
        # --------------------
        (
            progress_columns,
            train_logs,
            max_num_epochs,
            max_num_updates,
            curr_opt_step,
            curr_epoch,
            opt_step_is_saved,
            eval_is_done,
            gbs_running,
        ) = self._set_up_training()

        # --------------------
        # Training loop
        # --------------------
        self.vl_model.train()
        pynvml_handle = pynmvl_handle(self.accelerator)
        with Progress(*progress_columns, refresh_per_second=5, disable=True) as progress:
            progress_bar = progress.add_task(
                "[red]Training", disable=not self.accelerator.is_main_process, total=max_num_updates, visible=False
            )
            train_task = progress.tasks[-1]
            progress.update(progress_bar, advance=curr_opt_step)
            finished_training = False
            training_logged = True
            timer = DeviceAgnosticTimer()
            timer.start()

            if self.hparams.timing_break_down:
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    timer2 = Timer()
                    time_deltas = {}

            while not finished_training:
                self.train_loader.set_epoch(curr_epoch)

                # Handle resume based on `realtime_processing`
                if self.hparams.resume_run:
                    if not self.data_param.realtime_processing:
                        # When preparing, `accelerate` has a bug that overwrites the top-level `sampler`... but this is aesthetic!
                        #   The "actual" sampler that the DataLoader uses is the one that's tucked inside the `batch_sampler`
                        #   attribute when "single-process" (world_size = 1), or the `batch_sampler.batch_sampler` when
                        #   "multi-process" (world_size > 1) -- both of which point to our specialized ResumableSampler!
                        #
                        #   This is pretty annoying and nuanced; should PR into `accelerate` to fix this...
                        logger.warning("NOT realtime processing has not been extensively tested yet")
                        if self.accelerator.num_processes == 1:
                            # TODO; Fix this

                            self.train_loader.batch_sampler.sampler.set_state(self.train_loader.get_resume_state(0))

                        else:
                            # TODO :: This is actually broken and not respected by `accelerate` - fails!
                            # self.train_loader.batch_sampler.batch_sampler.sampler.set_state(
                            #     self.resumable_state.get_resume_state()
                            # )
                            raise NotImplementedError("Map Dataset Resume w/ DDP not yet implemented!")
                    else:
                        self.train_loader.load_resume_states()

                if self.hparams.timing_break_down:
                    self.accelerator.wait_for_everyone()
                    if self.accelerator.is_main_process:
                        timer2.start()

                for curr_idx, (dataset_idx, dataset_name, dataset_state, batch) in enumerate(self.train_loader):
                    # --------------------
                    # Check if the training is over and if so may be save the model before training batch
                    # --------------------
                    if self.hparams.timing_break_down:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            time_deltas["dl"] = timer2.delta()

                    finished_training, opt_step_is_saved = self._check_if_training_is_over_and_maybe_save_model(
                        curr_opt_step,
                        curr_epoch,
                        gbs_running,
                        max_num_updates,
                        train_logs,
                        opt_step_is_saved,
                    )
                    if finished_training:
                        break

                    # --------------------
                    # Activate/deactivate hooks for logging activations or not
                    # --------------------
                    # We are logging everything at `curr_opt_step`, but `curr_opt_step` is incremented a few lines later, so activating
                    # the activation tracking hooks based on `curr_opt_step + 1`. See `_log_activations` for more details.
                    if self.activation_tracker:
                        if (curr_opt_step + 1) % self.hparams.train_logging_activations_opt_steps == 0 and (
                            curr_idx + 1
                        ) % self.hparams.grad_acc_size == 0:
                            self.activation_tracker.activate_hooks()
                        else:
                            self.activation_tracker.deactivate_hooks()

                    if (
                        self.hparams.save_batch_max_idx is not None
                        and self.hparams.save_batch_min_idx is not None
                        and curr_idx <= self.hparams.save_batch_max_idx
                        and curr_idx >= self.hparams.save_batch_min_idx
                    ):
                        self._save_batch(batch, curr_idx)

                    # right before fwd-bwd-step
                    if self.hparams.timing_break_down:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            time_deltas["between_dl_fwd_bwd"] = timer2.delta()
                    total_energy_start = pynvml_get_total_energy_in_joules(pynvml_handle)

                    with self.accelerator.accumulate(self.vl_model):
                        (
                            per_token_loss,
                            z_loss,
                            num_images,
                            num_image_tokens,
                            num_tokens,
                            image_to_text_ratio,
                            num_padding,
                            pixel_values_sum,
                            tflops_per_batch_per_gpu,
                        ) = self._do_batch(
                            batch,
                            curr_opt_step=curr_opt_step,
                            dataset_name=dataset_name,
                            dataset_idx=dataset_idx,
                            validation=False,
                        )

                    # right after fwd-bwd-step
                    if self.hparams.timing_break_down:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            time_deltas["fwd-bwd-step"] = timer2.delta()

                    fwd_bwd_time = timer.stop()
                    fwd_bwd_time = torch.tensor(fwd_bwd_time, device=self.accelerator.device)

                    total_energy_delta_per_gpu = torch.tensor(
                        pynvml_get_total_energy_in_joules(pynvml_handle) - total_energy_start,
                        device=self.accelerator.device,
                    )

                    if (curr_idx + 1) % self.hparams.grad_acc_size == 0:
                        curr_opt_step += 1
                        opt_step_is_saved, eval_is_done, training_logged = False, False, False
                        progress.update(progress_bar, advance=1)

                    # --------------------
                    # Update logs
                    # --------------------
                    train_logs = self._update_logs(
                        curr_opt_step,
                        curr_epoch,
                        gbs_running.global_batch_size_current,
                        train_logs,
                        per_token_loss,
                        z_loss,
                        num_tokens,
                        num_images,
                        num_image_tokens,
                        image_to_text_ratio,
                        num_padding,
                        fwd_bwd_time,
                        pixel_values_sum,
                        tflops_per_batch_per_gpu,
                        total_energy_delta_per_gpu,
                        dataset_name,
                        self.hparams.train_logging_per_dataset_suffix,
                    )
                    timer = DeviceAgnosticTimer()
                    timer.start()
                    # --------------------
                    # Update datasets states
                    # --------------------
                    self._update_datasets_states(dataset_idx, dataset_state)

                    # --------------------
                    # Log training infos
                    # --------------------
                    if curr_opt_step % self.hparams.train_logging_opt_steps == 0 and not training_logged:
                        train_logs = self._log_training(curr_opt_step, train_task, train_logs)
                        training_logged = True

                    # --------------------
                    # Log activations
                    # --------------------
                    if self.activation_tracker:
                        batch_idx = train_logs["num_batches"]["all"]
                        self.activation_tracker.fill_in_batch_idx(batch_idx=batch_idx)
                        if curr_opt_step % self.hparams.train_logging_activations_opt_steps == 0:
                            self._log_activations(curr_opt_step=curr_opt_step)

                    # ---------------------------
                    # Global batch size ramp up
                    # ---------------------------
                    #
                    # This logic needs to happen after the batch has been processed and results
                    # logged, but before the model is saved for resume, so that the updated ramup up
                    # variables will have the correct values on resume
                    gbs_running.global_seen_samples += self.accelerator.num_processes * self.hparams.batch_size_per_gpu
                    if (
                        self.hparams.global_batch_size_ramp_up.start is not None
                        and self.hparams.global_batch_size_ramp_up.finish > gbs_running.global_batch_size_current
                        and gbs_running.global_seen_samples >= gbs_running.next_goal_samples
                    ):
                        gbs_running.next_goal_samples += self.hparams.global_batch_size_ramp_up.samples

                        gbs_running.global_batch_size_current += self.hparams.global_batch_size_ramp_up.increment
                        gbs_running.grad_acc_size_current = int(
                            gbs_running.global_batch_size_current
                            / (self.hparams.batch_size_per_gpu * self.accelerator.num_processes)
                        )

                        self.update_gas_and_gbs(
                            gbs_running.grad_acc_size_current, gbs_running.global_batch_size_current
                        )

                    # --------------------
                    # Check if the training is over and if so may be save the model before validation
                    # --------------------
                    finished_training, opt_step_is_saved = self._check_if_training_is_over_and_maybe_save_model(
                        curr_opt_step,
                        curr_epoch,
                        gbs_running,
                        max_num_updates,
                        train_logs,
                        opt_step_is_saved,
                    )

                    if finished_training:
                        break

                    if self.hparams.timing_break_down:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            time_deltas["post_fwd"] = timer2.delta()
                            time_deltas["iteration"] = timer2.elapsed()

                    # --------------------
                    # Validation loop
                    # --------------------
                    if (
                        self.config.hparams.do_validation
                        and not eval_is_done
                        and curr_opt_step != 0
                        and curr_opt_step % self.hparams.val_logging_opt_steps == 0
                    ):
                        if self.hparams.timing_break_down:
                            self.accelerator.wait_for_everyone()
                            if self.accelerator.is_main_process:
                                timer3 = Timer()
                                timer3.start()

                        gc.collect()
                        if self.accelerator.is_main_process and self.hparams.wandb_enable:
                            wandb.unwatch(self.dummy_module)
                        if self.activation_tracker:
                            self.activation_tracker.is_eval()
                        logger.info("** Starting validation **")
                        (
                            val_steps,
                            val_per_token_loss_acc,
                            val_num_images,
                            val_num_tokens,
                            val_image_to_text_ratio,
                            val_num_padding,
                        ) = self._do_validation(progress, curr_opt_step)

                        # --------------------
                        # Log validation infos
                        # --------------------
                        self._log_validation(
                            val_steps,
                            curr_opt_step,
                            val_per_token_loss_acc,
                            val_num_images,
                            val_num_tokens,
                            val_image_to_text_ratio,
                            val_num_padding,
                        )
                        eval_is_done = True
                        logger.info("** Finished validation **")
                        if self.accelerator.is_main_process and self.hparams.wandb_enable:
                            wandb.watch(
                                self.dummy_module,
                                log="all",
                                log_freq=self.hparams.wandb_log_freq * self.hparams.grad_acc_size,
                                idx=0,
                            )
                        if self.activation_tracker:
                            self.activation_tracker.is_train()
                        gc.collect()
                        self.accelerator.wait_for_everyone()

                        if self.hparams.timing_break_down:
                            self.accelerator.wait_for_everyone()
                            if self.accelerator.is_main_process:
                                print(f"[TIME] Validation: {format_secs_to_time(timer3.stop())}")

                        # restart timer from zero to avoid accounting for validation
                        timer = DeviceAgnosticTimer()
                        timer.start()

                    if self.hparams.timing_break_down:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            # Finalize
                            print(
                                f'[TIME] Iteration {format_secs_to_sec_fractions(time_deltas["iteration"])}:'
                                f' {format_secs_to_sec_fractions(time_deltas["dl"]):>6} dl |'
                                f' {format_secs_to_sec_fractions(time_deltas["between_dl_fwd_bwd"]):>6} between'
                                f' dl/fwd-bwd | {format_secs_to_sec_fractions(time_deltas["fwd-bwd-step"]):>6} fwd/bwd'
                                f' | {format_secs_to_sec_fractions(time_deltas["post_fwd"]):>6} post'
                            )

                            # restart for __iter__
                            timer2.stop()
                            timer2.start()

                    if maybe_torch_profile_scheduler is not None and self.accelerator.is_main_process:
                        maybe_torch_profile_scheduler.step()

                if not finished_training:
                    curr_epoch += 1
                    train_logs = self._end_of_epoch_reset_train_logs(train_logs)

                    self.train_loader.reset_state()

                if curr_epoch == max_num_epochs:
                    self._save(
                        train_logs,
                        curr_opt_step,
                        curr_epoch,
                        gbs_running,
                    )
                    finished_training = True
                    logger.info("** Maximum number of epochs has been reached **")
                    break

            if self.hparams.wandb_enable:
                self.accelerator.end_training()

        return train_logs

    def _get_system_metrics_logs(self, memory_value_max):
        return {"memory_max_over_nodes": memory_value_max}
