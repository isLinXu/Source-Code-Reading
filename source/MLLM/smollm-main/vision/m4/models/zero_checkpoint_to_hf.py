#!/usr/bin/env python

# This script combines the 2 steps of
# 1. calling zero_to_fp32.py to reconsolidate the shared deepspeed checkpoint
# 2. then resaving it as HF checkpoint, which also takes care of sharding large checkpoints
#
# example usage:
#
# this will generate the converted checkpoint under save_dir/opt_step-40/unwrapped_model
#
# ./m4/models/zero_checkpoint_to_hf.py save_dir/opt_step-40
#
# or you can override the destination by passing an explicit target dir, e.g.:
#
# ./m4/models/zero_checkpoint_to_hf.py save_dir/opt_step-40 save_dir/opt_step-40/output_dir

import argparse
import concurrent.futures
import os
import sys
from collections import OrderedDict
from functools import partial
from glob import glob
from pathlib import Path

import torch
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import PeftConfig, get_peft_model
from tqdm import tqdm


# auto-append the repo path to load m4 modules from instead of needing to set PYTHONPATH
repodir = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, repodir)

import m4.models
from m4.testing_utils import read_json_file


def delete_optim_state_params(path_bf16_zero_pp_file, old_directory, new_directory):
    data = torch.load(path_bf16_zero_pp_file)
    data["optimizer_state_dict"].pop("optimizer_state_dict", None)
    modif_path = path_bf16_zero_pp_file.replace(old_directory, new_directory)
    torch.save(data, modif_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint_dir", type=str, help="path to the desired checkpoint folder, e.g., path/to/opt_step-100"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        nargs="?",
        help="path to pass to save_pretrained, defaults to 'unwrapped_model' relative to the checkpoint_dir argument",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=48,
        help="number of processes to use for deleting the unnecessary optimizer state dict",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    config_dir = checkpoint_dir / "unwrapped_model"
    ds_checkpoint_dir = checkpoint_dir / "accelerator_state"
    config_file_path = config_dir / "config.json"

    if args.output_dir is None:
        output_dir = checkpoint_dir / "unwrapped_model"
    else:
        output_dir = args.output_dir
    adapter_output_dir = checkpoint_dir / "unwrapped_adapter"

    config = read_json_file(config_file_path)
    config_class = m4.models._SUPPORTED_MODELS.get(config["model_type"], None)
    if config_class is None:
        raise ValueError(f"{config['model_type']=} isn't supported by m4")
    modeling_class = m4.models.model_type_to_modeling_class.get(config["model_type"], None)

    print(f"Detected {config_class}")

    print("Creating a new checkpoint directory to save parameters without the optimizer states")
    new_ds_checkpoint_dir = str(ds_checkpoint_dir) + "_modif"
    os.system(f"rsync -av --exclude='bf16_zero_pp_rank*' {str(ds_checkpoint_dir)}/ {new_ds_checkpoint_dir}/")
    paths_original_states = glob(f"{ds_checkpoint_dir}/pytorch_model/bf16_zero_pp_rank*")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_proc) as executor:
        futures = [
            executor.submit(
                partial(
                    delete_optim_state_params,
                    old_directory=str(ds_checkpoint_dir),
                    new_directory=str(new_ds_checkpoint_dir),
                ),
                path,
            )
            for path in paths_original_states
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()

    print("Reconsolidating fp32 model from checkpoint shards (can take a long time)")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(new_ds_checkpoint_dir)  # already on cpu

    # Keeping debug to use if you ever need to debug state dict
    # print("Saved State Dict")
    # for k, v in state_dict.items():
    #     print(f"{k} {v.shape}")

    kwargs = {}
    print(f"Loading config from {config_dir}")
    model_config = config_class.from_pretrained(config_dir)

    print(f"Instantiating a {modeling_class} model in bf16")
    model = modeling_class.from_pretrained(
        None, config=model_config, state_dict=state_dict, torch_dtype=torch.bfloat16
    )

    if adapter_output_dir.exists():
        peft_config = PeftConfig.from_pretrained(adapter_output_dir)
        peft_model = get_peft_model(model, peft_config)
        peft_state_dict = OrderedDict()
        for name, param in state_dict.items():
            peft_state_dict["base_model.model." + name] = param

        peft_model.load_state_dict(peft_state_dict)

        # This is only saving the adapter, not the base model
        print(f"Saving adapter model to {adapter_output_dir}")
        peft_model.save_pretrained(adapter_output_dir)

        print(f"Saving base model to {output_dir}")
        base_model = peft_model.unload()
        base_model.save_pretrained(output_dir)
    else:
        print(f"Saving model to {output_dir}")
        model.save_pretrained(output_dir)

    # Keeping debug to use if you ever need to debug state dict
    # print("Model State Dict")
    # for k, v in model.state_dict().items():
    #     print(f"{k} {v.shape}")

    print(f"Deleting the modified accelerator state files at {new_ds_checkpoint_dir}")
    os.system(f"rm -r {new_ds_checkpoint_dir}")
