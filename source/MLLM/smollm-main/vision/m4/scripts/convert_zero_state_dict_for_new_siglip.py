import argparse
import os
from collections import OrderedDict

import torch


KEYS_TO_MODIFY_MAPPING = {
    "vision_model.vision_model": "vision_model",
}


def rename_state_dict(state_dict):
    model_state_dict = OrderedDict()

    for key, value in state_dict.items():
        # check if any key needs to be modified
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        model_state_dict[key] = value

    return model_state_dict


def convert_and_save_checkpoints(checkpoint_path, save=False):
    for filename in os.listdir(checkpoint_path):
        if "model_states" in filename:
            partial_zero_checkpoint = torch.load(checkpoint_path + "/" + filename)
            for key, checkpoint_item in partial_zero_checkpoint.items():
                if isinstance(checkpoint_item, OrderedDict):
                    partial_zero_checkpoint[key] = rename_state_dict(checkpoint_item)
                elif isinstance(checkpoint_item, list):
                    for i, list_item in enumerate(checkpoint_item):
                        if isinstance(list_item, OrderedDict):
                            partial_zero_checkpoint[key][i] = rename_state_dict(list_item)
            torch.save(partial_zero_checkpoint, checkpoint_path + "/" + filename)
            print(f"Saved updated checkpoint: {checkpoint_path + '/' + filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default="/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/temp_266_dir/opt_step-38000/accelerator_state/pytorch_model",
        type=str,
        help="Path to the PyTorch model.",
    )
    parser.add_argument("--save", default=True, type=str, help="Path to image")

    args = parser.parse_args()
    convert_and_save_checkpoints(args.checkpoint_path, args.save)
