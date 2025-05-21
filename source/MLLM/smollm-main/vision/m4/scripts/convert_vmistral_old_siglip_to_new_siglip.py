import argparse

from m4.models.vmistral.configuration_vmistral import VMistralConfig
from m4.models.vmistral.modeling_vmistral import VMistralForCausalLM


KEYS_TO_MODIFY_MAPPING = {
    "vision_model.vision_model": "vision_model",
}


def rename_state_dict(state_dict):
    model_state_dict = {}

    for key, value in state_dict.items():
        # check if any key needs to be modified
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)
        model_state_dict[key] = value

    return model_state_dict


def save_model_and_config(new_checkpoint_path, new_model):
    new_model.save_pretrained(new_checkpoint_path)


def check_loaded_model(new_checkpoint_path):
    new_model = VMistralForCausalLM.from_pretrained(
        new_checkpoint_path, is_resume=False, new_model=False, trust_remote_code=True
    )
    print(f"model: {new_model}")


def convert_checkpoint(checkpoint_path, new_checkpoint_path, new_siglip_model_path, save=False):
    model_name = checkpoint_path + "/unwrapped_model"
    model = VMistralForCausalLM.from_pretrained(model_name, is_resume=False, new_model=False, trust_remote_code=True)

    config_path = checkpoint_path + "/unwrapped_model/config.json"
    new_config = VMistralConfig.from_pretrained(
        config_path,
        new_model=False,
    )
    new_config.vision_config.vision_model_name = new_siglip_model_path
    new_model = VMistralForCausalLM.from_config(new_config)

    state_dict = rename_state_dict(model.state_dict())
    new_model.load_state_dict(state_dict)

    if save:
        save_model_and_config(new_checkpoint_path, new_model)

    check_loaded_model(new_checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default=(
            "/fsx/m4/experiments/local_experiment_dir/s3_async_temporary_checkpoint_folder/temp_266_dir/opt_step-38000"
        ),
        type=str,
        help="Path to the output PyTorch model.",
    )
    parser.add_argument("--new_checkpoint_path", default="new_model", type=str, help="Path to fairseq checkpoint")
    parser.add_argument(
        "--new_siglip_model_path",
        default="/fsx/leo/repos/siglip-so400m-14-384-flash-attn2",
        type=str,
        help="Path to new siglip model",
    )
    parser.add_argument("--save", default=True, type=str, help="Path to image")

    args = parser.parse_args()

    convert_checkpoint(args.checkpoint_path, args.new_checkpoint_path, args.new_siglip_model_path, args.save)
