import re

from transformers import AutoModel


# map to check the supported cv archs and also how to extract the model - in some arch, we want to
# go through a specific prefix to get to the model as in `model.vision_model` for clip
vision_model_name2model = {
    r"clip": lambda model: model.vision_model,
    r"siglip": lambda model: model.vision_model,
    r"vit": lambda model: model,
}


def vision_model_name_to_model(model_name_or_path, model):
    """returns the model if supported, asserts otherwise"""

    model_name_lowcase = model_name_or_path.lower()
    for rx, lookup in vision_model_name2model.items():
        if re.search(rx, model_name_lowcase):
            return lookup(model)
    else:
        raise ValueError(
            f"Unknown type of backbone vision model. Got {model_name_or_path}, supported regexes:"
            f" {list(vision_model_name2model.keys())}."
        )


def get_vision_model(config):
    vision_model_name = config.vision_model_name
    vision_model_params = eval(config.vision_model_params)

    model = AutoModel.from_pretrained(vision_model_name, **vision_model_params, trust_remote_code=True)
    return vision_model_name_to_model(vision_model_name, model)
