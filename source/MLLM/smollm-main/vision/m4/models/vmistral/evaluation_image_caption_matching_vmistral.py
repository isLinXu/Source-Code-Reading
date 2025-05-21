from typing import List

from m4.evaluation.custom_metrics.image_caption_matching_metrics import MetricsImageCaptionMatching
from m4.models.vgpt2.evaluation_image_caption_matching_vgpt2 import Vgpt2ImageCaptionMatching


class VMistralImageCaptionMatching(Vgpt2ImageCaptionMatching):
    model_class: str = "VMistralForCausalLM"
    tokenizer_max_seq_len = 4096


class WinogroundVMistralImageCaptionMatchingAccWithKLAndEntropy(VMistralImageCaptionMatching):
    dataset_name: str = "facebook/winoground"
    metric_name: str = "ImageCaptionMatchingMetrics"
    metric_kwargs = {
        "metrics": [
            MetricsImageCaptionMatching.TEXT_SCORE,
            MetricsImageCaptionMatching.IMAGE_SCORE,
            MetricsImageCaptionMatching.GROUP_SCORE,
        ]
    }
    # support split names are never used for this dataset
    default_query_split_name: str = "test"
    default_support_split_name: str = "test"
    test_query_split_name: str = "test"
    test_support_split_name: str = "test"
    image_column_names: List[str] = ["image_0", "image_1"]
    caption_column_names: List[str] = ["caption_0", "caption_1"]
    id_column_name: str = "id"
    length_normalize: bool = True
    selected_prompt_template_id = 0
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "{token_around_image}{image_token}{token_around_image}Description of the image: {caption}",
        },
        1: {
            "prefix": None,
            "example": "{bos_token}{token_around_image}{image_token}{token_around_image}{caption}{eos_token}",
        },
        2: {
            "prefix": None,
            "example": (
                "{bos_token}{token_around_image}{image_token}{token_around_image}In this image, we can see:"
                " {caption}{eos_token}"
            ),
        },
        3: {
            "prefix": None,
            "example": (
                "{bos_token}{token_around_image}{image_token}{token_around_image}Description of the image:"
                " {caption}{eos_token}"
            ),
        },
        4: {
            "prefix": None,
            "example": (
                "{token_around_image}{image_token}{token_around_image}Accurate description of the image: {caption}"
            ),
        },
        5: {
            "prefix": None,
            "example": (
                "{token_around_image}{image_token}{token_around_image}{caption}. Does this caption accurately describe"
                " the image? Yes or no? Yes"
            ),
        },
        6: {
            "prefix": None,
            "example": (
                "{token_around_image}{image_token}{token_around_image}caption: {caption}.\n Question: Does this"
                " caption accurately describe the image? Yes"
            ),
        },
    }
