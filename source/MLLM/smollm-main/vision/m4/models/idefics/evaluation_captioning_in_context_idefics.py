from m4.evaluation.custom_metrics.unfolded_image_captioning_metrics import ImageCaptioningMetrics
from m4.models.vgpt2.evaluation_captioning_in_context_vgpt2 import Vgpt2ImageCaptioningInContext


class IdeficsImageCaptioningInContext(Vgpt2ImageCaptioningInContext):
    model_class: str = "IdeficsForCausalLM"


class TextCapsIdeficsImageCaptioningInContextTextGenMetrics(IdeficsImageCaptioningInContext):
    dataset_name: str = "HuggingFaceM4/TextCaps"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
            ImageCaptioningMetrics.SPICE,
            ImageCaptioningMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    reference_captions_column_name: str = "reference_strs"


class TextCapsIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsIdeficsImageCaptioningInContextTextGenMetrics
):
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
            ImageCaptioningMetrics.DEFAULT_TO_SERVER_RESULTS,
        ]
    }


class TextCapsNewSplitsIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsIdeficsImageCaptioningInContextBleuCiderMeteorRouge
):
    dataset_name: str = "HuggingFaceM4/TextCaps_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    server_check_support_split_name: str = "server_check_support_set"
    server_check_query_split_name: str = "server_check_query_set"
    id_column_name: str = "image_id"
    selected_prompt_template_id = 7


class TextCapsSampleIdeficsImageCaptioningInContextTextGenMetrics(
    TextCapsIdeficsImageCaptioningInContextTextGenMetrics
):
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class TextCapsSampleIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsIdeficsImageCaptioningInContextBleuCiderMeteorRouge
):
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class CommonGenIdeficsImageCaptioningInContextTextGenMetrics(IdeficsImageCaptioningInContext):
    dataset_name: str = "HuggingFaceM4/common_gen"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
            ImageCaptioningMetrics.SPICE,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    context_column_name: str = "concepts"
    reference_captions_column_name: str = "target"
    stop_words = ["Input", "Output"]
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "{token_around_image}{image_token}{token_around_image}Input: {context}. Output: {caption}",
        }
    }


class CommonGenIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    CommonGenIdeficsImageCaptioningInContextTextGenMetrics
):
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
        ]
    }


class NoCapsIdeficsImageCaptioningInContextTextGenMetrics(IdeficsImageCaptioningInContext):
    dataset_name: str = "HuggingFaceM4/NoCaps"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
            ImageCaptioningMetrics.DEFAULT_TO_SERVER_RESULTS,
            # ImageCaptioningMetrics.SPICE,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    # This does not exist yet... it would require adding a training split to the dataset (see `create_sample_evaluation_datasets_simplified.py`)
    image_column_name: str = "image"
    reference_captions_column_name: str = "annotations_captions"


class NoCapsNewSplitsIdeficsImageCaptioningInContextTextGenMetrics(
    NoCapsIdeficsImageCaptioningInContextTextGenMetrics
):
    dataset_name: str = "HuggingFaceM4/NoCaps_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    server_check_support_split_name: str = "server_check_support_set"
    server_check_query_split_name: str = "server_check_query_set"
    id_column_name: str = "image_id"
    selected_prompt_template_id = 7


class NoCapsSampleIdeficsImageCaptioningInContextTextGenMetrics(NoCapsIdeficsImageCaptioningInContextTextGenMetrics):
    dataset_name: str = "HuggingFaceM4/NoCaps-Sample"


class CocoIdeficsImageCaptioningInContextBleuCiderMeteorRouge(IdeficsImageCaptioningInContext):
    dataset_name: str = "HuggingFaceM4/COCO"
    dataset_config = "2014_captions"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    reference_captions_column_name: str = "sentences_raw"


class CocoNewSplitsIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    CocoIdeficsImageCaptioningInContextBleuCiderMeteorRouge
):
    dataset_name: str = "HuggingFaceM4/coco_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 7


class CocoSampleIdeficsImageCaptioningInContextBleuCiderMeteorRouge(
    CocoIdeficsImageCaptioningInContextBleuCiderMeteorRouge
):
    dataset_name: str = "HuggingFaceM4/COCO-2014_captions-Sample"
    dataset_config = None


class Flickr30kNewSplitsIdeficsImageCaptioningInContextBleuCiderMeteorRouge(IdeficsImageCaptioningInContext):
    dataset_name: str = "HuggingFaceM4/flickr30k_support_query_sets"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
        ]
    }
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    image_column_name: str = "image"
    reference_captions_column_name: str = "sentences"
    selected_prompt_template_id = 7


class IIIT5KIdeficsImageCaptioningInContextExactMatch(IdeficsImageCaptioningInContext):
    dataset_name: str = "HuggingFaceM4/IIIT-5K"
    metric_name: str = "UnfoldedImageCaptioningMetrics"
    metric_kwargs = {"metrics": [ImageCaptioningMetrics.EXACT_MATCH]}
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    reference_captions_column_name: str = "label"
    stop_words = ["A photo"]
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "{token_around_image}{image_token}{token_around_image}A photo where it is written {caption}",
        }
    }


class IIIT5KSampleIdeficsImageCaptioningInContextExactMatch(IIIT5KIdeficsImageCaptioningInContextExactMatch):
    dataset_name: str = "HuggingFaceM4/IIIT-5K-Sample"
