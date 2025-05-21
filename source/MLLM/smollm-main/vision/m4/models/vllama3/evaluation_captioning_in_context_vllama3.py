from m4.evaluation.custom_metrics.unfolded_image_captioning_metrics import ImageCaptioningMetrics
from m4.models.vgpt2.evaluation_captioning_in_context_vgpt2 import Vgpt2ImageCaptioningInContext


class VLlama3ImageCaptioningInContext(Vgpt2ImageCaptioningInContext):
    model_class: str = "VLlama3ForCausalLM"
    tokenizer_max_seq_len = 10240


class TextCapsVLlama3ImageCaptioningInContextTextGenMetrics(VLlama3ImageCaptioningInContext):
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


class TextCapsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsVLlama3ImageCaptioningInContextTextGenMetrics
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


class TextCapsNewSplitsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge
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


class TextCapsSampleVLlama3ImageCaptioningInContextTextGenMetrics(
    TextCapsVLlama3ImageCaptioningInContextTextGenMetrics
):
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class TextCapsSampleVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    TextCapsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge
):
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class CommonGenVLlama3ImageCaptioningInContextTextGenMetrics(VLlama3ImageCaptioningInContext):
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
            "example": "<image>Input: {context}. Output: {caption}",
        }
    }


class CommonGenVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    CommonGenVLlama3ImageCaptioningInContextTextGenMetrics
):
    metric_kwargs = {
        "metrics": [
            ImageCaptioningMetrics.BLEU_4,
            ImageCaptioningMetrics.CIDER,
            ImageCaptioningMetrics.METEOR,
            ImageCaptioningMetrics.ROUGE_L,
        ]
    }


class NoCapsVLlama3ImageCaptioningInContextTextGenMetrics(VLlama3ImageCaptioningInContext):
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


class NoCapsNewSplitsVLlama3ImageCaptioningInContextTextGenMetrics(
    NoCapsVLlama3ImageCaptioningInContextTextGenMetrics
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


class NoCapsSampleVLlama3ImageCaptioningInContextTextGenMetrics(NoCapsVLlama3ImageCaptioningInContextTextGenMetrics):
    dataset_name: str = "HuggingFaceM4/NoCaps-Sample"


class CocoVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(VLlama3ImageCaptioningInContext):
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


class CocoNewSplitsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    CocoVLlama3ImageCaptioningInContextBleuCiderMeteorRouge
):
    dataset_name: str = "HuggingFaceM4/coco_support_query_sets"
    dataset_config = None
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"
    selected_prompt_template_id = 7


class CocoSampleVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(
    CocoVLlama3ImageCaptioningInContextBleuCiderMeteorRouge
):
    dataset_name: str = "HuggingFaceM4/COCO-2014_captions-Sample"
    dataset_config = None


class Flickr30kNewSplitsVLlama3ImageCaptioningInContextBleuCiderMeteorRouge(VLlama3ImageCaptioningInContext):
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


class IIIT5KVLlama3ImageCaptioningInContextExactMatch(VLlama3ImageCaptioningInContext):
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
            "example": "<image>A photo where it is written {caption}",
        }
    }


class IIIT5KSampleVLlama3ImageCaptioningInContextExactMatch(IIIT5KVLlama3ImageCaptioningInContextExactMatch):
    dataset_name: str = "HuggingFaceM4/IIIT-5K-Sample"
