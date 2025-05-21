from m4.evaluation.custom_metrics.perplexity_metrics import MetricsPerplexity
from m4.models.vgpt2.evaluation_perplexity_in_context_vgpt2 import Vgpt2PerplexityInContext
from m4.training.types import DatasetTypes


class IdeficsPerplexityInContext(Vgpt2PerplexityInContext):
    model_class: str = "IdeficsForCausalLM"
    add_end_of_doc_token: bool = True
    add_begin_of_doc_token: bool = True
    tokenizer_max_seq_len = 2048


class Cm4IdeficsPerplexityInContextMaxSeLen1024(IdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/cm4_valid"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    image_column_name: str = "images"
    text_column_name: str = "texts"
    default_query_split_name: str = "valid"
    ds_type: DatasetTypes = DatasetTypes.WEB_DOCUMENTS
    tokenizer_max_seq_len = 1024
    max_num_images = 70  # To be computed once the dataset is created


class Cm4IdeficsPerplexityInContextMaxSeLen512(Cm4IdeficsPerplexityInContextMaxSeLen1024):
    tokenizer_max_seq_len = 512
    max_num_images = 35  # To be computed once the dataset is created


class Cm4SampleIdeficsPerplexityInContextMaxSeLen1024(Cm4IdeficsPerplexityInContextMaxSeLen1024):
    dataset_name: str = "HuggingFaceM4/cm4_valid-Sample"
    max_num_images = 53


class Cm4SampleIdeficsPerplexityInContextMaxSeLen512(Cm4IdeficsPerplexityInContextMaxSeLen512):
    dataset_name: str = "HuggingFaceM4/cm4_valid-Sample"
    max_num_images = 53


class EnWikiIdeficsPerplexityInContextMaxSeLen1024(IdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    image_column_name: str = "images"
    text_column_name: str = "texts"
    default_query_split_name: str = "valid"
    ds_type: DatasetTypes = DatasetTypes.WEB_DOCUMENTS
    tokenizer_max_seq_len = 1024
    max_num_images = 70


class EnWikiIdeficsPerplexityInContextMaxSeLen512(EnWikiIdeficsPerplexityInContextMaxSeLen1024):
    tokenizer_max_seq_len = 512
    max_num_images = 35


class EnWikiSampleIdeficsPerplexityInContextMaxSeLen1024(EnWikiIdeficsPerplexityInContextMaxSeLen1024):
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid-Sample"


class EnWikiSampleIdeficsPerplexityInContextMaxSeLen512(EnWikiIdeficsPerplexityInContextMaxSeLen512):
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid-Sample"


class TextCapsIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/TextCaps"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "reference_strs"


class TextCapsSampleIdeficsPerplexityInContext(TextCapsIdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class CommonGenIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/common_gen"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    context_column_name: str = "concepts"
    text_column_name: str = "target"

    def _create_image_caption_pair_prompt(self, caption="", context=""):
        return (
            f"{self.token_around_image}{self.image_token}{self.token_around_image}Input: {context}. Output: {caption}"
        )


class NoCapsIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/NoCaps"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "annotations_captions"


class NoCapsSampleIdeficsPerplexityInContext(NoCapsIdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/NoCaps-Sample"


class CocoIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/COCO"
    dataset_config = "2014_captions"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "sentences_raw"


class CocoSampleIdeficsPerplexityInContext(CocoIdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/COCO-2014_captions-Sample"
    dataset_config = None


class IIIT5KIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/IIIT-5K"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "test"
    image_column_name: str = "image"
    text_column_name: str = "label"

    def _create_image_caption_pair_prompt(self, caption="", context=None):
        if context is not None:
            raise NotImplementedError("Context not implemented for this task")
        return (
            f"{self.token_around_image}{self.image_token}{self.token_around_image}A photo where"
            f" it is written {caption}"
        )


class IIIT5KSampleIdeficsPerplexityInContext(IIIT5KIdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/IIIT-5K-Sample"


class MiniGPTCaptionsIdeficsPerplexityInContext(IdeficsPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/mini-GPT-captions"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "train"
    image_column_name: str = "image"
    text_column_name: str = "reference_strs"
