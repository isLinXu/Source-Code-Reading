from m4.evaluation.custom_metrics.perplexity_metrics import MetricsPerplexity
from m4.models.vgpt2.evaluation_perplexity_in_context_vgpt2 import Vgpt2PerplexityInContext
from m4.training.types import DatasetTypes


class VMistralPerplexityInContext(Vgpt2PerplexityInContext):
    model_class: str = "VMistralForCausalLM"
    add_end_of_doc_token: bool = True
    add_begin_of_doc_token: bool = True
    tokenizer_max_seq_len = 4096


class Cm4VMistralPerplexityInContextMaxSeLen1024(VMistralPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/cm4_valid"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    image_column_name: str = "images"
    text_column_name: str = "texts"
    default_query_split_name: str = "valid"
    ds_type: DatasetTypes = DatasetTypes.WEB_DOCUMENTS
    tokenizer_max_seq_len = 1024
    max_num_images = 70  # To be computed once the dataset is created


class Cm4VMistralPerplexityInContextMaxSeLen512(Cm4VMistralPerplexityInContextMaxSeLen1024):
    tokenizer_max_seq_len = 512
    max_num_images = 35  # To be computed once the dataset is created


class Cm4SampleVMistralPerplexityInContextMaxSeLen1024(Cm4VMistralPerplexityInContextMaxSeLen1024):
    dataset_name: str = "HuggingFaceM4/cm4_valid-Sample"
    max_num_images = 53


class Cm4SampleVMistralPerplexityInContextMaxSeLen512(Cm4VMistralPerplexityInContextMaxSeLen512):
    dataset_name: str = "HuggingFaceM4/cm4_valid-Sample"
    max_num_images = 53


class EnWikiVMistralPerplexityInContextMaxSeLen1024(VMistralPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    image_column_name: str = "images"
    text_column_name: str = "texts"
    default_query_split_name: str = "valid"
    ds_type: DatasetTypes = DatasetTypes.WEB_DOCUMENTS
    tokenizer_max_seq_len = 1024
    max_num_images = 70


class EnWikiVMistralPerplexityInContextMaxSeLen512(EnWikiVMistralPerplexityInContextMaxSeLen1024):
    tokenizer_max_seq_len = 512
    max_num_images = 35


class EnWikiSampleVMistralPerplexityInContextMaxSeLen1024(EnWikiVMistralPerplexityInContextMaxSeLen1024):
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid-Sample"


class EnWikiSampleVMistralPerplexityInContextMaxSeLen512(EnWikiVMistralPerplexityInContextMaxSeLen512):
    dataset_name: str = "HuggingFaceM4/enwiki-v2_valid-Sample"


class TextCapsVMistralPerplexityInContext(VMistralPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/TextCaps"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "reference_strs"


class TextCapsSampleVMistralPerplexityInContext(TextCapsVMistralPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/TextCaps-Sample"


class CommonGenVMistralPerplexityInContext(VMistralPerplexityInContext):
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


class NoCapsVMistralPerplexityInContext(VMistralPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/NoCaps"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "annotations_captions"


class NoCapsSampleVMistralPerplexityInContext(NoCapsVMistralPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/NoCaps-Sample"


class CocoVMistralPerplexityInContext(VMistralPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/COCO"
    dataset_config = "2014_captions"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "validation"
    image_column_name: str = "image"
    text_column_name: str = "sentences_raw"


class CocoSampleVMistralPerplexityInContext(CocoVMistralPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/COCO-2014_captions-Sample"
    dataset_config = None


class IIIT5KVMistralPerplexityInContext(VMistralPerplexityInContext):
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


class IIIT5KSampleVMistralPerplexityInContext(IIIT5KVMistralPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/IIIT-5K-Sample"


class MiniGPTCaptionsVMistralPerplexityInContext(VMistralPerplexityInContext):
    dataset_name: str = "HuggingFaceM4/mini-GPT-captions"
    metric_name: str = "PerplexityMetrics"
    metric_kwargs = {"metrics": [MetricsPerplexity.PERPLEXITY]}
    default_query_split_name: str = "train"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    text_column_name: str = "reference_strs"
