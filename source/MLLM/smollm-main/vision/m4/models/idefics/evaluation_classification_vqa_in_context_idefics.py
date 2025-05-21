from m4.evaluation.custom_metrics.classification_vqa_metrics import ClassifVQAMetrics
from m4.models.vgpt2.evaluation_classification_vqa_in_context_vgpt2 import Vgpt2ClassificationVQAInContext


class IdeficsClassificationVQAInContext(Vgpt2ClassificationVQAInContext):
    model_class: str = "IdeficsForCausalLM"
    tokenizer_max_seq_len = 2048


class VQAv2IdeficsClassificationVQAInContextAcc(IdeficsClassificationVQAInContext):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif"
    metric_name: str = "ClassificationVQAMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifVQAMetrics.VQA_ACCURACY,
            ClassifVQAMetrics.ENTROPY_DISTRIBUTION,
            ClassifVQAMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"
    length_normalize: bool = False


class VQAv2SampleIdeficsClassificationVQAInContextAcc(VQAv2IdeficsClassificationVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Sample"
