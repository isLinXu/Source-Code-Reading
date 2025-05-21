from enum import Enum
from typing import List, Optional


class Predictor(Enum):
    in_contexter = "in_contexter"
    linear_prober = "linear_prober"


class BaseTask:
    dataset_name: str  # Dataset (example: birdsnap)
    dataset_config: Optional[str] = None  # Dataset config (example: partition_1)
    default_support_split_name: Optional[str] = None
    default_query_split_name: str
    metric_name: str  # the metric to use (example: accuracy) - use evaluate
    metrics_kwargs: Optional[dict] = {}
    extra_metrics: Optional[list] = None
    model_class: str  # The model
    predictor_class: Predictor
    id_column_name: Optional[str] = None

    def __init__(self, **kwargs) -> None:
        pass


class BaseTaskClassification(BaseTask):
    image_column_names: List[str]
    label_column_name: str
    context_column_names: Optional[List[str]] = None
    tested_ex_excluded_context_columns: Optional[List[str]] = None
    tested_labels_column_name: Optional[str] = None
    relevance_scores_column_name: Optional[str] = None


class BaseTaskOpenEndedVQA(BaseTask):
    image_column_name: str
    question_column_name: str
    answers_column_name: str
    context_column_names: Optional[List[str]] = None


class BaseTaskImageCaptioning(BaseTask):
    image_column_name: str
    reference_captions_column_name: str
    context_column_name: Optional[str] = None


class BaseTaskImageCaptionMatching(BaseTask):
    image_column_names: List[str]
    caption_column_names: List[str]
