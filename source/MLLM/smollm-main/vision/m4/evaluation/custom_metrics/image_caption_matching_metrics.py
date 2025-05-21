import logging
from enum import Enum
from typing import List

import datasets
import evaluate


_DESCRIPTION = ""
_CITATION = ""
_KWARGS_DESCRIPTION = ""

logger = logging.getLogger(__name__)


class MetricsImageCaptionMatching(Enum):
    GROUP_SCORE = "group_score"
    TEXT_SCORE = "text_score"
    IMAGE_SCORE = "image_score"


class ImageCaptionMatchingMetrics(evaluate.Metric):
    """This class takes care of computing the metrics listed in `metrics`."""

    def __init__(self, metrics: List[MetricsImageCaptionMatching], **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics

        if metrics is None:
            raise ValueError("`metrics` must be specified")

    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "example_ids": datasets.Sequence(datasets.Value("string")),
                    "caption_ids": datasets.Sequence(datasets.Value("int32")),
                    "image_ids": datasets.Sequence(datasets.Value("int32")),
                    "splitted_scores_per_example": datasets.Sequence(datasets.Value("float64")),
                }
            ),
        )

    def _compute(self, example_ids, image_ids, caption_ids, splitted_scores_per_example):
        data_per_id = {}

        for ex_ids, ex_image_ids, ex_caption_ids, ex_scores in zip(
            example_ids, image_ids, caption_ids, splitted_scores_per_example
        ):
            # If condition is a dirty trick to handle the case of distributed evaluation where some instances can be
            # repeated over a few proceses to make the batches even.
            # In this case, we just verify that all processes predicted the same thing, and only take one copy of predictions
            # in order to not mess up metrics. Ideally this "unique" logic should be handled outside of the metric or maybe
            # in the add_batch call...
            assert all(id == ex_ids[0] for id in ex_ids)
            if ex_ids[0] not in data_per_id:
                data_per_id[int(ex_ids[0])] = {
                    "ex_caption_ids": ex_caption_ids,
                    "ex_image_ids": ex_image_ids,
                    "ex_scores": ex_scores,
                }
            else:
                ex_scores_differences = [
                    data_per_id[ex_ids[0]]["ex_scores"][i] - ex_scores[i] for i in range(len(ex_scores))
                ]
                assert data_per_id[ex_ids[0]]["ex_image_ids"] == ex_image_ids
                assert data_per_id[ex_ids[0]]["ex_caption_ids"] == ex_caption_ids
                ex_scores_difference = sum(ex_scores_differences)
                logger.warning(
                    f"example_id repeated: {ex_ids[0]} \n original sample: {data_per_id[ex_ids[0]]}"
                    f"text_score of discarded sample: {ex_scores} | difference = {ex_scores_difference}"
                )

        group_scores = [1] * len(data_per_id)
        image_scores = [1] * len(data_per_id)
        text_scores = [1] * len(data_per_id)
        for ex_idx in data_per_id.keys():
            ex_idx = int(ex_idx)
            ex_caption_ids = data_per_id[ex_idx]["ex_caption_ids"]
            ex_image_ids = data_per_id[ex_idx]["ex_image_ids"]
            ex_scores = data_per_id[ex_idx]["ex_scores"]
            for score_0, caption_idx_0, image_idx_0 in zip(ex_scores, ex_caption_ids, ex_image_ids):
                for score_1, caption_idx_1, image_idx_1 in zip(ex_scores, ex_caption_ids, ex_image_ids):
                    if caption_idx_0 == image_idx_0 and caption_idx_1 != image_idx_1:
                        # If we have a matching pair with a lower log_prob than any of the wrong pairs, the group_score is 0
                        if score_0 < score_1:
                            group_scores[ex_idx] = 0
                    if (
                        caption_idx_0 == image_idx_0
                        and caption_idx_1 != image_idx_1
                        and caption_idx_0 != caption_idx_1
                    ):
                        # If we have a matching pair with a lower log_prob than a pair with the same caption, but a different image, image_score is 0
                        if score_0 < score_1:
                            image_scores[ex_idx] = 0
                    if caption_idx_0 == image_idx_0 and caption_idx_1 != image_idx_1 and image_idx_0 != image_idx_1:
                        # If we have a matching pair with a lower log_prob than a pair with the same image, but a different caption, text_score is 0
                        if score_0 < score_1:
                            text_scores[ex_idx] = 0

        results = {}
        if MetricsImageCaptionMatching.TEXT_SCORE in self.metrics:
            results["text_score"] = float(sum(text_scores) / len(text_scores))

        if MetricsImageCaptionMatching.IMAGE_SCORE in self.metrics:
            results["image_score"] = float(sum(image_scores) / len(image_scores))

        if MetricsImageCaptionMatching.GROUP_SCORE in self.metrics:
            results["group_score"] = float(sum(group_scores) / len(group_scores))

        return results
