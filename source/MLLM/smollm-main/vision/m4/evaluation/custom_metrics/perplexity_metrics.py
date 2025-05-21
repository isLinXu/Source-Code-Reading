import logging
from enum import Enum
from typing import List

import datasets
import evaluate


_DESCRIPTION = ""
_CITATION = ""
_KWARGS_DESCRIPTION = ""

logger = logging.getLogger(__name__)


class MetricsPerplexity(Enum):
    PERPLEXITY = "perplexity"


class PerplexityMetrics(evaluate.Metric):
    """This class takes care of computing the metrics listed in `metrics`."""

    def __init__(self, metrics: List[MetricsPerplexity], **kwargs):
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
                    "example_ids": datasets.Value("string"),
                    "perplexities": datasets.Value("float32"),
                }
            ),
        )

    def _compute(self, example_ids, perplexities):
        data_per_id = {}
        for ex_id, perplexity in zip(example_ids, perplexities):
            # If condition is a dirty trick to handle the case of distributed evaluation where some instances can be
            # repeated over a few proceses to make the batches even.
            # In this case, we just verify that all processes predicted the same thing, and only take one copy of predictions
            # in order to not mess up metrics. Ideally this "unique" logic should be handled outside of the metric or maybe
            # in the add_batch call...
            if ex_id not in data_per_id:
                data_per_id[ex_id] = perplexity
            else:
                difference = data_per_id[ex_id] - perplexity
                logger.warning(
                    f"example_id repeated: {ex_id} \n Perplexity of original sample: {data_per_id[ex_id]} | Perplexity"
                    f" of discarded sample: {perplexity} | difference = {difference}"
                )
        # assert list(range(len(data_per_id))) == sorted(data_per_id.keys())

        results = {}
        if MetricsPerplexity.PERPLEXITY in self.metrics:
            perplexity_scores = []
            for ex_id, perplexity in data_per_id.items():
                perplexity_scores.append(perplexity)

            results["perplexity"] = float(sum(perplexity_scores) / len(perplexity_scores))
        return results
