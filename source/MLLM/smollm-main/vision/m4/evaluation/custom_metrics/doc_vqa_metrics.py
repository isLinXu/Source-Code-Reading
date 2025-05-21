import logging
from enum import Enum
from typing import List

import datasets
import evaluate
import Levenshtein

from m4.evaluation.custom_metrics.utils import VQANormalizationGtVisionLab


logger = logging.getLogger(__name__)

_DESCRIPTION = ""
_CITATION = ""
_KWARGS_DESCRIPTION = ""

logger = logging.getLogger(__name__)


class DVQAMetrics(Enum):
    ANLS = "average_normalized_levenshtein_similarity"
    DEFAULT_TO_SERVER_RESULTS = "default_to_server_results"


def normalized_levenshtein(s1, s2):
    len_s1, len_s2 = len(s1), len(s2)
    distance = Levenshtein.distance(s1, s2)
    return distance / max(len_s1, len_s2)


def similarity_score(a_ij, o_q_i, tau=0.5):
    nl = normalized_levenshtein(a_ij, o_q_i)
    return 1 - nl if nl < tau else 0


def average_normalized_levenshtein_similarity(ground_truth, predicted_answers):
    assert len(ground_truth) == len(predicted_answers), "Length of ground_truth and predicted_answers must match."

    N = len(ground_truth)
    total_score = 0

    for i in range(N):
        a_i = ground_truth[i]
        o_q_i = predicted_answers[i]
        if o_q_i == "":
            logger.warning("Skipped an empty prediction.")
            max_score = 0
        else:
            max_score = max(similarity_score(a_ij, o_q_i) for a_ij in a_i)

        total_score += max_score

    return total_score / N


class DocVQAMetrics(evaluate.Metric):
    """"""

    def __init__(self, metrics: List[DVQAMetrics], save_generations: bool, **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics
        self.save_generations = save_generations
        self.gt_vision_lab_normalization = VQANormalizationGtVisionLab()

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
                    "generated_texts": datasets.Value("string"),
                    "answers": datasets.Sequence(datasets.Value("string")),
                }
            ),
        )

    def _compute(self, example_ids, generated_texts, answers):
        data_per_id = {}
        for ex_id, gen_text, ans in zip(example_ids, generated_texts, answers):
            gen_text = (
                gen_text.strip()
            )  # This is not technically part of the metric itself. Equivalent of `vqa_normalize_text` for in vqa

            # If condition is a dirty trick to handle the case of distributed evaluation where some instances can be
            # repeated over a few proceses to make the batches even.
            # In this case, we just verify that all processes predicted the same thing, and only take one copy of predictions
            # in order to not mess up metrics. Ideally this "unique" logic should be handled outside of the metric or maybe
            # in the add_batch call...
            if ex_id not in data_per_id:
                data_per_id[ex_id] = {
                    "generated_text": gen_text,
                    "answers": ans,
                }
            else:
                if data_per_id[ex_id]["generated_text"] != gen_text:
                    logger.warning(
                        f"Example {ex_id} has different predictions accross processes. We have: {gen_text} and"
                        f" {data_per_id[ex_id]['generated_text']}"
                    )
                if data_per_id[ex_id]["answers"] != ans:
                    logger.warning(
                        f"Example {ex_id} has different answers accross processes. We have: {ans} and"
                        f" {data_per_id[ex_id]['answers']}"
                    )

        results = {}
        default_to_save_generations = answers[0] is None and DVQAMetrics.DEFAULT_TO_SERVER_RESULTS in self.metrics
        if self.save_generations or default_to_save_generations:
            # If answers are None, we default to the server results
            results["server_results"] = [
                {
                    "question_id": (
                        ex_id
                    ),  # TODO: change that field (and perhaps the dump format itself) to actually match the server
                    "answer": data["generated_text"].strip("."),
                }
                for ex_id, data in data_per_id.items()
            ]

        if default_to_save_generations:
            return results

        # assert list(range(len(data_per_id))) == sorted(data_per_id.keys())
        generated_texts_unique = [data_per_id[i]["generated_text"].strip(".").lower() for i in set(example_ids)]
        answers_unique = [[a.lower() for a in data_per_id[i]["answers"]] for i in set(example_ids)]
        # ANLS
        results["anls"] = average_normalized_levenshtein_similarity(
            predicted_answers=generated_texts_unique, ground_truth=answers_unique
        )
        return results
