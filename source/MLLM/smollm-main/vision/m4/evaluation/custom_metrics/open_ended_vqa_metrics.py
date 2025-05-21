import logging
from enum import Enum
from typing import List

import datasets
import evaluate
import numpy as np
from scipy import stats

from m4.evaluation.custom_metrics.utils import (
    VQANormalizationGtVisionLab,
    check_is_number,
    convert_to_number,
    normalize_str_mmmu,
    parse_open_response_mmmu,
    vqa_normalize_text,
)


logger = logging.getLogger(__name__)

_DESCRIPTION = ""
_CITATION = ""
_KWARGS_DESCRIPTION = ""

logger = logging.getLogger(__name__)

MCQ_POSSIBLE_CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]


class OEVQAMetrics(Enum):
    FIRST_WORD_VQA_ACCURACY = "first_word_vqa_accuracy"
    OE_VQA_ACCURACY = "oe_vqa_accuracy"
    OE_MMMU_STYLE_VQA_ACCURACY = "oe_mmmu_style_vqa_accuracy"
    OE_MMMU_STYLE_PER_BUCKET_ACCURACY = "oe_mmmu_style_per_bucket_accuracy"
    OE_ONLY_MMMU_STYLE_VQA_ACCURACY = "oe_only_mmmu_style_vqa_accuracy"
    OE_ONLY_MMMU_STYLE_PER_BUCKET_ACCURACY = "oe_only_mmmu_style_per_bucket_accuracy"
    OE_RELAXED_VQA_ACCURACY = "oe_relaxed_vqa_accuracy"
    GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY = "gt_vision_lab_first_word_vqa_accuracy"
    GT_VISION_LAB_OE_VQA_ACCURACY = "gt_vision_lab_oe_vqa_accuracy"
    DEFAULT_TO_SERVER_RESULTS = "default_to_server_results"
    DEFAULT_TO_SERVER_RESULTS_MMVET = "default_to_server_results_mmvet"
    DEFAULT_TO_SERVER_RESULTS_LLAVA_WILD = "default_to_server_results_llava_wild"


class OpenEndedVQAMetrics(evaluate.Metric):
    """This class takes care of computing the metrics listed in `metrics`."""

    def __init__(self, metrics: List[OEVQAMetrics], save_generations: bool, **kwargs):
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
                    "buckets": datasets.Value("string"),
                }
            ),
        )

    def _compute_vqa_accuracy(self, generated_texts_unique, answers_unique, normalize_text_fn):
        first_word_vqa_accuracy_scores = []
        oe_accuracy_scores = []
        for generated_text, answers_ in zip(generated_texts_unique, answers_unique):
            generated_text = normalize_text_fn(generated_text)
            generated_first_word = generated_text.split(" ")[0]
            answers_ = [normalize_text_fn(answer_) for answer_ in answers_]

            if len(answers_) == 1:
                # This is the case for GQA for example
                first_word_vqa_accuracy_scores.append((generated_first_word == answers_[0]) * 1.0)
                oe_accuracy_scores.append((generated_text == answers_[0]) * 1.0)

            else:
                gt_first_word_acc = []
                gt_oe_acc = []
                for idx_ref in range(len(answers_)):
                    other_answers_ = [other_answer for idx, other_answer in enumerate(answers_) if idx != idx_ref]

                    matched_with_first_word = [
                        other_answer for other_answer in other_answers_ if other_answer == generated_first_word
                    ]
                    matched_with_oe_text = [
                        other_answer for other_answer in other_answers_ if other_answer == generated_text
                    ]

                    first_word_acc = min(1, len(matched_with_first_word) / 3)
                    oe_acc = min(1, len(matched_with_oe_text) / 3)

                    gt_first_word_acc.append(first_word_acc)
                    gt_oe_acc.append(oe_acc)

                first_word_vqa_accuracy_scores.append(sum(gt_first_word_acc) / len(gt_first_word_acc))
                oe_accuracy_scores.append(sum(gt_oe_acc) / len(gt_oe_acc))
        return first_word_vqa_accuracy_scores, oe_accuracy_scores

    def _compute_mmmu_style_vqa_accuracy(self, generated_texts_unique, answers_unique, normalize_text_fn, accept_mcq):
        oe_accuracy_scores = []
        for generated_text, answers in zip(generated_texts_unique, answers_unique):
            is_mcq = answers[0] in MCQ_POSSIBLE_CHOICES and accept_mcq
            if is_mcq:
                generated_text_extracted_answer_candidates = [normalize_text_fn(generated_text)]

            else:
                generated_text_extracted_answer_candidates = parse_open_response_mmmu(
                    generated_text, normalize_text_fn
                )
            answers = [normalize_text_fn(answer) for answer in answers]
            correct = 0
            for answer in answers:
                for generated_answer_candidate in generated_text_extracted_answer_candidates:
                    if isinstance(answer, str) and isinstance(generated_answer_candidate, str):
                        # In the case of an mcq question, there is only one answer, and the answer has to be exact.
                        if is_mcq and generated_answer_candidate == answer:
                            correct = 1
                            break
                        elif answer in generated_answer_candidate:
                            correct = 1
                            break
                    # If it's a number, it has been converted to a float rounded to 2 decimals
                    elif (
                        isinstance(answer, float)
                        and isinstance(generated_answer_candidate, float)
                        and generated_answer_candidate == answer
                    ):
                        correct = 1
                        break
                    else:
                        pass  # This is the case of a number and a string, we don't want to compare them

            oe_accuracy_scores.append(correct)

        return oe_accuracy_scores

    def _compute_relaxed_vqa_accuracy(self, generated_texts_unique, answers_unique, normalize_text_fn):
        """
        From https://aclanthology.org/2022.findings-acl.177.pdf
        We use a relaxed accuracy measure for the numeric answers to allow a minor inaccuracy that may result from the automatic data extraction process. We consider an answer to be correct if it is within 5% of the gold answer. For non-numeric answers, we still need an exact match to consider an answer to be correct.
        """
        oe_accuracy_scores = []
        for generated_text, answers in zip(generated_texts_unique, answers_unique):
            generated_text = normalize_text_fn(generated_text)
            answers = [normalize_text_fn(a) for a in answers]
            correct = 0
            for answer in answers:
                if check_is_number(answer):
                    if check_is_number(generated_text):
                        generated_text_f = convert_to_number(generated_text)
                        answer_f = convert_to_number(answer)
                        if answer_f != 0.0:
                            correct = abs(generated_text_f - answer_f) / answer_f < 0.05 or correct
                        else:
                            correct = generated_text_f == answer_f
                        break
                elif generated_text == answer:
                    correct = 1
            oe_accuracy_scores.append(correct)
        return oe_accuracy_scores

    def _compute(self, example_ids, generated_texts, answers, buckets):
        data_per_id = {}
        for ex_id, gen_text, ans, bucket in zip(example_ids, generated_texts, answers, buckets):
            # If condition is a dirty trick to handle the case of distributed evaluation where some instances can be
            # repeated over a few proceses to make the batches even.
            # In this case, we just verify that all processes predicted the same thing, and only take one copy of predictions
            # in order to not mess up metrics. Ideally this "unique" logic should be handled outside of the metric or maybe
            # in the add_batch call...
            if ex_id not in data_per_id:
                data_per_id[ex_id] = {
                    "generated_text": gen_text,
                    "answers": ans,
                    "bucket": bucket,
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

        # assert list(range(len(data_per_id))) == sorted(data_per_id.keys())
        generated_texts_unique = [data_per_id[i]["generated_text"] for i in set(example_ids)]
        answers_unique = [data_per_id[i]["answers"] for i in set(example_ids)]
        results = {}
        default_to_save_generations = (
            answers_unique[0] is None or answers_unique[0][0] == ""
        ) and OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS in self.metrics
        if self.save_generations or default_to_save_generations:
            # If answers are None, we default to the server results
            if (
                OEVQAMetrics.OE_MMMU_STYLE_VQA_ACCURACY in self.metrics
                or OEVQAMetrics.OE_ONLY_MMMU_STYLE_VQA_ACCURACY in self.metrics
            ):
                results["server_results"] = [
                    {
                        "question_id": ex_id,
                        "answer": data["generated_text"],
                    }
                    for ex_id, data in data_per_id.items()
                ]
            else:
                results["server_results"] = [
                    {
                        "question_id": ex_id,
                        "answer": self.gt_vision_lab_normalization.vqa_normalize_text(data["generated_text"]),
                    }
                    for ex_id, data in data_per_id.items()
                ]

        if default_to_save_generations:
            return results

        if OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS_MMVET in self.metrics:
            results["server_results"] = {ex_id: data["generated_text"] for ex_id, data in data_per_id.items()}
            return results
        elif OEVQAMetrics.DEFAULT_TO_SERVER_RESULTS_LLAVA_WILD in self.metrics:
            results["server_results"] = [
                {
                    "question_id": ex_id,
                    "answer": data["generated_text"],
                }
                for ex_id, data in data_per_id.items()
            ]
            return results

        # VQA Accuracy
        # From "VQA: Visual Question Answering" paper:
        # an answer is deemed 100% accurate if at least 3 workers provided that exact answer. 2 Before comparison,
        #  all responses are made lowercase, numbers converted to digits, and punctuation & articles removed.
        if (
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY in self.metrics
            or OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY in self.metrics
        ):
            gt_vision_lab_first_word_vqa_accuracy_scores, get_visison_lab_oe_accuracy_scores = (
                self._compute_vqa_accuracy(
                    generated_texts_unique, answers_unique, self.gt_vision_lab_normalization.vqa_normalize_text
                )
            )

            if OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY in self.metrics:
                results["gt_vision_lab_first_word_vqa_accuracy"] = float(
                    sum(gt_vision_lab_first_word_vqa_accuracy_scores)
                    / len(gt_vision_lab_first_word_vqa_accuracy_scores)
                )

            if OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY in self.metrics:
                results["gt_vision_lab_oe_vqa_accuracy"] = float(
                    sum(get_visison_lab_oe_accuracy_scores) / len(get_visison_lab_oe_accuracy_scores)
                )
                confidence_level = 0.95
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
                std_dev = np.std(get_visison_lab_oe_accuracy_scores)
                results["gt_vision_lab_oe_vqa_accuracy_std"] = std_dev
                results["gt_vision_lab_oe_vqa_accuracy_margin_of_error"] = z_score * (
                    std_dev / np.sqrt(len(get_visison_lab_oe_accuracy_scores))
                )

        if (
            OEVQAMetrics.OE_MMMU_STYLE_PER_BUCKET_ACCURACY in self.metrics
            or OEVQAMetrics.OE_ONLY_MMMU_STYLE_PER_BUCKET_ACCURACY in self.metrics
        ):
            if (
                OEVQAMetrics.OE_MMMU_STYLE_PER_BUCKET_ACCURACY in self.metrics
                and OEVQAMetrics.OE_ONLY_MMMU_STYLE_PER_BUCKET_ACCURACY in self.metrics
            ):
                raise ValueError(
                    "Cannot compute both OE_MMMU_STYLE_PER_BUCKET_ACCURACY and OE_ONLY_MMMU_STYLE_PER_BUCKET_ACCURACY"
                    " at the same time."
                )

            # Here each bucket has the form "bucket_col_0=sub_bucket_name_x/bucket_col_1=sub_bucket_name_y/... etc."
            buckets_aggregated = [data_per_id[example_id]["bucket"] for example_id in set(example_ids)]

            # Get columns and unique buckets
            unique_buckets = set(buckets_aggregated)
            bucket_columns = [column_buckets.split("=")[0] for column_buckets in buckets_aggregated[0].split("/")]

            # Initialize the scores dict
            scores_dict = {}
            for bucket_column in bucket_columns:
                scores_dict[bucket_column] = {}
            for unique_bucket in unique_buckets:
                column_sub_bucket_names = [column_bucket.split("=")[1] for column_bucket in unique_bucket.split("/")]
                for bucket_column, sub_bucket_name in zip(bucket_columns, column_sub_bucket_names):
                    scores_dict[bucket_column][sub_bucket_name] = []

            # Need np array to use .where
            generated_texts_unique_np = np.array(generated_texts_unique)
            answers_unique_np = np.array(answers_unique, dtype=object)
            buckets_aggregated = np.array(buckets_aggregated)
            for b_ in unique_buckets:
                # Find the positions of the unique_bucket in the buckets_aggregated to compute the scores
                bucket_position = np.where(buckets_aggregated == b_)[0]
                oe_mmmu_style_bucket_scores = self._compute_mmmu_style_vqa_accuracy(
                    generated_texts_unique_np[bucket_position],
                    answers_unique_np[bucket_position],
                    normalize_str_mmmu,
                    # Do not accept mcq when using OE_ONLY_MMMU_STYLE_PER_BUCKET_ACCURACY metric
                    accept_mcq=OEVQAMetrics.OE_ONLY_MMMU_STYLE_PER_BUCKET_ACCURACY not in self.metrics,
                )

                # Each sub_bucket (column, name) pair from this buckets_aggregated entry
                # extends the list of scores of its corresponding entry in the scores_dict
                # with the oe_mmmu_style_bucket_scores.
                sub_buckets_tuples = [
                    (column_bucket.split("=")[0], column_bucket.split("=")[1]) for column_bucket in b_.split("/")
                ]
                for sub_bucket_col, sub_bucket_name in sub_buckets_tuples:
                    scores_dict[sub_bucket_col][sub_bucket_name].extend(oe_mmmu_style_bucket_scores)

            for key, value in scores_dict.items():
                for k, v in value.items():
                    scores_dict[key][k] = {"accuracy": float(sum(v) / len(v)), "std": np.std(v)}

            if OEVQAMetrics.OE_ONLY_MMMU_STYLE_PER_BUCKET_ACCURACY in self.metrics:
                results["oe_only_mmmu_style_per_bucket_accuracy"] = scores_dict
            elif OEVQAMetrics.OE_MMMU_STYLE_PER_BUCKET_ACCURACY in self.metrics:
                results["oe_mmmu_style_per_bucket_accuracy"] = scores_dict

        if OEVQAMetrics.OE_MMMU_STYLE_VQA_ACCURACY in self.metrics:
            oe_mmmu_style_accuracy_scores = self._compute_mmmu_style_vqa_accuracy(
                generated_texts_unique, answers_unique, normalize_str_mmmu, accept_mcq=True
            )

            results["oe_mmmu_style_vqa_accuracy"] = float(
                sum(oe_mmmu_style_accuracy_scores) / len(oe_mmmu_style_accuracy_scores)
            )
        if OEVQAMetrics.OE_ONLY_MMMU_STYLE_VQA_ACCURACY in self.metrics:
            oe_mmmu_style_accuracy_scores = self._compute_mmmu_style_vqa_accuracy(
                generated_texts_unique, answers_unique, normalize_str_mmmu, accept_mcq=False
            )

            results["oe_only_mmmu_style_vqa_accuracy"] = float(
                sum(oe_mmmu_style_accuracy_scores) / len(oe_mmmu_style_accuracy_scores)
            )

        if OEVQAMetrics.OE_RELAXED_VQA_ACCURACY in self.metrics:
            oe_relaxed_vqa_accuracy = self._compute_relaxed_vqa_accuracy(
                generated_texts_unique, answers_unique, lambda txt: txt.strip(".")
            )

            results["oe_relaxed_vqa_accuracy"] = float(sum(oe_relaxed_vqa_accuracy) / len(oe_relaxed_vqa_accuracy))

        if OEVQAMetrics.FIRST_WORD_VQA_ACCURACY in self.metrics or OEVQAMetrics.OE_VQA_ACCURACY in self.metrics:
            first_word_vqa_accuracy_scores, oe_accuracy_scores = self._compute_vqa_accuracy(
                generated_texts_unique, answers_unique, vqa_normalize_text
            )

            if OEVQAMetrics.FIRST_WORD_VQA_ACCURACY in self.metrics:
                results["first_word_vqa_accuracy"] = float(
                    sum(first_word_vqa_accuracy_scores) / len(first_word_vqa_accuracy_scores)
                )
            if OEVQAMetrics.OE_VQA_ACCURACY in self.metrics:
                results["oe_vqa_accuracy"] = float(sum(oe_accuracy_scores) / len(oe_accuracy_scores))
        return results
