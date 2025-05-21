import logging
from enum import Enum
from typing import List

import datasets
import evaluate
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, ndcg_score, roc_auc_score


_DESCRIPTION = ""
_CITATION = ""
_KWARGS_DESCRIPTION = ""
logger = logging.getLogger(__name__)


class ClassifMetrics(Enum):
    KL_DISTRIBUTION = "kl_distribution"
    KL_MEAN = "kl_mean"
    ENTROPY_DISTRIBUTION = "entropy_distribution"
    ENTROPY_MEAN = "entropy_mean"
    ACCURACY = "accuracy"
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"
    F1_SCORE = "f1_score"
    ROC_AUC = "roc_auc"
    PER_BUCKET_ACCURACY = "per_bucket_accuracy"
    NDCG = "NDCG"
    DEFAULT_TO_SERVER_RESULTS = "default_to_server_results"


class UnfoldedClassificationMetrics(evaluate.Metric):
    """
    For each example, there are N classes. One line is the pair (example,class) which means
    there are N lines for each example.
    This class takes care of aggregating predictions per example and computing the metrics listed in
    `metrics`.
    `bucket` is an optional argument that can be used to aggregate predictions per bucket.
    A bucket is typically a certain slice of the dataset (for instance, all instances where age=30).
    """

    def __init__(self, metrics: List[ClassifMetrics], **kwargs):
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
                    "predictions": datasets.Value("float64"),
                    "example_ids": datasets.Value("string"),
                    "true_labels": datasets.Value("int64"),
                    "tested_labels": datasets.Value("int64"),
                    "relevance_scores": datasets.Value("float64"),
                    "buckets": datasets.Value("string"),
                }
            ),
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html"],
        )

    def _compute(
        self,
        predictions,
        example_ids,
        true_labels,
        tested_labels,
        relevance_scores,
        buckets=None,
        normalize=True,
        sample_weight=None,
        tol=0.001,
    ):
        data_per_id = {}
        if buckets is None:
            buckets = [None] * len(example_ids)
        for (
            example_id,
            prediction,
            true_label,
            tested_label,
            relevance_score,
            bucket,
        ) in zip(example_ids, predictions, true_labels, tested_labels, relevance_scores, buckets):
            if example_id not in data_per_id:
                data_per_id[example_id] = {
                    "predictions": [],
                    "tested_labels": [],
                    "relevance_scores": [],
                }
            # If condition is a dirty trick to handle the case of distributed evaluation where some instances can be
            # repeated over a few proceses to make the batches even.
            # In this case, we just verify that all processes predicted the same thing, and only take one copy of predictions
            # in order to not mess up metrics. Ideally this "unique" logic should be handled outside of the metric or maybe
            # in the add_batch call...
            if tested_label in data_per_id[example_id]["tested_labels"]:
                idx_already_present = data_per_id[example_id]["tested_labels"].index(tested_label)
                # It happens in practice that different predictions for the same `example_id` differ by
                # a tiny bit, hence the use of a tolerance to validate the `assert`
                difference = abs(data_per_id[example_id]["predictions"][idx_already_present] - prediction)
                logger.warning(
                    f"prediction already present: {data_per_id[example_id]['predictions'][idx_already_present]}, new"
                    f" prediction: {prediction}, Difference: {difference}"
                )
                assert data_per_id[example_id]["true_label"] == true_label
                assert data_per_id[example_id]["bucket"] == bucket
            else:
                data_per_id[example_id]["predictions"].append(prediction)
                data_per_id[example_id]["true_label"] = true_label
                data_per_id[example_id]["tested_labels"].append(tested_label)
                data_per_id[example_id]["relevance_scores"].append(relevance_score)
                data_per_id[example_id]["bucket"] = bucket
        # assert list(range(len(data_per_id))) == sorted(data_per_id.keys())

        results = {}

        default_to_save_generations = (
            true_labels[0] == -1
        ) and ClassifMetrics.DEFAULT_TO_SERVER_RESULTS in self.metrics
        if default_to_save_generations:
            # If there is no answers, we default to the server results
            results["server_results"] = [
                {
                    "id": ex_id,
                    "label": data["tested_labels"][np.argmax(data["predictions"])],
                }
                for ex_id, data in data_per_id.items()
            ]
            return results

        if (
            ClassifMetrics.ACCURACY in self.metrics
            or ClassifMetrics.MEAN_PER_CLASS_ACCURACY in self.metrics
            or ClassifMetrics.ROC_AUC in self.metrics
            or ClassifMetrics.PER_BUCKET_ACCURACY in self.metrics
            or ClassifMetrics.NDCG in self.metrics
        ):
            references = []
            buckets_aggregated = []
            top1_predictions = []
            all_predictions = []
            all_relevance_scores = []
            for example_id in data_per_id.keys():
                idx = np.argmax(data_per_id[example_id]["predictions"])
                references.append(data_per_id[example_id]["true_label"])
                buckets_aggregated.append(data_per_id[example_id]["bucket"])
                top1_predictions.append(data_per_id[example_id]["tested_labels"][idx])
                all_predictions.append(data_per_id[example_id]["predictions"])
                all_relevance_scores.append(data_per_id[example_id]["relevance_scores"])

        # Top-1 Accuracy
        if ClassifMetrics.ACCURACY in self.metrics:
            results["accuracy"] = float(
                accuracy_score(references, top1_predictions, normalize=normalize, sample_weight=sample_weight)
            )

        # Mean Per Class Accuracy
        if ClassifMetrics.MEAN_PER_CLASS_ACCURACY in self.metrics:
            # Technically, `num_classes` should be an argument/attribute, and not computed from the references, but
            # IF references cover all the classes, it should be equivalent
            classes = set(references)
            accuracy_per_class = {}
            references = np.array(references)
            top1_predictions = np.array(top1_predictions)
            for c_ in classes:
                class_position = find_positions(references, c_)
                accuracy_per_class[c_] = accuracy_score(
                    references[class_position],
                    top1_predictions[class_position],
                    normalize=normalize,
                )
            results["mean_per_class_accuracy"] = sum(accuracy_per_class.values()) / len(classes)

        if ClassifMetrics.PER_BUCKET_ACCURACY in self.metrics:
            # Per bucket accuracy
            unique_buckets = set(buckets_aggregated)
            accuracy_per_bucket = {}
            references = np.array(references)
            top1_predictions = np.array(top1_predictions)
            buckets_aggregated = np.array(buckets_aggregated)
            for b_ in unique_buckets:
                bucket_position = find_positions(buckets_aggregated, b_)
                accuracy_per_bucket[b_] = accuracy_score(
                    references[bucket_position],
                    top1_predictions[bucket_position],
                    normalize=normalize,
                )
            results["per_bucket_accuracy"] = accuracy_per_bucket
            results["std_per_bucket_accuracy"] = np.std(list(accuracy_per_bucket.values()))

        if ClassifMetrics.F1_SCORE in self.metrics:
            results["f1_score"] = float(f1_score(references, top1_predictions, sample_weight=sample_weight))

        if ClassifMetrics.NDCG in self.metrics:
            results["NDCG"] = float(ndcg_score(all_relevance_scores, all_predictions))

        # KL-Divergence
        if ClassifMetrics.KL_DISTRIBUTION in self.metrics or ClassifMetrics.KL_MEAN in self.metrics:
            # Source: https://machinelearningmastery.com/divergence-between-probability-distributions/
            # If we are attempting to approximate an unknown probability distribution, then the target probability
            # distribution from data is P and Q is our approximation of the distribution.
            # KL(P || Q) = – sum x in X P(x) * log(Q(x) / P(x))
            # In the case of classification, KL and Cross-Entropy are equivalent.
            kl_scores = []
            for example_id in data_per_id.keys():
                q = softmax(np.array(data_per_id[example_id]["predictions"]))
                idx_true_label = data_per_id[example_id]["tested_labels"].index(data_per_id[example_id]["true_label"])
                kl = -np.log(q[idx_true_label])
                kl_scores.append(kl)

        if ClassifMetrics.KL_DISTRIBUTION in self.metrics:
            results["kl_distribution"] = kl_scores

        if ClassifMetrics.KL_MEAN in self.metrics:
            results["kl_mean"] = float(np.mean(kl_scores))

        # Entropy
        if ClassifMetrics.ENTROPY_DISTRIBUTION in self.metrics or ClassifMetrics.ENTROPY_MEAN in self.metrics:
            entropy_scores = []
            for example_id in data_per_id.keys():
                q = softmax(np.array(data_per_id[example_id]["predictions"]))
                # Source https://en.wikipedia.org/wiki/Entropy_(information_theory)
                # Given a discrete random variable X, which takes values in the alphabet M and is distributed according
                # to p : X → [ 0 , 1 ]
                # H(X):=-\sum_{x \in M} p(x) \log p(x)
                entropy = -np.sum(np.log(q) * q)
                entropy_scores.append(entropy)

        if ClassifMetrics.ENTROPY_DISTRIBUTION in self.metrics:
            results["entropy_distribution"] = entropy_scores

        if ClassifMetrics.ENTROPY_MEAN in self.metrics:
            results["entropy_mean"] = float(np.mean(entropy_scores))

        if ClassifMetrics.ROC_AUC in self.metrics:
            # Compute ROC curve and ROC area for each class
            results["roc_auc_score"] = roc_auc_score(references, top1_predictions, average="macro")
        return results


def find_positions(array_to_check, item_to_find):
    indices = np.where(array_to_check == item_to_find)[0]
    return indices
