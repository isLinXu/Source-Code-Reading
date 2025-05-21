import logging
import string
from enum import Enum
from typing import List

import datasets
import evaluate
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from m4.sourcing.data_collection.processors import FilteringFunctions
from m4.sourcing.data_collection.utils.filtering_utils import other_special_characters


_DESCRIPTION = ""
_CITATION = ""
_KWARGS_DESCRIPTION = ""

logger = logging.getLogger(__name__)


class ImageCaptioningMetrics(Enum):
    BLEU_4 = ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
    CIDER = "CIDEr"
    METEOR = "METEOR"
    ROUGE_L = "ROUGE_L"
    SPICE = "SPICE"
    EXACT_MATCH = "EXACT_MATCH"
    DEFAULT_TO_SERVER_RESULTS = "default_to_server_results"


class ExactMatch:
    def process_text(self, text, keep_only_first_word=False):
        # For our case, we are stripping the punctuations to simplify the comparison as in generative settings we only want the last word without any `.` for example.
        # Note that this is not a fair comparison setting to the prior work which would predict the punctuations along with the word as the exact match is critical for OCR applications.
        text = text.lower()
        text = FilteringFunctions.standardize_whitespace(text)
        if keep_only_first_word:
            words = FilteringFunctions.get_words_from_text(
                text,
                lower_case=True,
                strip_words=True,
                strip_characters=set(string.punctuation + string.whitespace + other_special_characters),
            )
            if words:
                text = words[0]
            else:
                text = ""
        return text

    def compute_score(self, generated_captions, reference_captions):
        score = 0
        for gen_cap, ref_caps in zip(generated_captions, reference_captions):
            gen_cap = self.process_text(text=gen_cap, keep_only_first_word=True)
            ref_caps = [self.process_text(text=ref_cap) for ref_cap in ref_caps]
            if gen_cap in ref_caps:
                score += 1
        if len(generated_captions) > 0:
            score = score / len(generated_captions)
        return score


# We separate the scorers that are implemented in pycocoevalcap from the ones that are not because they don't have the
# same API
PYCOCOEVAL_SCORERS_MAP = {
    ImageCaptioningMetrics.BLEU_4: (Bleu, {"n": 4}),
    ImageCaptioningMetrics.CIDER: (Cider, {}),
    ImageCaptioningMetrics.METEOR: (Meteor, {}),
    ImageCaptioningMetrics.ROUGE_L: (Rouge, {}),
    ImageCaptioningMetrics.SPICE: (Spice, {}),
}

OTHER_SCORERS_MAP = {
    ImageCaptioningMetrics.EXACT_MATCH: (ExactMatch, {}),
}


def instantiate_scorers(metrics, scorers_map):
    scorers = [
        (scorers_map[metric][0](**scorers_map[metric][1]), metric.value) for metric in metrics if metric in scorers_map
    ]
    return scorers


class UnfoldedImageCaptioningMetrics(evaluate.Metric):
    """This class takes care of computing the metrics listed in `metrics`."""

    def __init__(self, metrics: List[ImageCaptioningMetrics], save_generations: bool, **kwargs):
        super().__init__(**kwargs)
        self.metrics = metrics
        self.save_generations = save_generations

        self.pycoco_scorers = instantiate_scorers(self.metrics, PYCOCOEVAL_SCORERS_MAP)
        self.other_scorers = instantiate_scorers(self.metrics, OTHER_SCORERS_MAP)

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
                    "generated_captions": datasets.Value("string"),
                    "reference_captions": datasets.Sequence(datasets.Value("string")),
                }
            ),
        )

    def _compute(self, example_ids, generated_captions, reference_captions):
        data_per_id = {}

        for ex_id, gen_cap, ref_caps in zip(example_ids, generated_captions, reference_captions):
            # If condition is a dirty trick to handle the case of distributed evaluation where some instances can be
            # repeated over a few proceses to make the batches even.
            # In this case, we just verify that all processes predicted the same thing, and only take one copy of predictions
            # in order to not mess up metrics. Ideally this "unique" logic should be handled outside of the metric or maybe
            # in the add_batch call...
            if ex_id not in data_per_id:
                data_per_id[ex_id] = {
                    "generated_caption": gen_cap,
                    "reference_captions": ref_caps,
                }
            else:
                if data_per_id[ex_id]["generated_caption"] == gen_cap:
                    logger.warning(
                        f"Example {ex_id} has different predictions accross processes. We have: {gen_cap} and"
                        f" {data_per_id[ex_id]['generated_caption']}"
                    )
                if data_per_id[ex_id]["reference_captions"] == ref_caps:
                    logger.warning(
                        f"Example {ex_id} has different answers accross processes. We have: {ref_caps} and"
                        f" {data_per_id[ex_id]['reference_captions']}"
                    )

        # assert list(range(len(data_per_id))) == sorted(data_per_id.keys())

        results = {}
        default_to_save_generations = (
            reference_captions[0] is None or len(reference_captions[0]) == 0
        ) and ImageCaptioningMetrics.DEFAULT_TO_SERVER_RESULTS in self.metrics

        if self.save_generations or default_to_save_generations:
            # If answers are None, we default to the server results
            results["server_results"] = [
                {
                    "image_id": ex_id,
                    "caption": data["generated_caption"],
                }
                for ex_id, data in data_per_id.items()
            ]

        if default_to_save_generations:
            return results

        # We put the results in the format expected by the tokenizer of pycocoevalcap
        gts = {}
        res = {}
        caption_counter = 0
        for ex_id, data_dict in data_per_id.items():
            res[ex_id] = [{"image_id": ex_id, "caption": data_dict["generated_caption"], "id": caption_counter}]
            caption_counter += 1
            gts[ex_id] = [
                {"image_id": ex_id, "caption": ref_str, "id": caption_counter + idx}
                for idx, ref_str in enumerate(data_dict["reference_captions"])
            ]
            caption_counter += len(data_dict["reference_captions"])

        if len(self.pycoco_scorers) > 0:
            tokenizer = PTBTokenizer()

            gts = tokenizer.tokenize(gts)
            res = tokenizer.tokenize(res)

            for scorer, method in self.pycoco_scorers:
                score, scores = scorer.compute_score(gts, res)
                if type(method) == list:
                    for sc, scs, m in zip(score, scores, method):
                        results[f"{m}"] = sc
                        results[f"{m}_all"] = convert_to_list(scs)
                else:
                    results[f"{method}"] = score
                    results[f"{method}_all"] = convert_to_list(scores)

        if len(self.other_scorers) > 0:
            for scorer, method in self.other_scorers:
                generated_captions = [data_per_id[ex_id]["generated_caption"] for ex_id in data_per_id]
                reference_captions = [data_per_id[ex_id]["reference_captions"] for ex_id in data_per_id]
                score = scorer.compute_score(generated_captions, reference_captions)
                results[f"{method}"] = score

        return results


def convert_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    assert isinstance(obj, list), f"Expected list or np.ndarray, got {type(obj)}"
    return obj
