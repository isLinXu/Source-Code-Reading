import random
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset

from m4.evaluation.config import ShotSelectionMode
from m4.evaluation.custom_metrics.classification_vqa_metrics import ClassifVQAMetrics
from m4.evaluation.vqa_labels import _VQA_ANSWERS
from m4.models.vgpt2.evaluation_classification_in_context_vgpt2 import Vgpt2ClassificationInContext


class Vgpt2ClassificationVQAInContext(Vgpt2ClassificationInContext):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def get_info_from_dataset(self, dataset):
        _str2int_mapping = {answer: i for i, answer in enumerate(_VQA_ANSWERS)}
        self.class_names = _VQA_ANSWERS
        self.class_str2int = lambda s: _str2int_mapping[s]
        self.class_int2str = lambda i: _VQA_ANSWERS[i]
        self.class_ids = [self.class_str2int(class_name) for class_name in self.class_names]

    def prepare_dataset(self, exs: Dict, **kwargs) -> Dict:
        """
        Prepare batch of examples.
        Each example (X, y) where y is among (y1, y2, ..., yN) - the labels options -
        is turned into [(X, y1), (X, y2), ... (X, yN)].
        """
        support_dataset: Dataset = kwargs["support_dataset"]
        support_dataset_vision_encoder_embeddings: Optional[np.ndarray] = kwargs.get(
            "support_dataset_vision_encoder_embeddings", None
        )
        num_shots: int = kwargs["num_shots"]
        shot_selection_mode: ShotSelectionMode = kwargs["shot_selection_mode"]

        nb_exs = len(exs["id"])

        def retrieve_idx_closest_examples(ref_embedding, embeddings_to_compare, num_examples):
            "Returns the indices of the `num_examples` closest embeddings in ascending order"
            sim = np.dot(embeddings_to_compare, ref_embedding)
            # We can achieve linear complexity because we don't need to sort all the numbers,
            # but only find the `num_examples` largest ones
            idx_closest_ex = np.argpartition(sim, -num_examples)[-num_examples:]
            idx_closest_ex = idx_closest_ex[np.argsort(sim[idx_closest_ex])].tolist()
            return idx_closest_ex

        if (shot_selection_mode == ShotSelectionMode.random) or (num_shots == 0):
            idx_shots = [random.sample(range(len(support_dataset)), num_shots) for _ in range(nb_exs)]
        else:
            idx_shots = [
                retrieve_idx_closest_examples(ref_embedding, support_dataset_vision_encoder_embeddings, num_shots)
                for ref_embedding in exs["vision_encoder_embeddings"]
            ]

        # Prepare text shots
        texts_shots = [
            "".join(
                [
                    self._create_prompt(
                        question=support_dataset[idx_shot][self.question_column_name],
                        answer=Counter(support_dataset[idx_shot][self.answers_column_name]).most_common(1)[0][0],
                    )
                    for idx_shot in idx_shots_ex
                ]
            )
            for idx_shots_ex in idx_shots
        ]
        texts_shots = texts_shots * len(
            self.class_names
        )  # These are the priming text shots - size: batch_size * nb_of_labels
        tested_label_prompts = [
            self._create_prompt(question=question, answer=class_name)
            for class_name in self.class_names
            for question in exs[self.question_column_name]
        ]  # These are the tested labels - size: batch_size * nb_of_labels
        tot_texts = [
            text_shot + tested_label_prompt
            for text_shot, tested_label_prompt in zip(texts_shots, tested_label_prompts)
        ]  # These are the concatenation of the priming text shots and tested labels - size: batch_size * nb_of_labels
        # Ignoring their associated priming shots, the list has the following order: [x1,A; x2,A; ... xN,A; x1,B; x2,B; ...]

        tot_texts = [text.strip() for text in tot_texts]
        # Tokenize and masks
        tokens = self.tokenizer(
            tot_texts,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer_max_seq_len,
            padding=True,
        )
        input_ids = [tokens.input_ids[idx] for idx in range(len(tot_texts))]
        attention_mask = [tokens.attention_mask[idx] for idx in range(len(tot_texts))]

        # Prepare image shots
        pixel_values_shots = [
            [self.image_transform(support_dataset[idx_shot][self.image_column_name]) for idx_shot in idx_shots_ex]
            for idx_shots_ex in idx_shots
        ]  # These are the priming image shots - size: batch_size
        tested_pixel_values = [
            self.image_transform(img) for img in exs[self.image_column_name]
        ]  # These are the tested images - size: batch_size

        tot_pixel_values_not_duplicated = []
        tot_pixel_attention_masks_no_duplicated = []
        for pv_shots, pv in zip(pixel_values_shots, tested_pixel_values):
            num_images = len(pv_shots) + 1  # 1 for pv
            max_height = max([im.size(1) for im in pv_shots] + [pv.size(1)])
            max_width = max([im.size(2) for im in pv_shots] + [pv.size(2)])
            padded_image_tensor = torch.zeros(num_images, 3, max_height, max_width)
            padded_pixel_attention_masks = torch.zeros(num_images, max_height, max_width, dtype=torch.bool)

            for idx, im in enumerate(pv_shots):
                im_height, im_width = im.size(1), im.size(2)
                padded_image_tensor[idx, :, :im_height, :im_width] = im
                padded_pixel_attention_masks[idx, :im_height, :im_width] = True
            pv_height, pv_width = pv.size(1), pv.size(2)
            padded_image_tensor[-1, :, :pv_height, :pv_width] = pv
            padded_pixel_attention_masks[-1, :pv_height, :pv_width] = True

            tot_pixel_values_not_duplicated.append(padded_image_tensor)
            tot_pixel_attention_masks_no_duplicated.append(padded_pixel_attention_masks)

        pixel_values = tot_pixel_values_not_duplicated * len(self.class_names)  # size: batch_size * nb_of_labels
        pixel_attention_masks = tot_pixel_attention_masks_no_duplicated * len(self.classes_names)

        example_ids: List[int] = exs["id"] * len(self.class_names)
        true_labels: List[List[str]] = exs[self.answers_column_name] * len(self.class_names)
        tested_labels: List[str] = [
            self.class_int2str(idx_class_name)
            for idx_class_name in range(len(self.class_names))
            for _ in range(nb_exs)
        ]

        return {
            "example_ids": example_ids,
            "true_labels": true_labels,
            "tested_labels": tested_labels,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_attention_masks": pixel_attention_masks,
        }

    def _create_prompt(self, question, answer=""):
        return (
            f"{self.token_around_image}{self.image_token}{self.token_around_image}Question:"
            f" {question} Answer: {answer}"
        )


class VQAv2Vgpt2ClassificationVQAInContextAcc(Vgpt2ClassificationVQAInContext):
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


class VQAv2SampleVgpt2ClassificationVQAInContextAcc(VQAv2Vgpt2ClassificationVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Sample"
