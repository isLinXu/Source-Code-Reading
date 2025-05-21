import os
import random
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from transformers.utils import ModelOutput

from m4.evaluation.config import ShotSelectionMode
from m4.evaluation.custom_metrics.unfolded_classification_metrics import ClassifMetrics
from m4.evaluation.tasks import BaseTaskClassification, Predictor
from m4.evaluation.utils import EvaluationVersion
from m4.training.packing import get_splitted_images_and_corresponding_text
from m4.training.utils import (
    FAKE_TOKEN_AROUND_IMAGE_V1,
    FAKE_TOKEN_AROUND_IMAGE_V2,
    IMAGE_TOKEN,
    build_image_transform,
)


class Vgpt2ClassificationInContext(BaseTaskClassification):
    model_class: str = "VGPT2LMHeadModel"
    predictor_class: Predictor = Predictor.in_contexter
    target_keys: List[str] = ["tested_labels", "example_ids", "true_labels", "relevance_scores"]
    buckets_keys: List[str] = []
    # Buckets are optionally populated for classification in context. They are only useful when it is useful to get results bucket. A bucket is typically a certain slice of the dataset (for instance, all instances where age=30).
    mapping_class_names_to_prompt_names: Optional[Dict[str, str]] = None
    prompt_templates_dict: Dict[int, Dict[str, str]] = {}
    prompt_templates_dict_instruct: Dict[int, Dict[str, str]] = {}
    bool_instruct_templates = False
    mapping_class_prompt_name_id_to_prompt_template_id: Optional[Dict[int, int]] = None
    tokenizer_max_seq_len = 1024

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer_name = kwargs.pop("tokenizer_name")
        evaluation_version = kwargs.pop("evaluation_version")
        tokenizer_use_fast = kwargs.pop("tokenizer_use_fast", False)
        self.vision_encoder_max_image_size = kwargs.pop("vision_encoder_max_image_size")
        vision_encoder_type = kwargs.pop("vision_encoder_type")
        self.image_seq_len = kwargs.pop("image_seq_len")
        self.image_transform = build_image_transform(
            max_image_size=self.vision_encoder_max_image_size,
            image_size=None,
            eval=True,
            vision_encoder_type=vision_encoder_type,
        )
        self.scale_up_images = kwargs.pop("scale_up_images")
        self.image_size_after_scaling = kwargs.pop("image_size_after_scaling")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name, truncation_side="left", use_fast=tokenizer_use_fast, token=os.getenv("HF_TOKEN", True)
        )
        self.tokenizer.padding_side = "left"
        self.image_token = IMAGE_TOKEN
        if evaluation_version == EvaluationVersion.v1:
            self.token_around_image = FAKE_TOKEN_AROUND_IMAGE_V1
        elif evaluation_version == EvaluationVersion.v2:
            self.token_around_image = FAKE_TOKEN_AROUND_IMAGE_V2
        else:
            raise ValueError(f"Invalid evaluation version: {evaluation_version}")

    def simpler_get_splitted_images_and_corresponding_text(self, image):
        splitted_images_array, text_splitted_images = get_splitted_images_and_corresponding_text(
            image=image,
            vision_encoder_max_image_size=self.vision_encoder_max_image_size,
            max_image_size=self.image_size_after_scaling,
            pre_split_scale_up_max=None,
            pre_split_scale_up_frequency=None,
            image_seq_len=self.image_seq_len,
            # Any value sufficiently high such that the image will always be resized to max_image_size
            scale_up_factor=100 if self.scale_up_images else 1,
        )
        return splitted_images_array, text_splitted_images

    def get_info_from_dataset(self, dataset):
        self.class_names = dataset.features[self.label_column_name].names
        self.class_str2int = dataset.features[self.label_column_name].str2int
        self.class_int2str = dataset.features[self.label_column_name].int2str
        self.class_ids = [self.class_str2int(class_name) for class_name in self.class_names]

    def get_data_collator(self, **kwargs):
        def data_collator(batch):
            exs = {key: [ex[key] for ex in batch] for key in batch[0].keys()}
            batch = self.prepare_dataset(exs, **kwargs)
            return batch

        return data_collator

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
        prompt_template_id: int = kwargs["prompt_template_id"]

        # Apply mapping from class names to prompt names
        prompted_class_names = sorted(
            set([self._get_class_name_value(prompt_template_id, class_name) for class_name in self.class_names])
        )
        class_prompt2int = {
            self._get_class_name_value(prompt_template_id, class_name): self.class_str2int(class_name)
            for class_name in self.class_names
        }
        class_int2prompt = {
            self.class_str2int(class_name): self._get_class_name_value(prompt_template_id, class_name)
            for class_name in self.class_names
        }

        nb_exs = len(exs["id"])
        # If the first image column is a list. We use it as the only image column, and <image> tokens are hardcoded in the dataset
        multiple_images_in_single_column = isinstance(support_dataset[0][self.image_column_names[0]], list)
        if multiple_images_in_single_column and len(self.image_column_names) > 1:
            raise ValueError(
                "We can either have multiple image columns, or multiple images in one column but not both"
            )

        if not self.tested_labels_column_name:
            nb_tested_labels_per_ex = len(prompted_class_names)
            tested_labels_exs = [[class_name for _ in range(nb_exs)] for class_name in prompted_class_names]
            tested_labels: List[int] = [
                class_prompt2int[label] for _tested_label in tested_labels_exs for label in _tested_label
            ]
        else:
            nb_tested_labels_per_ex = len(exs[self.tested_labels_column_name][0])
            tested_labels_exs = [
                [exs[self.tested_labels_column_name][idx_ex][idx_class] for idx_ex in range(nb_exs)]
                for idx_class in range(nb_tested_labels_per_ex)
            ]
            tested_labels: List[int] = [
                class_prompt2int[exs[self.tested_labels_column_name][idx_ex][idx_class]]
                for idx_class in range(nb_tested_labels_per_ex)
                for idx_ex in range(nb_exs)
            ]

        if self.relevance_scores_column_name:
            relevance_scores = [
                exs[self.relevance_scores_column_name][idx_ex][idx_class]
                for idx_class in range(nb_tested_labels_per_ex)
                for idx_ex in range(nb_exs)
            ]
        else:
            # Fake variable to match the common signature
            relevance_scores = [0.0] * nb_exs * nb_tested_labels_per_ex

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
                    self._create_example_prompt(
                        prompt_template_id=prompt_template_id,
                        class_name=class_int2prompt[support_dataset[idx_shot][self.label_column_name]],
                        images=(
                            support_dataset[idx_shot][self.image_column_names[0]]
                            if multiple_images_in_single_column
                            else [
                                support_dataset[idx_shot][image_column_name]
                                for image_column_name in self.image_column_names
                            ]
                        ),
                        multiple_images_in_single_column=multiple_images_in_single_column,
                        contexts=(
                            [
                                (context_column_name, support_dataset[context_column_name][idx_shot])
                                for context_column_name in self.context_column_names
                            ]
                            if self.context_column_names
                            else None
                        ),
                        excluded_context_columns=[],
                    )
                    for idx_shot in idx_shots_ex
                ]
            )
            for idx_shots_ex in idx_shots
        ]
        texts_shots = (
            texts_shots * nb_tested_labels_per_ex
        )  # These are the priming text shots - size: batch_size * nb_of_labels
        tested_label_prompts = [
            self._create_example_prompt(
                prompt_template_id=prompt_template_id,
                class_name=tested_labels_exs[idx_class][idx_ex],
                images=(
                    exs[self.image_column_names[0]][idx_ex]
                    if multiple_images_in_single_column
                    else [exs[image_column_name][idx_ex] for image_column_name in self.image_column_names]
                ),
                multiple_images_in_single_column=multiple_images_in_single_column,
                contexts=(
                    [
                        (context_column_name, exs[context_column_name][idx_ex])
                        for context_column_name in self.context_column_names
                    ]
                    if self.context_column_names
                    else None
                ),
                excluded_context_columns=(
                    self.tested_ex_excluded_context_columns if self.tested_ex_excluded_context_columns else []
                ),
            )
            for idx_class in range(nb_tested_labels_per_ex)
            for idx_ex in range(nb_exs)
        ]  # These are the tested labels - size: batch_size * nb_of_labels
        tot_texts = [
            self._create_prefix_prompt(prompt_template_id=prompt_template_id) + text_shot + tested_label_prompt
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
            add_special_tokens=False,
        )
        input_ids = [tokens.input_ids[idx] for idx in range(len(tot_texts))]
        attention_mask = [tokens.attention_mask[idx] for idx in range(len(tot_texts))]
        if multiple_images_in_single_column:
            pixel_values_shots = [
                [
                    self.image_transform(sub_image)
                    for idx_shot in idx_shots_ex
                    for img in support_dataset[idx_shot][self.image_column_names[0]]
                    for sub_image in self.simpler_get_splitted_images_and_corresponding_text(image=img)[0]
                ]
                for idx_shots_ex in idx_shots
            ]
        else:
            pixel_values_shots = [
                [
                    self.image_transform(sub_image)
                    for idx_shot in idx_shots_ex
                    for image_column_name in self.image_column_names
                    for sub_image in self.simpler_get_splitted_images_and_corresponding_text(
                        image=support_dataset[idx_shot][image_column_name],
                    )[0]
                ]
                for idx_shots_ex in idx_shots
            ]

        # These are the tested images - size: batch_size
        if multiple_images_in_single_column:
            tested_pixel_values = [
                [
                    self.image_transform(sub_image)
                    for image in images
                    for sub_image in self.simpler_get_splitted_images_and_corresponding_text(image=image)[0]
                ]
                for images in exs[self.image_column_names[0]]
            ]

        else:
            tested_pixel_values = [
                [
                    self.image_transform(sub_image)
                    for col in self.image_column_names
                    for sub_image in self.simpler_get_splitted_images_and_corresponding_text(image=exs[col][i])[0]
                ]
                for i in range(len(exs["id"]))
            ]

        pixel_values = []
        pixel_attention_masks = []
        for pv_shots, pv in zip(pixel_values_shots, tested_pixel_values):
            num_images = len(pv_shots) + len(pv)
            max_height = max([im.size(1) for im in pv_shots] + [im.size(1) for im in pv])
            max_width = max([im.size(2) for im in pv_shots] + [im.size(2) for im in pv])
            padded_image_tensor = torch.zeros(num_images, 3, max_height, max_width)
            padded_pixel_attention_masks = torch.zeros(num_images, max_height, max_width, dtype=torch.bool)

            for idx, im in enumerate(pv_shots + pv):
                im_height, im_width = im.size(1), im.size(2)
                padded_image_tensor[idx, :, :im_height, :im_width] = im
                padded_pixel_attention_masks[idx, :im_height, :im_width] = True

            pixel_values.append(padded_image_tensor)
            pixel_attention_masks.append(padded_pixel_attention_masks)

        pixel_values = pixel_values * nb_tested_labels_per_ex  # size: batch_size * nb_of_labels
        pixel_attention_masks = pixel_attention_masks * nb_tested_labels_per_ex

        example_ids: List[int] = exs["id"] * nb_tested_labels_per_ex

        true_labels = exs[self.label_column_name]
        # Handle the case where the true labels are not provided
        labels_are_none = all(label is None for label in true_labels)
        if not labels_are_none:
            true_labels = [
                class_prompt2int[self._get_class_name_value(prompt_template_id, self.class_int2str(label_id))]
                for label_id in true_labels
            ]
            true_labels: List[int] = true_labels * nb_tested_labels_per_ex
        else:
            true_labels: List[int] = [-1] * len(true_labels) * nb_tested_labels_per_ex
        if self.buckets_keys:

            def bucket_infos_to_str(bucket_infos):
                name = []
                for info, info_type in zip(bucket_infos, self.buckets_keys):
                    name.append(f"{info_type}={info}")
                return "/".join(name)

            columns_to_concatenate = [exs[key] for key in self.buckets_keys]
            buckets = [bucket_infos_to_str(bucket_infos) for bucket_infos in zip(*columns_to_concatenate)] * len(
                prompted_class_names
            )
        else:
            buckets = [None] * len(example_ids)

        return {
            "example_ids": example_ids,
            "true_labels": true_labels,
            "tested_labels": tested_labels,
            "relevance_scores": relevance_scores,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_attention_masks": pixel_attention_masks,
            "buckets": buckets,
        }

    def _get_class_name_value(self, prompt_template_id, class_name):
        if self.mapping_class_prompt_name_id_to_prompt_template_id is not None:
            mapping_class_names_to_prompt_names = None
            for (
                class_prompt_name_id,
                prompt_template_ids,
            ) in self.mapping_class_prompt_name_id_to_prompt_template_id.items():
                if prompt_template_id in prompt_template_ids:
                    mapping_class_names_to_prompt_names = self.mapping_class_names_to_prompt_names[
                        class_prompt_name_id
                    ]
                    break
            if mapping_class_names_to_prompt_names is None:
                raise ValueError(
                    f"Prompt template id {prompt_template_id} not found in"
                    " mapping_class_prompt_name_id_to_prompt_template_id"
                )
            return mapping_class_names_to_prompt_names[class_name]
        elif self.mapping_class_names_to_prompt_names is not None:
            return self.mapping_class_names_to_prompt_names[class_name]
        else:
            return class_name

    def _create_example_prompt(
        self,
        prompt_template_id,
        class_name,
        images,
        multiple_images_in_single_column=False,
        contexts=None,
        excluded_context_columns=[],
    ):
        if self.bool_instruct_templates:
            prompt_templates_dict = self.prompt_templates_dict_instruct
        else:
            prompt_templates_dict = self.prompt_templates_dict
        prompt_template = prompt_templates_dict[prompt_template_id]["example"]

        template_kwargs = {
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "class_name": class_name,
        }

        if contexts is not None:
            for context_column_name, additional_context in contexts:
                if context_column_name not in excluded_context_columns:
                    template_kwargs[f"{context_column_name}"] = additional_context

        # Substract the template pattern for each excluded_column
        for excluded_column in excluded_context_columns:
            pattern = r"\{" + re.escape(excluded_column) + r"\}"
            prompt_template = re.sub(pattern, "", prompt_template)

        prompt = prompt_template.format(**template_kwargs)

        if prompt.count("<image>") != len(images):
            raise ValueError(
                "Mismatch between the number of <image> tokens in the prompt and the actual number of images"
            )
        prompt = prompt.replace("<image>", "<IMAGE>")
        for image in images:
            _, text_splitted_images = self.simpler_get_splitted_images_and_corresponding_text(image=image)
            prompt = prompt.replace("<IMAGE>", text_splitted_images, 1)
        return prompt

    def _create_prefix_prompt(self, prompt_template_id):
        if self.bool_instruct_templates:
            prompt_templates_dict = self.prompt_templates_dict_instruct
        else:
            prompt_templates_dict = self.prompt_templates_dict
        prompt_template = prompt_templates_dict[prompt_template_id]["prefix"]
        if prompt_template is None:
            return ""
        else:
            prompt = prompt_template.format(
                bos_token=self.tokenizer.bos_token,
                eos_token=self.tokenizer.eos_token,
            )
        return prompt

    def predict(self, **kwargs):
        model = kwargs["model"]
        input_ids = torch.stack(kwargs["input_ids"]).to(model.device)
        attention_mask = torch.stack(kwargs["attention_mask"]).to(model.device)

        total_batch_size = len(kwargs["pixel_values"])
        max_num_images = max([i.size(0) for i in kwargs["pixel_values"]])
        max_height = max([i.size(2) for i in kwargs["pixel_values"]])
        max_width = max([i.size(3) for i in kwargs["pixel_values"]])
        pixel_values = torch.zeros(total_batch_size, max_num_images, 3, max_height, max_width)
        pixel_attention_mask = torch.zeros(total_batch_size, max_num_images, max_height, max_width, dtype=torch.bool)
        for idx, (sample_images, sample_pixel_attention_mask) in enumerate(
            zip(kwargs["pixel_values"], kwargs["pixel_attention_masks"])
        ):
            im_batch_height, im_batch_width = sample_images.size()[2:]
            pixel_values[idx, : sample_images.shape[0], :, :im_batch_height, :im_batch_width] = sample_images
            pixel_attention_mask[idx, : sample_pixel_attention_mask.shape[0], :im_batch_height, :im_batch_width] = (
                sample_pixel_attention_mask
            )
        pixel_values = pixel_values.to(model.device)
        pixel_attention_mask = pixel_attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
            )
        outputs.input_ids = input_ids
        outputs.attention_mask = attention_mask
        return outputs

    def format_model_outputs_to_predictions(self, outputs: ModelOutput) -> torch.Tensor:
        batch_size = outputs.logits.shape[0]
        # Shift so that tokens < n predict n
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = outputs.input_ids[..., 1:].contiguous()
        shift_attention_mask = outputs.attention_mask[:, 1:].contiguous()

        # Flatten the tokens
        loss_fct = CrossEntropyLoss(reduction="none")
        log_probs = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        log_probs = log_probs.view(batch_size, -1)
        masked_log_probs = shift_attention_mask * log_probs
        score_per_example = masked_log_probs.sum(dim=-1)
        if self.length_normalize:
            score_per_example = score_per_example / shift_attention_mask.sum(dim=-1)

        return score_per_example.cpu()

    def add_batch_metric(self, metric, **kwargs):
        outputs = self.predict(**kwargs)
        predictions = self.format_model_outputs_to_predictions(outputs)
        predictions = predictions.to(torch.float64)
        additional_args = {key: kwargs[key] for key in self.target_keys}
        additional_args["buckets"] = (
            kwargs["buckets"] if self.buckets_keys else [""] * len(kwargs[self.target_keys[0]])
        )
        metric.add_batch(
            predictions=predictions,
            **additional_args,
        )
        return metric


class Food101Vgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "food101"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a photo of {class_name}, a type of food.",
        },
    }


class Food101SampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    Food101Vgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/food101-Sample"


class Cifar10Vgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "cifar10"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["img"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a photo of a {class_name}.",
        },
    }


class Cifar10SampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    Cifar10Vgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/cifar10-Sample"


class Cifar10DummyVgpt2ClassificationInContextAccWithKLAndEntropy(
    Cifar10Vgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/cifar10-Dummy"


class Cifar100Vgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "cifar100"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "fine_label"
    image_column_names: List[str] = ["img"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a photo of a {class_name}.",
        },
    }


class Cifar100SampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    Cifar100Vgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/cifar100-Sample"


class StanfordCarsVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/Stanford-Cars"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a photo of a {class_name}.",
        },
    }


class StanfordCarsSampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    StanfordCarsVgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/Stanford-Cars-Sample"


class DTDVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/DTD_Describable-Textures-Dataset"
    dataset_config = "partition_1"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a photo of a {class_name} texture.",
        },
    }


class DTDSampleVgpt2ClassificationInContextAccWithKLAndEntropy(DTDVgpt2ClassificationInContextAccWithKLAndEntropy):
    dataset_name: str = "HuggingFaceM4/DTD_Describable-Textures-Dataset-partition_1-Sample"


class RenderedSST2Vgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/RenderedSST2"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a {class_name} review of a movie.",
        },
    }


class RenderedSST2SampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    RenderedSST2Vgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/RenderedSST2-Sample"


class SUN397Vgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/sun397"
    dataset_config = "standard-part1-120k"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a photo of a {class_name}.",
        },
    }


class SUN397SampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    SUN397Vgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/sun397-standard-part1-120k-Sample"
    dataset_config = None


class OxfordPetsVgpt2ClassificationInContextMeanPerClassAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/Oxford-IIIT-Pet"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.MEAN_PER_CLASS_ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>A photo of a {class_name}, a type of pet.",
        },
    }


class OxfordPetsSampleVgpt2ClassificationInContextMeanPerClassAccWithKLAndEntropy(
    OxfordPetsVgpt2ClassificationInContextMeanPerClassAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/Oxford-IIIT-Pet-Sample"
    dataset_config = None


class Caltech101Vgpt2ClassificationInContextMeanPerClassAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/Caltech-101"
    dataset_config = "with_background_category"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.MEAN_PER_CLASS_ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a photo of a {class_name}.",
        },
    }


class Caltech101SampleVgpt2ClassificationInContextMeanPerClassAccWithKLAndEntropy(
    Caltech101Vgpt2ClassificationInContextMeanPerClassAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/Caltech-101-with_background_category-Sample"
    dataset_config = None


class ImageNet1kVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "imagenet-1k"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    mapping_class_names_to_prompt_names: dict = {
        "tench, Tinca tinca": "tench",
        "goldfish, Carassius auratus": "goldfish",
        "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias": "great white shark",
        "tiger shark, Galeocerdo cuvieri": "tiger shark",
        "hammerhead, hammerhead shark": "hammerhead shark",
        "electric ray, crampfish, numbfish, torpedo": "electric ray",
        "stingray": "stingray",
        "cock": "rooster",
        "hen": "hen",
        "ostrich, Struthio camelus": "ostrich",
        "brambling, Fringilla montifringilla": "brambling",
        "goldfinch, Carduelis carduelis": "goldfinch",
        "house finch, linnet, Carpodacus mexicanus": "house finch",
        "junco, snowbird": "junco",
        "indigo bunting, indigo finch, indigo bird, Passerina cyanea": "indigo bunting",
        "robin, American robin, Turdus migratorius": "American robin",
        "bulbul": "bulbul",
        "jay": "jay",
        "magpie": "magpie",
        "chickadee": "chickadee",
        "water ouzel, dipper": "American dipper",
        "kite": "kite (bird of prey)",
        "bald eagle, American eagle, Haliaeetus leucocephalus": "bald eagle",
        "vulture": "vulture",
        "great grey owl, great gray owl, Strix nebulosa": "great grey owl",
        "European fire salamander, Salamandra salamandra": "fire salamander",
        "common newt, Triturus vulgaris": "smooth newt",
        "eft": "newt",
        "spotted salamander, Ambystoma maculatum": "spotted salamander",
        "axolotl, mud puppy, Ambystoma mexicanum": "axolotl",
        "bullfrog, Rana catesbeiana": "American bullfrog",
        "tree frog, tree-frog": "tree frog",
        "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui": "tailed frog",
        "loggerhead, loggerhead turtle, Caretta caretta": "loggerhead sea turtle",
        "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea": "leatherback sea turtle",
        "mud turtle": "mud turtle",
        "terrapin": "terrapin",
        "box turtle, box tortoise": "box turtle",
        "banded gecko": "banded gecko",
        "common iguana, iguana, Iguana iguana": "green iguana",
        "American chameleon, anole, Anolis carolinensis": "Carolina anole",
        "whiptail, whiptail lizard": "desert grassland whiptail lizard",
        "agama": "agama",
        "frilled lizard, Chlamydosaurus kingi": "frilled-necked lizard",
        "alligator lizard": "alligator lizard",
        "Gila monster, Heloderma suspectum": "Gila monster",
        "green lizard, Lacerta viridis": "European green lizard",
        "African chameleon, Chamaeleo chamaeleon": "chameleon",
        "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis": "Komodo dragon",
        "African crocodile, Nile crocodile, Crocodylus niloticus": "Nile crocodile",
        "American alligator, Alligator mississipiensis": "American alligator",
        "triceratops": "triceratops",
        "thunder snake, worm snake, Carphophis amoenus": "worm snake",
        "ringneck snake, ring-necked snake, ring snake": "ring-necked snake",
        "hognose snake, puff adder, sand viper": "eastern hog-nosed snake",
        "green snake, grass snake": "smooth green snake",
        "king snake, kingsnake": "kingsnake",
        "garter snake, grass snake": "garter snake",
        "water snake": "water snake",
        "vine snake": "vine snake",
        "night snake, Hypsiglena torquata": "night snake",
        "boa constrictor, Constrictor constrictor": "boa constrictor",
        "rock python, rock snake, Python sebae": "African rock python",
        "Indian cobra, Naja naja": "Indian cobra",
        "green mamba": "green mamba",
        "sea snake": "sea snake",
        "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus": "Saharan horned viper",
        "diamondback, diamondback rattlesnake, Crotalus adamanteus": "eastern diamondback rattlesnake",
        "sidewinder, horned rattlesnake, Crotalus cerastes": "sidewinder rattlesnake",
        "trilobite": "trilobite",
        "harvestman, daddy longlegs, Phalangium opilio": "harvestman",
        "scorpion": "scorpion",
        "black and gold garden spider, Argiope aurantia": "yellow garden spider",
        "barn spider, Araneus cavaticus": "barn spider",
        "garden spider, Aranea diademata": "European garden spider",
        "black widow, Latrodectus mactans": "southern black widow",
        "tarantula": "tarantula",
        "wolf spider, hunting spider": "wolf spider",
        "tick": "tick",
        "centipede": "centipede",
        "black grouse": "black grouse",
        "ptarmigan": "ptarmigan",
        "ruffed grouse, partridge, Bonasa umbellus": "ruffed grouse",
        "prairie chicken, prairie grouse, prairie fowl": "prairie grouse",
        "peacock": "peafowl",
        "quail": "quail",
        "partridge": "partridge",
        "African grey, African gray, Psittacus erithacus": "african grey parrot",
        "macaw": "macaw",
        "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita": "sulphur-crested cockatoo",
        "lorikeet": "lorikeet",
        "coucal": "coucal",
        "bee eater": "bee eater",
        "hornbill": "hornbill",
        "hummingbird": "hummingbird",
        "jacamar": "jacamar",
        "toucan": "toucan",
        "drake": "duck",
        "red-breasted merganser, Mergus serrator": "red-breasted merganser",
        "goose": "goose",
        "black swan, Cygnus atratus": "black swan",
        "tusker": "tusker",
        "echidna, spiny anteater, anteater": "echidna",
        "platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus": "platypus",
        "wallaby, brush kangaroo": "wallaby",
        "koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus": "koala",
        "wombat": "wombat",
        "jellyfish": "jellyfish",
        "sea anemone, anemone": "sea anemone",
        "brain coral": "brain coral",
        "flatworm, platyhelminth": "flatworm",
        "nematode, nematode worm, roundworm": "nematode",
        "conch": "conch",
        "snail": "snail",
        "slug": "slug",
        "sea slug, nudibranch": "sea slug",
        "chiton, coat-of-mail shell, sea cradle, polyplacophore": "chiton",
        "chambered nautilus, pearly nautilus, nautilus": "chambered nautilus",
        "Dungeness crab, Cancer magister": "Dungeness crab",
        "rock crab, Cancer irroratus": "rock crab",
        "fiddler crab": "fiddler crab",
        "king crab, Alaska crab, Alaskan king crab, Alaska king crab, Paralithodes camtschatica": "red king crab",
        "American lobster, Northern lobster, Maine lobster, Homarus americanus": "American lobster",
        "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish": "spiny lobster",
        "crayfish, crawfish, crawdad, crawdaddy": "crayfish",
        "hermit crab": "hermit crab",
        "isopod": "isopod",
        "white stork, Ciconia ciconia": "white stork",
        "black stork, Ciconia nigra": "black stork",
        "spoonbill": "spoonbill",
        "flamingo": "flamingo",
        "little blue heron, Egretta caerulea": "little blue heron",
        "American egret, great white heron, Egretta albus": "great egret",
        "bittern": "bittern bird",
        "crane": "crane bird",
        "limpkin, Aramus pictus": "limpkin",
        "European gallinule, Porphyrio porphyrio": "common gallinule",
        "American coot, marsh hen, mud hen, water hen, Fulica americana": "American coot",
        "bustard": "bustard",
        "ruddy turnstone, Arenaria interpres": "ruddy turnstone",
        "red-backed sandpiper, dunlin, Erolia alpina": "dunlin",
        "redshank, Tringa totanus": "common redshank",
        "dowitcher": "dowitcher",
        "oystercatcher, oyster catcher": "oystercatcher",
        "pelican": "pelican",
        "king penguin, Aptenodytes patagonica": "king penguin",
        "albatross, mollymawk": "albatross",
        "grey whale, gray whale, devilfish, Eschrichtius gibbosus, Eschrichtius robustus": "grey whale",
        "killer whale, killer, orca, grampus, sea wolf, Orcinus orca": "killer whale",
        "dugong, Dugong dugon": "dugong",
        "sea lion": "sea lion",
        "Chihuahua": "Chihuahua",
        "Japanese spaniel": "Japanese Chin",
        "Maltese dog, Maltese terrier, Maltese": "Maltese",
        "Pekinese, Pekingese, Peke": "Pekingese",
        "Shih-Tzu": "Shih Tzu",
        "Blenheim spaniel": "King Charles Spaniel",
        "papillon": "Papillon",
        "toy terrier": "toy terrier",
        "Rhodesian ridgeback": "Rhodesian Ridgeback",
        "Afghan hound, Afghan": "Afghan Hound",
        "basset, basset hound": "Basset Hound",
        "beagle": "Beagle",
        "bloodhound, sleuthhound": "Bloodhound",
        "bluetick": "Bluetick Coonhound",
        "black-and-tan coonhound": "Black and Tan Coonhound",
        "Walker hound, Walker foxhound": "Treeing Walker Coonhound",
        "English foxhound": "English foxhound",
        "redbone": "Redbone Coonhound",
        "borzoi, Russian wolfhound": "borzoi",
        "Irish wolfhound": "Irish Wolfhound",
        "Italian greyhound": "Italian Greyhound",
        "whippet": "Whippet",
        "Ibizan hound, Ibizan Podenco": "Ibizan Hound",
        "Norwegian elkhound, elkhound": "Norwegian Elkhound",
        "otterhound, otter hound": "Otterhound",
        "Saluki, gazelle hound": "Saluki",
        "Scottish deerhound, deerhound": "Scottish Deerhound",
        "Weimaraner": "Weimaraner",
        "Staffordshire bullterrier, Staffordshire bull terrier": "Staffordshire Bull Terrier",
        "American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier": (
            "American Staffordshire Terrier"
        ),
        "Bedlington terrier": "Bedlington Terrier",
        "Border terrier": "Border Terrier",
        "Kerry blue terrier": "Kerry Blue Terrier",
        "Irish terrier": "Irish Terrier",
        "Norfolk terrier": "Norfolk Terrier",
        "Norwich terrier": "Norwich Terrier",
        "Yorkshire terrier": "Yorkshire Terrier",
        "wire-haired fox terrier": "Wire Fox Terrier",
        "Lakeland terrier": "Lakeland Terrier",
        "Sealyham terrier, Sealyham": "Sealyham Terrier",
        "Airedale, Airedale terrier": "Airedale Terrier",
        "cairn, cairn terrier": "Cairn Terrier",
        "Australian terrier": "Australian Terrier",
        "Dandie Dinmont, Dandie Dinmont terrier": "Dandie Dinmont Terrier",
        "Boston bull, Boston terrier": "Boston Terrier",
        "miniature schnauzer": "Miniature Schnauzer",
        "giant schnauzer": "Giant Schnauzer",
        "standard schnauzer": "Standard Schnauzer",
        "Scotch terrier, Scottish terrier, Scottie": "Scottish Terrier",
        "Tibetan terrier, chrysanthemum dog": "Tibetan Terrier",
        "silky terrier, Sydney silky": "Australian Silky Terrier",
        "soft-coated wheaten terrier": "Soft-coated Wheaten Terrier",
        "West Highland white terrier": "West Highland White Terrier",
        "Lhasa, Lhasa apso": "Lhasa Apso",
        "flat-coated retriever": "Flat-Coated Retriever",
        "curly-coated retriever": "Curly-coated Retriever",
        "golden retriever": "Golden Retriever",
        "Labrador retriever": "Labrador Retriever",
        "Chesapeake Bay retriever": "Chesapeake Bay Retriever",
        "German short-haired pointer": "German Shorthaired Pointer",
        "vizsla, Hungarian pointer": "Vizsla",
        "English setter": "English Setter",
        "Irish setter, red setter": "Irish Setter",
        "Gordon setter": "Gordon Setter",
        "Brittany spaniel": "Brittany dog",
        "clumber, clumber spaniel": "Clumber Spaniel",
        "English springer, English springer spaniel": "English Springer Spaniel",
        "Welsh springer spaniel": "Welsh Springer Spaniel",
        "cocker spaniel, English cocker spaniel, cocker": "Cocker Spaniel",
        "Sussex spaniel": "Sussex Spaniel",
        "Irish water spaniel": "Irish Water Spaniel",
        "kuvasz": "Kuvasz",
        "schipperke": "Schipperke",
        "groenendael": "Groenendael dog",
        "malinois": "Malinois",
        "briard": "Briard",
        "kelpie": "Australian Kelpie",
        "komondor": "Komondor",
        "Old English sheepdog, bobtail": "Old English Sheepdog",
        "Shetland sheepdog, Shetland sheep dog, Shetland": "Shetland Sheepdog",
        "collie": "collie",
        "Border collie": "Border Collie",
        "Bouvier des Flandres, Bouviers des Flandres": "Bouvier des Flandres dog",
        "Rottweiler": "Rottweiler",
        "German shepherd, German shepherd dog, German police dog, alsatian": "German Shepherd Dog",
        "Doberman, Doberman pinscher": "Dobermann",
        "miniature pinscher": "Miniature Pinscher",
        "Greater Swiss Mountain dog": "Greater Swiss Mountain Dog",
        "Bernese mountain dog": "Bernese Mountain Dog",
        "Appenzeller": "Appenzeller Sennenhund",
        "EntleBucher": "Entlebucher Sennenhund",
        "boxer": "Boxer",
        "bull mastiff": "Bullmastiff",
        "Tibetan mastiff": "Tibetan Mastiff",
        "French bulldog": "French Bulldog",
        "Great Dane": "Great Dane",
        "Saint Bernard, St Bernard": "St. Bernard",
        "Eskimo dog, husky": "husky",
        "malamute, malemute, Alaskan malamute": "Alaskan Malamute",
        "Siberian husky": "Siberian Husky",
        "dalmatian, coach dog, carriage dog": "Dalmatian",
        "affenpinscher, monkey pinscher, monkey dog": "Affenpinscher",
        "basenji": "Basenji",
        "pug, pug-dog": "pug",
        "Leonberg": "Leonberger",
        "Newfoundland, Newfoundland dog": "Newfoundland dog",
        "Great Pyrenees": "Great Pyrenees dog",
        "Samoyed, Samoyede": "Samoyed",
        "Pomeranian": "Pomeranian",
        "chow, chow chow": "Chow Chow",
        "keeshond": "Keeshond",
        "Brabancon griffon": "brussels griffon",
        "Pembroke, Pembroke Welsh corgi": "Pembroke Welsh Corgi",
        "Cardigan, Cardigan Welsh corgi": "Cardigan Welsh Corgi",
        "toy poodle": "Toy Poodle",
        "miniature poodle": "Miniature Poodle",
        "standard poodle": "Standard Poodle",
        "Mexican hairless": "Mexican hairless dog (xoloitzcuintli)",
        "timber wolf, grey wolf, gray wolf, Canis lupus": "grey wolf",
        "white wolf, Arctic wolf, Canis lupus tundrarum": "Alaskan tundra wolf",
        "red wolf, maned wolf, Canis rufus, Canis niger": "red wolf or maned wolf",
        "coyote, prairie wolf, brush wolf, Canis latrans": "coyote",
        "dingo, warrigal, warragal, Canis dingo": "dingo",
        "dhole, Cuon alpinus": "dhole",
        "African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus": "African wild dog",
        "hyena, hyaena": "hyena",
        "red fox, Vulpes vulpes": "red fox",
        "kit fox, Vulpes macrotis": "kit fox",
        "Arctic fox, white fox, Alopex lagopus": "Arctic fox",
        "grey fox, gray fox, Urocyon cinereoargenteus": "grey fox",
        "tabby, tabby cat": "tabby cat",
        "tiger cat": "tiger cat",
        "Persian cat": "Persian cat",
        "Siamese cat, Siamese": "Siamese cat",
        "Egyptian cat": "Egyptian Mau",
        "cougar, puma, catamount, mountain lion, painter, panther, Felis concolor": "cougar",
        "lynx, catamount": "lynx",
        "leopard, Panthera pardus": "leopard",
        "snow leopard, ounce, Panthera uncia": "snow leopard",
        "jaguar, panther, Panthera onca, Felis onca": "jaguar",
        "lion, king of beasts, Panthera leo": "lion",
        "tiger, Panthera tigris": "tiger",
        "cheetah, chetah, Acinonyx jubatus": "cheetah",
        "brown bear, bruin, Ursus arctos": "brown bear",
        "American black bear, black bear, Ursus americanus, Euarctos americanus": "American black bear",
        "ice bear, polar bear, Ursus Maritimus, Thalarctos maritimus": "polar bear",
        "sloth bear, Melursus ursinus, Ursus ursinus": "sloth bear",
        "mongoose": "mongoose",
        "meerkat, mierkat": "meerkat",
        "tiger beetle": "tiger beetle",
        "ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle": "ladybug",
        "ground beetle, carabid beetle": "ground beetle",
        "long-horned beetle, longicorn, longicorn beetle": "longhorn beetle",
        "leaf beetle, chrysomelid": "leaf beetle",
        "dung beetle": "dung beetle",
        "rhinoceros beetle": "rhinoceros beetle",
        "weevil": "weevil",
        "fly": "fly",
        "bee": "bee",
        "ant, emmet, pismire": "ant",
        "grasshopper, hopper": "grasshopper",
        "cricket": "cricket insect",
        "walking stick, walkingstick, stick insect": "stick insect",
        "cockroach, roach": "cockroach",
        "mantis, mantid": "praying mantis",
        "cicada, cicala": "cicada",
        "leafhopper": "leafhopper",
        "lacewing, lacewing fly": "lacewing",
        "dragonfly, darning needle, devil's darning needle, sewing needle, snake feeder, snake doctor, mosquito hawk, skeeter hawk": (
            "dragonfly"
        ),
        "damselfly": "damselfly",
        "admiral": "red admiral butterfly",
        "ringlet, ringlet butterfly": "ringlet butterfly",
        "monarch, monarch butterfly, milkweed butterfly, Danaus plexippus": "monarch butterfly",
        "cabbage butterfly": "small white butterfly",
        "sulphur butterfly, sulfur butterfly": "sulphur butterfly",
        "lycaenid, lycaenid butterfly": "gossamer-winged butterfly",
        "starfish, sea star": "starfish",
        "sea urchin": "sea urchin",
        "sea cucumber, holothurian": "sea cucumber",
        "wood rabbit, cottontail, cottontail rabbit": "cottontail rabbit",
        "hare": "hare",
        "Angora, Angora rabbit": "Angora rabbit",
        "hamster": "hamster",
        "porcupine, hedgehog": "porcupine",
        "fox squirrel, eastern fox squirrel, Sciurus niger": "fox squirrel",
        "marmot": "marmot",
        "beaver": "beaver",
        "guinea pig, Cavia cobaya": "guinea pig",
        "sorrel": "common sorrel horse",
        "zebra": "zebra",
        "hog, pig, grunter, squealer, Sus scrofa": "pig",
        "wild boar, boar, Sus scrofa": "wild boar",
        "warthog": "warthog",
        "hippopotamus, hippo, river horse, Hippopotamus amphibius": "hippopotamus",
        "ox": "ox",
        "water buffalo, water ox, Asiatic buffalo, Bubalus bubalis": "water buffalo",
        "bison": "bison",
        "ram, tup": "ram (adult male sheep)",
        "bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, Rocky Mountain sheep, Ovis canadensis": (
            "bighorn sheep"
        ),
        "ibex, Capra ibex": "Alpine ibex",
        "hartebeest": "hartebeest",
        "impala, Aepyceros melampus": "impala (antelope)",
        "gazelle": "gazelle",
        "Arabian camel, dromedary, Camelus dromedarius": "arabian camel",
        "llama": "llama",
        "weasel": "weasel",
        "mink": "mink",
        "polecat, fitch, foulmart, foumart, Mustela putorius": "European polecat",
        "black-footed ferret, ferret, Mustela nigripes": "black-footed ferret",
        "otter": "otter",
        "skunk, polecat, wood pussy": "skunk",
        "badger": "badger",
        "armadillo": "armadillo",
        "three-toed sloth, ai, Bradypus tridactylus": "three-toed sloth",
        "orangutan, orang, orangutang, Pongo pygmaeus": "orangutan",
        "gorilla, Gorilla gorilla": "gorilla",
        "chimpanzee, chimp, Pan troglodytes": "chimpanzee",
        "gibbon, Hylobates lar": "gibbon",
        "siamang, Hylobates syndactylus, Symphalangus syndactylus": "siamang",
        "guenon, guenon monkey": "guenon",
        "patas, hussar monkey, Erythrocebus patas": "patas monkey",
        "baboon": "baboon",
        "macaque": "macaque",
        "langur": "langur",
        "colobus, colobus monkey": "black-and-white colobus",
        "proboscis monkey, Nasalis larvatus": "proboscis monkey",
        "marmoset": "marmoset",
        "capuchin, ringtail, Cebus capucinus": "white-headed capuchin",
        "howler monkey, howler": "howler monkey",
        "titi, titi monkey": "titi monkey",
        "spider monkey, Ateles geoffroyi": "Geoffroy's spider monkey",
        "squirrel monkey, Saimiri sciureus": "common squirrel monkey",
        "Madagascar cat, ring-tailed lemur, Lemur catta": "ring-tailed lemur",
        "indri, indris, Indri indri, Indri brevicaudatus": "indri",
        "Indian elephant, Elephas maximus": "Asian elephant",
        "African elephant, Loxodonta africana": "African bush elephant",
        "lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens": "red panda",
        "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca": "giant panda",
        "barracouta, snoek": "snoek fish",
        "eel": "eel",
        "coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch": "silver salmon",
        "rock beauty, Holocanthus tricolor": "rock beauty fish",
        "anemone fish": "clownfish",
        "sturgeon": "sturgeon",
        "gar, garfish, garpike, billfish, Lepisosteus osseus": "gar fish",
        "lionfish": "lionfish",
        "puffer, pufferfish, blowfish, globefish": "pufferfish",
        "abacus": "abacus",
        "abaya": "abaya",
        "academic gown, academic robe, judge's robe": "academic gown",
        "accordion, piano accordion, squeeze box": "accordion",
        "acoustic guitar": "acoustic guitar",
        "aircraft carrier, carrier, flattop, attack aircraft carrier": "aircraft carrier",
        "airliner": "airliner",
        "airship, dirigible": "airship",
        "altar": "altar",
        "ambulance": "ambulance",
        "amphibian, amphibious vehicle": "amphibious vehicle",
        "analog clock": "analog clock",
        "apiary, bee house": "apiary",
        "apron": "apron",
        "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin": (
            "trash can"
        ),
        "assault rifle, assault gun": "assault rifle",
        "backpack, back pack, knapsack, packsack, rucksack, haversack": "backpack",
        "bakery, bakeshop, bakehouse": "bakery",
        "balance beam, beam": "balance beam",
        "balloon": "balloon",
        "ballpoint, ballpoint pen, ballpen, Biro": "ballpoint pen",
        "Band Aid": "Band-Aid",
        "banjo": "banjo",
        "bannister, banister, balustrade, balusters, handrail": "baluster / handrail",
        "barbell": "barbell",
        "barber chair": "barber chair",
        "barbershop": "barbershop",
        "barn": "barn",
        "barometer": "barometer",
        "barrel, cask": "barrel",
        "barrow, garden cart, lawn cart, wheelbarrow": "wheelbarrow",
        "baseball": "baseball",
        "basketball": "basketball",
        "bassinet": "bassinet",
        "bassoon": "bassoon",
        "bathing cap, swimming cap": "swimming cap",
        "bath towel": "bath towel",
        "bathtub, bathing tub, bath, tub": "bathtub",
        "beach wagon, station wagon, wagon, estate car, beach waggon, station waggon, waggon": "station wagon",
        "beacon, lighthouse, beacon light, pharos": "lighthouse",
        "beaker": "beaker",
        "bearskin, busby, shako": "military hat (bearskin or shako)",
        "beer bottle": "beer bottle",
        "beer glass": "beer glass",
        "bell cote, bell cot": "bell tower",
        "bib": "baby bib",
        "bicycle-built-for-two, tandem bicycle, tandem": "tandem bicycle",
        "bikini, two-piece": "bikini",
        "binder, ring-binder": "ring binder",
        "binoculars, field glasses, opera glasses": "binoculars",
        "birdhouse": "birdhouse",
        "boathouse": "boathouse",
        "bobsled, bobsleigh, bob": "bobsleigh",
        "bolo tie, bolo, bola tie, bola": "bolo tie",
        "bonnet, poke bonnet": "poke bonnet",
        "bookcase": "bookcase",
        "bookshop, bookstore, bookstall": "bookstore",
        "bottlecap": "bottle cap",
        "bow": "hunting bow",
        "bow tie, bow-tie, bowtie": "bow tie",
        "brass, memorial tablet, plaque": "brass memorial plaque",
        "brassiere, bra, bandeau": "bra",
        "breakwater, groin, groyne, mole, bulwark, seawall, jetty": "breakwater",
        "breastplate, aegis, egis": "breastplate",
        "broom": "broom",
        "bucket, pail": "bucket",
        "buckle": "buckle",
        "bulletproof vest": "bulletproof vest",
        "bullet train, bullet": "high-speed train",
        "butcher shop, meat market": "butcher shop",
        "cab, hack, taxi, taxicab": "taxicab",
        "caldron, cauldron": "cauldron",
        "candle, taper, wax light": "candle",
        "cannon": "cannon",
        "canoe": "canoe",
        "can opener, tin opener": "can opener",
        "cardigan": "cardigan",
        "car mirror": "car mirror",
        "carousel, carrousel, merry-go-round, roundabout, whirligig": "carousel",
        "carpenter's kit, tool kit": "tool kit",
        "carton": "cardboard box / carton",
        "car wheel": "car wheel",
        "cash machine, cash dispenser, automated teller machine, automatic teller machine, automated teller, automatic teller, ATM": (
            "automated teller machine"
        ),
        "cassette": "cassette",
        "cassette player": "cassette player",
        "castle": "castle",
        "catamaran": "catamaran",
        "CD player": "CD player",
        "cello, violoncello": "cello",
        "cellular telephone, cellular phone, cellphone, cell, mobile phone": "mobile phone",
        "chain": "chain",
        "chainlink fence": "chain-link fence",
        "chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour": "chain mail",
        "chain saw, chainsaw": "chainsaw",
        "chest": "storage chest",
        "chiffonier, commode": "chiffonier",
        "chime, bell, gong": "bell or wind chime",
        "china cabinet, china closet": "china cabinet",
        "Christmas stocking": "Christmas stocking",
        "church, church building": "church",
        "cinema, movie theater, movie theatre, movie house, picture palace": "movie theater",
        "cleaver, meat cleaver, chopper": "cleaver",
        "cliff dwelling": "cliff dwelling",
        "cloak": "cloak",
        "clog, geta, patten, sabot": "clogs",
        "cocktail shaker": "cocktail shaker",
        "coffee mug": "coffee mug",
        "coffeepot": "coffeemaker",
        "coil, spiral, volute, whorl, helix": "spiral or coil",
        "combination lock": "combination lock",
        "computer keyboard, keypad": "computer keyboard",
        "confectionery, confectionary, candy store": "candy store",
        "container ship, containership, container vessel": "container ship",
        "convertible": "convertible",
        "corkscrew, bottle screw": "corkscrew",
        "cornet, horn, trumpet, trump": "cornet",
        "cowboy boot": "cowboy boot",
        "cowboy hat, ten-gallon hat": "cowboy hat",
        "cradle": "cradle",
        "crane2": "construction crane",
        "crash helmet": "crash helmet",
        "crate": "crate",
        "crib, cot": "infant bed",
        "Crock Pot": "Crock Pot",
        "croquet ball": "croquet ball",
        "crutch": "crutch",
        "cuirass": "cuirass",
        "dam, dike, dyke": "dam",
        "desk": "desk",
        "desktop computer": "desktop computer",
        "dial telephone, dial phone": "rotary dial telephone",
        "diaper, nappy, napkin": "diaper",
        "digital clock": "digital clock",
        "digital watch": "digital watch",
        "dining table, board": "dining table",
        "dishrag, dishcloth": "dishcloth",
        "dishwasher, dish washer, dishwashing machine": "dishwasher",
        "disk brake, disc brake": "disc brake",
        "dock, dockage, docking facility": "dock",
        "dogsled, dog sled, dog sleigh": "dog sled",
        "dome": "dome",
        "doormat, welcome mat": "doormat",
        "drilling platform, offshore rig": "drilling rig",
        "drum, membranophone, tympan": "drum",
        "drumstick": "drumstick",
        "dumbbell": "dumbbell",
        "Dutch oven": "Dutch oven",
        "electric fan, blower": "electric fan",
        "electric guitar": "electric guitar",
        "electric locomotive": "electric locomotive",
        "entertainment center": "entertainment center",
        "envelope": "envelope",
        "espresso maker": "espresso machine",
        "face powder": "face powder",
        "feather boa, boa": "feather boa",
        "file, file cabinet, filing cabinet": "filing cabinet",
        "fireboat": "fireboat",
        "fire engine, fire truck": "fire truck",
        "fire screen, fireguard": "fire screen",
        "flagpole, flagstaff": "flagpole",
        "flute, transverse flute": "flute",
        "folding chair": "folding chair",
        "football helmet": "football helmet",
        "forklift": "forklift",
        "fountain": "fountain",
        "fountain pen": "fountain pen",
        "four-poster": "four-poster bed",
        "freight car": "freight car",
        "French horn, horn": "French horn",
        "frying pan, frypan, skillet": "frying pan",
        "fur coat": "fur coat",
        "garbage truck, dustcart": "garbage truck",
        "gasmask, respirator, gas helmet": "gas mask or respirator",
        "gas pump, gasoline pump, petrol pump, island dispenser": "gas pump",
        "goblet": "goblet",
        "go-kart": "go-kart",
        "golf ball": "golf ball",
        "golfcart, golf cart": "golf cart",
        "gondola": "gondola",
        "gong, tam-tam": "gong",
        "gown": "gown",
        "grand piano, grand": "grand piano",
        "greenhouse, nursery, glasshouse": "greenhouse",
        "grille, radiator grille": "radiator grille",
        "grocery store, grocery, food market, market": "grocery store",
        "guillotine": "guillotine",
        "hair slide": "hair clip",
        "hair spray": "hair spray",
        "half track": "half-track",
        "hammer": "hammer",
        "hamper": "hamper",
        "hand blower, blow dryer, blow drier, hair dryer, hair drier": "hair dryer",
        "hand-held computer, hand-held microcomputer": "hand-held computer",
        "handkerchief, hankie, hanky, hankey": "handkerchief",
        "hard disc, hard disk, fixed disk": "hard disk drive",
        "harmonica, mouth organ, harp, mouth harp": "harmonica",
        "harp": "harp",
        "harvester, reaper": "combine harvester",
        "hatchet": "hatchet",
        "holster": "holster",
        "home theater, home theatre": "home theater",
        "honeycomb": "honeycomb",
        "hook, claw": "hook",
        "hoopskirt, crinoline": "hoop skirt",
        "horizontal bar, high bar": "gymnastic horizontal bar",
        "horse cart, horse-cart": "horse-drawn vehicle",
        "hourglass": "hourglass",
        "iPod": "iPod",
        "iron, smoothing iron": "clothes iron",
        "jack-o'-lantern": "carved pumpkin",
        "jean, blue jean, denim": "jeans",
        "jeep, landrover": "jeep",
        "jersey, T-shirt, tee shirt": "T-shirt",
        "jigsaw puzzle": "jigsaw puzzle",
        "jinrikisha, ricksha, rickshaw": "rickshaw",
        "joystick": "joystick",
        "kimono": "kimono",
        "knee pad": "knee pad",
        "knot": "knot",
        "lab coat, laboratory coat": "lab coat",
        "ladle": "ladle",
        "lampshade, lamp shade": "lampshade",
        "laptop, laptop computer": "laptop computer",
        "lawn mower, mower": "lawn mower",
        "lens cap, lens cover": "lens cap",
        "letter opener, paper knife, paperknife": "letter opener",
        "library": "library",
        "lifeboat": "lifeboat",
        "lighter, light, igniter, ignitor": "lighter",
        "limousine, limo": "limousine",
        "liner, ocean liner": "ocean liner",
        "lipstick, lip rouge": "lipstick",
        "Loafer": "slip-on shoe",
        "lotion": "lotion",
        "loudspeaker, speaker, speaker unit, loudspeaker system, speaker system": "music speaker",
        "loupe, jeweler's loupe": "loupe magnifying glass",
        "lumbermill, sawmill": "sawmill",
        "magnetic compass": "magnetic compass",
        "mailbag, postbag": "messenger bag",
        "mailbox, letter box": "mailbox",
        "maillot": "tights",
        "maillot, tank suit": "one-piece bathing suit",
        "manhole cover": "manhole cover",
        "maraca": "maraca",
        "marimba, xylophone": "marimba",
        "mask": "mask",
        "matchstick": "matchstick",
        "maypole": "maypole",
        "maze, labyrinth": "maze",
        "measuring cup": "measuring cup",
        "medicine chest, medicine cabinet": "medicine cabinet",
        "megalith, megalithic structure": "megalith",
        "microphone, mike": "microphone",
        "microwave, microwave oven": "microwave oven",
        "military uniform": "military uniform",
        "milk can": "milk can",
        "minibus": "minibus",
        "miniskirt, mini": "miniskirt",
        "minivan": "minivan",
        "missile": "missile",
        "mitten": "mitten",
        "mixing bowl": "mixing bowl",
        "mobile home, manufactured home": "mobile home",
        "Model T": "ford model t",
        "modem": "modem",
        "monastery": "monastery",
        "monitor": "monitor",
        "moped": "moped",
        "mortar": "mortar and pestle",
        "mortarboard": "graduation cap",
        "mosque": "mosque",
        "mosquito net": "mosquito net",
        "motor scooter, scooter": "vespa",
        "mountain bike, all-terrain bike, off-roader": "mountain bike",
        "mountain tent": "tent",
        "mouse, computer mouse": "computer mouse",
        "mousetrap": "mousetrap",
        "moving van": "moving van",
        "muzzle": "muzzle",
        "nail": "metal nail",
        "neck brace": "neck brace",
        "necklace": "necklace",
        "nipple": "baby pacifier",
        "notebook, notebook computer": "notebook computer",
        "obelisk": "obelisk",
        "oboe, hautboy, hautbois": "oboe",
        "ocarina, sweet potato": "ocarina",
        "odometer, hodometer, mileometer, milometer": "odometer",
        "oil filter": "oil filter",
        "organ, pipe organ": "pipe organ",
        "oscilloscope, scope, cathode-ray oscilloscope, CRO": "oscilloscope",
        "overskirt": "overskirt",
        "oxcart": "bullock cart",
        "oxygen mask": "oxygen mask",
        "packet": "product packet / packaging",
        "paddle, boat paddle": "paddle",
        "paddlewheel, paddle wheel": "paddle wheel",
        "padlock": "padlock",
        "paintbrush": "paintbrush",
        "pajama, pyjama, pj's, jammies": "pajamas",
        "palace": "palace",
        "panpipe, pandean pipe, syrinx": "pan flute",
        "paper towel": "paper towel",
        "parachute, chute": "parachute",
        "parallel bars, bars": "parallel bars",
        "park bench": "park bench",
        "parking meter": "parking meter",
        "passenger car, coach, carriage": "railroad car",
        "patio, terrace": "patio",
        "pay-phone, pay-station": "payphone",
        "pedestal, plinth, footstall": "pedestal",
        "pencil box, pencil case": "pencil case",
        "pencil sharpener": "pencil sharpener",
        "perfume, essence": "perfume",
        "Petri dish": "Petri dish",
        "photocopier": "photocopier",
        "pick, plectrum, plectron": "plectrum",
        "pickelhaube": "Pickelhaube",
        "picket fence, paling": "picket fence",
        "pickup, pickup truck": "pickup truck",
        "pier": "pier",
        "piggy bank, penny bank": "piggy bank",
        "pill bottle": "pill bottle",
        "pillow": "pillow",
        "ping-pong ball": "ping-pong ball",
        "pinwheel": "pinwheel",
        "pirate, pirate ship": "pirate ship",
        "pitcher, ewer": "drink pitcher",
        "plane, carpenter's plane, woodworking plane": "block plane",
        "planetarium": "planetarium",
        "plastic bag": "plastic bag",
        "plate rack": "plate rack",
        "plow, plough": "farm plow",
        "plunger, plumber's helper": "plunger",
        "Polaroid camera, Polaroid Land camera": "Polaroid camera",
        "pole": "pole",
        "police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria": "police van",
        "poncho": "poncho",
        "pool table, billiard table, snooker table": "pool table",
        "pop bottle, soda bottle": "soda bottle",
        "pot, flowerpot": "plant pot",
        "potter's wheel": "potter's wheel",
        "power drill": "power drill",
        "prayer rug, prayer mat": "prayer rug",
        "printer": "printer",
        "prison, prison house": "prison",
        "projectile, missile": "missile",
        "projector": "projector",
        "puck, hockey puck": "hockey puck",
        "punching bag, punch bag, punching ball, punchball": "punching bag",
        "purse": "purse",
        "quill, quill pen": "quill",
        "quilt, comforter, comfort, puff": "quilt",
        "racer, race car, racing car": "race car",
        "racket, racquet": "racket",
        "radiator": "radiator",
        "radio, wireless": "radio",
        "radio telescope, radio reflector": "radio telescope",
        "rain barrel": "rain barrel",
        "recreational vehicle, RV, R.V.": "recreational vehicle",
        "reel": "fishing casting reel",
        "reflex camera": "reflex camera",
        "refrigerator, icebox": "refrigerator",
        "remote control, remote": "remote control",
        "restaurant, eating house, eating place, eatery": "restaurant",
        "revolver, six-gun, six-shooter": "revolver",
        "rifle": "rifle",
        "rocking chair, rocker": "rocking chair",
        "rotisserie": "rotisserie",
        "rubber eraser, rubber, pencil eraser": "eraser",
        "rugby ball": "rugby ball",
        "rule, ruler": "ruler measuring stick",
        "running shoe": "sneaker",
        "safe": "safe",
        "safety pin": "safety pin",
        "saltshaker, salt shaker": "salt shaker",
        "sandal": "sandal",
        "sarong": "sarong",
        "sax, saxophone": "saxophone",
        "scabbard": "scabbard",
        "scale, weighing machine": "weighing scale",
        "school bus": "school bus",
        "schooner": "schooner",
        "scoreboard": "scoreboard",
        "screen, CRT screen": "CRT monitor",
        "screw": "screw",
        "screwdriver": "screwdriver",
        "seat belt, seatbelt": "seat belt",
        "sewing machine": "sewing machine",
        "shield, buckler": "shield",
        "shoe shop, shoe-shop, shoe store": "shoe store",
        "shoji": "shoji screen / room divider",
        "shopping basket": "shopping basket",
        "shopping cart": "shopping cart",
        "shovel": "shovel",
        "shower cap": "shower cap",
        "shower curtain": "shower curtain",
        "ski": "ski",
        "ski mask": "balaclava ski mask",
        "sleeping bag": "sleeping bag",
        "slide rule, slipstick": "slide rule",
        "sliding door": "sliding door",
        "slot, one-armed bandit": "slot machine",
        "snorkel": "snorkel",
        "snowmobile": "snowmobile",
        "snowplow, snowplough": "snowplow",
        "soap dispenser": "soap dispenser",
        "soccer ball": "soccer ball",
        "sock": "sock",
        "solar dish, solar collector, solar furnace": "solar thermal collector",
        "sombrero": "sombrero",
        "soup bowl": "soup bowl",
        "space bar": "keyboard space bar",
        "space heater": "space heater",
        "space shuttle": "space shuttle",
        "spatula": "spatula",
        "speedboat": "motorboat",
        "spider web, spider's web": "spider web",
        "spindle": "spindle",
        "sports car, sport car": "sports car",
        "spotlight, spot": "spotlight",
        "stage": "stage",
        "steam locomotive": "steam locomotive",
        "steel arch bridge": "through arch bridge",
        "steel drum": "steel drum",
        "stethoscope": "stethoscope",
        "stole": "scarf",
        "stone wall": "stone wall",
        "stopwatch, stop watch": "stopwatch",
        "stove": "stove",
        "strainer": "strainer",
        "streetcar, tram, tramcar, trolley, trolley car": "tram",
        "stretcher": "stretcher",
        "studio couch, day bed": "couch",
        "stupa, tope": "stupa",
        "submarine, pigboat, sub, U-boat": "submarine",
        "suit, suit of clothes": "suit",
        "sundial": "sundial",
        "sunglass": "sunglasses",
        "sunglasses, dark glasses, shades": "sunglasses",
        "sunscreen, sunblock, sun blocker": "sunscreen",
        "suspension bridge": "suspension bridge",
        "swab, swob, mop": "mop",
        "sweatshirt": "sweatshirt",
        "swimming trunks, bathing trunks": "swim trunks / shorts",
        "swing": "swing",
        "switch, electric switch, electrical switch": "electrical switch",
        "syringe": "syringe",
        "table lamp": "table lamp",
        "tank, army tank, armored combat vehicle, armoured combat vehicle": "tank",
        "tape player": "tape player",
        "teapot": "teapot",
        "teddy, teddy bear": "teddy bear",
        "television, television system": "television",
        "tennis ball": "tennis ball",
        "thatch, thatched roof": "thatched roof",
        "theater curtain, theatre curtain": "front curtain",
        "thimble": "thimble",
        "thresher, thrasher, threshing machine": "threshing machine",
        "throne": "throne",
        "tile roof": "tile roof",
        "toaster": "toaster",
        "tobacco shop, tobacconist shop, tobacconist": "tobacco shop",
        "toilet seat": "toilet seat",
        "torch": "torch",
        "totem pole": "totem pole",
        "tow truck, tow car, wrecker": "tow truck",
        "toyshop": "toy store",
        "tractor": "tractor",
        "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi": "semi-trailer truck",
        "tray": "tray",
        "trench coat": "trench coat",
        "tricycle, trike, velocipede": "tricycle",
        "trimaran": "trimaran",
        "tripod": "tripod",
        "triumphal arch": "triumphal arch",
        "trolleybus, trolley coach, trackless trolley": "trolleybus",
        "trombone": "trombone",
        "tub, vat": "hot tub",
        "turnstile": "turnstile",
        "typewriter keyboard": "typewriter keyboard",
        "umbrella": "umbrella",
        "unicycle, monocycle": "unicycle",
        "upright, upright piano": "upright piano",
        "vacuum, vacuum cleaner": "vacuum cleaner",
        "vase": "vase",
        "vault": "vaulted or arched ceiling",
        "velvet": "velvet fabric",
        "vending machine": "vending machine",
        "vestment": "vestment",
        "viaduct": "viaduct",
        "violin, fiddle": "violin",
        "volleyball": "volleyball",
        "waffle iron": "waffle iron",
        "wall clock": "wall clock",
        "wallet, billfold, notecase, pocketbook": "wallet",
        "wardrobe, closet, press": "wardrobe",
        "warplane, military plane": "military aircraft",
        "washbasin, handbasin, washbowl, lavabo, wash-hand basin": "sink",
        "washer, automatic washer, washing machine": "washing machine",
        "water bottle": "water bottle",
        "water jug": "water jug",
        "water tower": "water tower",
        "whiskey jug": "whiskey jug",
        "whistle": "whistle",
        "wig": "hair wig",
        "window screen": "window screen",
        "window shade": "window shade",
        "Windsor tie": "Windsor tie",
        "wine bottle": "wine bottle",
        "wing": "airplane wing",
        "wok": "wok",
        "wooden spoon": "wooden spoon",
        "wool, woolen, woollen": "wool",
        "worm fence, snake fence, snake-rail fence, Virginia fence": "split-rail fence",
        "wreck": "shipwreck",
        "yawl": "sailboat",
        "yurt": "yurt",
        "web site, website, internet site, site": "website",
        "comic book": "comic book",
        "crossword puzzle, crossword": "crossword",
        "street sign": "traffic or street sign",
        "traffic light, traffic signal, stoplight": "traffic light",
        "book jacket, dust cover, dust jacket, dust wrapper": "dust jacket",
        "menu": "menu",
        "plate": "plate",
        "guacamole": "guacamole",
        "consomme": "consomme",
        "hot pot, hotpot": "hot pot",
        "trifle": "trifle",
        "ice cream, icecream": "ice cream",
        "ice lolly, lolly, lollipop, popsicle": "popsicle",
        "French loaf": "baguette",
        "bagel, beigel": "bagel",
        "pretzel": "pretzel",
        "cheeseburger": "cheeseburger",
        "hotdog, hot dog, red hot": "hot dog",
        "mashed potato": "mashed potatoes",
        "head cabbage": "cabbage",
        "broccoli": "broccoli",
        "cauliflower": "cauliflower",
        "zucchini, courgette": "zucchini",
        "spaghetti squash": "spaghetti squash",
        "acorn squash": "acorn squash",
        "butternut squash": "butternut squash",
        "cucumber, cuke": "cucumber",
        "artichoke, globe artichoke": "artichoke",
        "bell pepper": "bell pepper",
        "cardoon": "cardoon",
        "mushroom": "mushroom",
        "Granny Smith": "Granny Smith apple",
        "strawberry": "strawberry",
        "orange": "orange",
        "lemon": "lemon",
        "fig": "fig",
        "pineapple, ananas": "pineapple",
        "banana": "banana",
        "jackfruit, jak, jack": "jackfruit",
        "custard apple": "cherimoya (custard apple)",
        "pomegranate": "pomegranate",
        "hay": "hay",
        "carbonara": "carbonara",
        "chocolate sauce, chocolate syrup": "chocolate syrup",
        "dough": "dough",
        "meat loaf, meatloaf": "meatloaf",
        "pizza, pizza pie": "pizza",
        "potpie": "pot pie",
        "burrito": "burrito",
        "red wine": "red wine",
        "espresso": "espresso",
        "cup": "tea cup",
        "eggnog": "eggnog",
        "alp": "mountain",
        "bubble": "bubble",
        "cliff, drop, drop-off": "cliff",
        "coral reef": "coral reef",
        "geyser": "geyser",
        "lakeside, lakeshore": "lakeshore",
        "promontory, headland, head, foreland": "promontory",
        "sandbar, sand bar": "sandbar",
        "seashore, coast, seacoast, sea-coast": "beach",
        "valley, vale": "valley",
        "volcano": "volcano",
        "ballplayer, baseball player": "baseball player",
        "groom, bridegroom": "bridegroom",
        "scuba diver": "scuba diver",
        "rapeseed": "rapeseed",
        "daisy": "daisy",
        "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum": (
            "yellow lady's slipper"
        ),
        "corn": "corn",
        "acorn": "acorn",
        "hip, rose hip, rosehip": "rose hip",
        "buckeye, horse chestnut, conker": "horse chestnut seed",
        "coral fungus": "coral fungus",
        "agaric": "agaric",
        "gyromitra": "gyromitra",
        "stinkhorn, carrion fungus": "stinkhorn mushroom",
        "earthstar": "earth star fungus",
        "hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa": "hen of the woods mushroom",
        "bolete": "bolete",
        "ear, spike, capitulum": "corn cob",
        "toilet tissue, toilet paper, bathroom tissue": "toilet paper",
    }
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a photo of a {class_name}.",
        },
    }


class ImageNet1kSampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    ImageNet1kVgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/imagenet-1k-Sample"


class HatefulMemesVgpt2ClassificationInContextAccWithKLAndEntropyROCAUC(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/Hateful-Memes"
    dataset_config: str = "seen"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
            ClassifMetrics.ROC_AUC,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    context_column_names: List[str] = ["text"]
    length_normalize: bool = False
    mapping_class_prompt_name_id_to_prompt_template_id: dict = {
        0: [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 17, 18, 20, 21, 22, 23, 24, 25],
        1: [7, 8, 15, 16, 19, 26, 27],
    }
    mapping_class_names_to_prompt_names: dict = {
        0: {"not-hateful": "no", "hateful": "yes"},
        1: {"not-hateful": "not hateful", "hateful": "hateful"},
    }
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": '<image>is an image with written: "{text}". Is it hateful? Answer: {class_name}',
        },
        1: {
            "prefix": "{bos_token}",
            "example": (
                '<image> The text:"{text}" is written on this meme. Is this meme hateful? Answer:{class_name}\n'
            ),
        },
        2: {
            "prefix": "{bos_token}",
            "example": '<image>is an image with written "{text}" on it. Is it hateful? Answer: {class_name}\n',
        },
        3: {
            "prefix": (
                "{bos_token}It's a conversation between a human, the user, and an intelligent visual AI, Bot. The user"
                " sends memes with text written on them, and Bot has to say whether the meme is hateful or not."
            ),
            "example": '<image>is an image with written "{text}" on it. Is it hateful? Answer: {class_name}\n',
        },
        4: {
            "prefix": (
                "{bos_token}You are an  intelligent virtual assistant that has to determine whether the meme given as"
                " input, along with a transcription of the text marked on it, is hateful (yes) or not (no)."
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}\n',
        },
        5: {
            "prefix": (
                "{bos_token}Classify the following memes. Answer yes if they are hateful and no if they are not.\n"
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}\n',
        },
        6: {
            "prefix": (
                "{bos_token}Analyze the given meme and determine if it falls under the category of hate speech (yes)"
                " or not (no).\n"
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}\n',
        },
        7: {
            "prefix": (
                "{bos_token}Analyze the given meme and determine if it falls under the category of hateful or not.\n"
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}\n',
        },
        8: {
            "prefix": (
                "{bos_token}Analyze the given meme and determine if it falls under the category of hateful or not."
                " Both the image and the text written on it matters for the prediction.\n"
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}\n',
        },
        9: {
            "prefix": "{bos_token}",
            "example": '<image> The text:"{text}" is written on this meme. Is this meme hateful? Answer:{class_name}',
        },
        10: {
            "prefix": "{bos_token}",
            "example": '<image>is an image with written "{text}" on it. Is it hateful? Answer: {class_name}',
        },
        11: {
            "prefix": (
                "{bos_token}It's a conversation between a human, the user, and an intelligent visual AI, Bot. The user"
                " sends memes with text written on them, and Bot has to say whether the meme is hateful or not."
            ),
            "example": '<image>is an image with written "{text}" on it. Is it hateful? Answer: {class_name}',
        },
        12: {
            "prefix": (
                "{bos_token}You are an  intelligent virtual assistant that has to determine whether the meme given as"
                " input, along with a transcription of the text marked on it, is hateful (yes) or not (no)."
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}',
        },
        13: {
            "prefix": (
                "{bos_token}Classify the following memes. Answer yes if they are hateful and no if they are not."
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}',
        },
        14: {
            "prefix": (
                "{bos_token}Analyze the given meme and determine if it falls under the category of hate speech (yes)"
                " or not (no)."
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}',
        },
        15: {
            "prefix": (
                "{bos_token}Analyze the given meme and determine if it falls under the category of hateful or not."
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}',
        },
        16: {
            "prefix": (
                "{bos_token}Analyze the given meme and determine if it falls under the category of hateful or not."
                " Both the image and the text written on it matters for the prediction."
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}',
        },
        17: {
            "prefix": (
                "{bos_token}It's a conversation between a human, the user, and an intelligent visual AI, Bot. The user"
                " sends pictures with text written on them, and Bot has to say whether this is is hateful or not."
            ),
            "example": '<image>is an image with written "{text}" on it. Is it hateful? Answer: {class_name}.',
        },
        18: {
            "prefix": (
                "{bos_token}It's a conversation between a human, the user, and an intelligent visual AI, Bot. The user"
                " sends pictures with text written on them, and Bot has to say whether this is is hateful or not.\n"
            ),
            "example": '<image>is an image with written "{text}" on it. Is it hateful? Answer: {class_name}.',
        },
        19: {
            "prefix": (
                "{bos_token}Analyze the given meme and determine if it falls under the category of hateful or not."
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}.',
        },
        20: {
            "prefix": None,
            "example": (
                '{bos_token}<image>is an image with written: "{text}". Is it hateful? Answer: {class_name}.{eos_token}'
            ),
        },
        21: {
            "prefix": "{bos_token}",
            "example": '<image>is an image with written: "{text}". Is it hateful? Answer: {class_name}',
        },
        22: {
            "prefix": (
                "It's a conversation between a human, the user, and an intelligent visual AI, Bot. The user"
                " sends memes with text written on them, and Bot has to say whether the meme is hateful or not."
            ),
            "example": '<image>is an image with written "{text}" on it. Is it hateful? Answer: {class_name}',
        },
        23: {
            "prefix": (
                "You are an  intelligent virtual assistant that has to determine whether the meme given as"
                " input, along with a transcription of the text marked on it, is hateful (yes) or not (no)."
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}',
        },
        24: {
            "prefix": "Classify the following memes. Answer yes if they are hateful and no if they are not.",
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}',
        },
        25: {
            "prefix": (
                "Analyze the given meme and determine if it falls under the category of hate speech (yes) or not (no)."
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}',
        },
        26: {
            "prefix": "Analyze the given meme and determine if it falls under the category of hateful or not.",
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}',
        },
        27: {
            "prefix": (
                "Analyze the given meme and determine if it falls under the category of hateful or not."
                " Both the image and the text written on it matters for the prediction."
            ),
            "example": 'Input:<image> Transcription: "{text}" Answer: {class_name}',
        },
    }


class ClevrVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/clevr"
    dataset_config: str = "classification"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    label_column_name: str = "answer"
    image_column_names: List[str] = ["image"]
    context_column_names: List[str] = ["question"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>Question: {question} Answer: {class_name}",
        },
    }


class ClevrSampleVgpt2ClassificationInContextAccWithKLAndEntropy(ClevrVgpt2ClassificationInContextAccWithKLAndEntropy):
    dataset_name: str = "HuggingFaceM4/clevr-Sample"


class SNLIVEImageOnlySampleVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/SNLI-VE-Sample"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.MEAN_PER_CLASS_ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    context_column_names: List[str] = ["hypothesis"]
    length_normalize: bool = False
    mapping_class_names_to_prompt_names: dict = {
        "entailment": "correct",
        "neutral": "inconclusive",
        "contradiction": "incorrect",
    }
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": (
                "<image>Using only the preceding image, and what you"
                " know about the world, {hypothesis} is definitely correct, incorrect, or inconclusive? Answer:"
                " {class_name}."
            ),
        },
    }


class SNLIVEImagePremiseSampleVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/SNLI-VE_modif_premise_hypothesis-Sample"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.MEAN_PER_CLASS_ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    # The dataset has been modified so the column context of the dataset is constructed as follows:
    # f"{premise} Using only the image, the preceding description, and what you know about the world, \"{hypothesis}\""
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    context_column_names: List[str] = ["context"]
    length_normalize: bool = False
    mapping_class_names_to_prompt_names: dict = {
        "entailment": "correct",
        "neutral": "inconclusive",
        "contradiction": "incorrect",
    }
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>{context} is definitely correct, incorrect, or inconclusive? Answer: {class_name}",
        },
    }


class FairFaceAgeVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/FairFace"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
            ClassifMetrics.PER_BUCKET_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    label_column_name: str = "age"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    mapping_class_names_to_prompt_names: dict = {
        "0-2": "between and 0 and 2 years old",
        "3-9": "between and 3 and 9 years old",
        "10-19": "between and 10 and 19 years old",
        "20-29": "between and 20 and 29 years old",
        "30-39": "between and 30 and 39 years old",
        "40-49": "between and 40 and 49 years old",
        "50-59": "between and 50 and 59 years old",
        "60-69": "between and 60 and 69 years old",
        "more than 70": "more than 70 years old",
    }
    buckets_keys: List = ["gender", "race"]
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a photo of a person {class_name}.",
        },
    }


class FairFaceAgeSampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    FairFaceAgeVgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/FairFace-Sample"


class FairFaceGenderVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/FairFace"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
            ClassifMetrics.PER_BUCKET_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    label_column_name: str = "gender"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    mapping_class_names_to_prompt_names: dict = {
        "Male": "male",
        "Female": "female",
    }
    buckets_keys: List = ["gender", "race"]
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a photo of a {class_name} person.",
        },
    }


class FairFaceGenderSampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    FairFaceGenderVgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/FairFace-Sample"


class FairFaceRaceVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/FairFace"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
            ClassifMetrics.PER_BUCKET_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    label_column_name: str = "race"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    mapping_class_names_to_prompt_names: dict = {
        "East Asian": "east asian",
        "Indian": "indian",
        "Black": "black",
        "White": "white",
        "Middle Eastern": "middle eastern",
        "Latino_Hispanic": "latino or hispanic",
        "Southeast Asian": "southest asian",
    }
    buckets_keys: List = ["gender", "race"]
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>a photo of a {class_name} person.",
        },
    }


class FairFaceRaceSampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    FairFaceGenderVgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/FairFace-Sample"


class NLVR2SampleVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/NLVR2-Sample"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["left_image", "right_image"]
    context_column_names: List[str] = ["sentence"]
    length_normalize: bool = False
    mapping_class_names_to_prompt_names: dict = {
        "True": "True",
        "False": "False",
    }
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": (
                "<image>{image_token}{token_around_image}{sentence} Is"
                " the preceding statement True or False? Answer: {class_name}."
            ),
        },
    }


class NLVR2NewSplitsVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/NLVR2_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"

    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    label_column_name: str = "label"
    image_column_names: List[str] = ["left_image", "right_image"]
    context_column_names: List[str] = ["sentence"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": (
                "<image>{image_token}{token_around_image}{sentence} Is"
                " the preceding statement True or False? Answer: {class_name}."
            ),
        },
    }


class ScienceQAVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/ScienceQA_modif"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    context_column_names: List[str] = ["context"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>{context}Answer: The best answer is {class_name}.",
        },
    }


class ScienceQANewSplitsVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/ScienceQA_support_query_sets"
    default_query_split_name: str = "validation_query_set"
    default_support_split_name: str = "validation_support_set"
    validation_query_split_name: str = "validation_query_set"
    validation_support_split_name: str = "validation_support_set"
    test_query_split_name: str = "test_query_set"
    test_support_split_name: str = "test_support_set"

    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    label_column_name: str = "label"
    image_column_names: List[str] = ["image"]
    context_column_names: List[str] = ["question", "hint", "context", "lecture", "solution"]
    tested_ex_excluded_context_columns: List[str] = ["lecture", "solution"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": (
                "<image>{question} {hint} {context} Answer: The correct answer is {class_name}. {lecture} {solution}"
            ),
        },
    }


class ScienceQASampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    ScienceQAVgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/ScienceQA_modif-Sample"


class IIIT5KVgpt2ClassificationInContextAccWithKLAndEntropy(Vgpt2ClassificationInContext):
    dataset_name: str = "HuggingFaceM4/IIIT-5K-classif"
    metric_name: str = "UnfoldedClassificationMetrics"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    default_query_split_name: str = "test"
    default_support_split_name: str = "train"
    label_column_name: str = "label"
    tested_labels_column_name: str = "small_lexicon"
    image_column_names: List[str] = ["image"]
    length_normalize: bool = False
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": '<image>"{class_name}" is written on the picture.',
        },
    }


class IIIT5KSampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    IIIT5KVgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/IIIT-5K-classif-Sample"


class SimpleImageNet1kVgpt2ClassificationInContextAccWithKLAndEntropy(
    ImageNet1kVgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/simple-imagenet-1k"
    metric_kwargs = {
        "metrics": [
            ClassifMetrics.ACCURACY,
            ClassifMetrics.KL_DISTRIBUTION,
            ClassifMetrics.KL_MEAN,
            ClassifMetrics.ENTROPY_DISTRIBUTION,
            ClassifMetrics.ENTROPY_MEAN,
        ]
    }
    tested_labels_column_name: str = "lexicon"


class SimpleImageNet1kSampleVgpt2ClassificationInContextAccWithKLAndEntropy(
    SimpleImageNet1kVgpt2ClassificationInContextAccWithKLAndEntropy
):
    dataset_name: str = "HuggingFaceM4/simple-imagenet-1k-Sample"
