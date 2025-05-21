import os
from itertools import chain
from typing import Dict, List, Optional

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer

from m4.evaluation.custom_metrics.image_caption_matching_metrics import MetricsImageCaptionMatching
from m4.evaluation.tasks import BaseTaskImageCaptionMatching, Predictor
from m4.evaluation.utils import EvaluationVersion
from m4.training.utils import (
    FAKE_TOKEN_AROUND_IMAGE_V1,
    FAKE_TOKEN_AROUND_IMAGE_V2,
    IMAGE_TOKEN,
    build_image_transform,
)


class Vgpt2ImageCaptionMatching(BaseTaskImageCaptionMatching):
    model_class: str = "VGPT2LMHeadModel"
    predictor_class: Predictor = Predictor.in_contexter
    target_keys: List[str] = ["example_ids", "caption_ids", "image_ids"]
    buckets_keys: List[str] = []
    # Buckets are optionally populated for classification in context. They are only useful when it is useful to get results bucket. A bucket is typically a certain slice of the dataset (for instance, all instances where age=30).
    mapping_class_names_to_prompt_names: Optional[Dict[str, str]] = None
    prompt_templates_dict: Dict[int, Dict[str, str]] = {}
    mapping_class_prompt_name_id_to_prompt_template_id: Optional[Dict[int, int]] = None
    tokenizer_max_seq_len = 1024

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.tokenizer_name = kwargs.pop("tokenizer_name")
        evaluation_version = kwargs.pop("evaluation_version")
        tokenizer_use_fast = kwargs.pop("tokenizer_use_fast", False)
        image_size = kwargs.pop("image_size")
        vision_encoder_type = kwargs.pop("vision_encoder_type")
        self.image_seq_len = kwargs.pop("image_seq_len")
        self.image_transform = build_image_transform(
            max_image_size=image_size, image_size=None, eval=True, vision_encoder_type=vision_encoder_type
        )

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

        nb_captions = len(self.caption_column_names)
        nb_images = len(self.image_column_names)
        self.captions_images_order_per_ex = [
            (caption_idx, image_idx) for caption_idx in range(nb_captions) for image_idx in range(nb_images)
        ]

    def get_info_from_dataset(self, dataset):
        pass

    def get_data_collator(self, **kwargs):
        def data_collator(batch):
            exs = {key: [ex[key] for ex in batch] for key in batch[0].keys()}
            batch = self.prepare_dataset(exs, **kwargs)
            return batch

        return data_collator

    def _split_array(self, array, nb_combinations):
        total_elements = len(array)
        elements_per_combination = total_elements // nb_combinations

        splitted_array = [
            array[i : i + elements_per_combination] for i in range(0, total_elements, elements_per_combination)
        ]

        return splitted_array

    def prepare_dataset(self, exs: Dict, **kwargs) -> Dict:
        """
        Prepare batch of examples.

        """

        prompt_template_id: int = kwargs["prompt_template_id"]

        nb_exs = len(exs["id"])
        nb_captions = len(self.caption_column_names)
        nb_images = len(self.image_column_names)

        # If we have caption_column_names = ["caption_0", "caption_1"] and image_column_names= ["image_0", "image_1"]. We get the sequence [caption_0, caption_0, caption_1, caption_1]
        general_dict = {"tested_prompts": [], "caption_ids": [], "image_ids": [], "ex_ids": []}
        for idx_ex in range(nb_exs):
            for caption_idx, caption_column in enumerate(self.caption_column_names):
                for image_idx in range(nb_images):
                    tested_prompt = self._create_example_prompt(
                        prompt_template_id=prompt_template_id,
                        caption=exs[caption_column][idx_ex],
                    )
                    general_dict["tested_prompts"].append(tested_prompt)
                    general_dict["caption_ids"].append(caption_idx)
                    general_dict["image_ids"].append(image_idx)
                    general_dict["ex_ids"].append(exs["id"][idx_ex])

        tot_texts = [
            self._create_prefix_prompt(prompt_template_id=prompt_template_id) + tested_prompt
            for tested_prompt in general_dict["tested_prompts"]
        ]

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

        general_dict["input_ids"] = [tokens.input_ids[idx] for idx in range(len(tot_texts))]
        general_dict["attention_mask"] = [tokens.attention_mask[idx] for idx in range(len(tot_texts))]

        # If we have caption_column_names = ["caption_0", "caption_1"] and image_column_names= ["image_0", "image_1"]. We get the sequence image_0, image_1, image_0, image_1
        pixel_values_dict = {"pixel_values": [], "caption_ids": [], "image_ids": [], "ex_ids": []}
        for idx_ex in range(nb_exs):
            for caption_idx in range(nb_captions):
                for image_idx, col in enumerate(self.image_column_names):
                    pixel_values_dict["pixel_values"].append(self.image_transform(exs[col][idx_ex]).unsqueeze(0))
                    pixel_values_dict["caption_ids"].append(caption_idx)
                    pixel_values_dict["image_ids"].append(image_idx)
                    pixel_values_dict["ex_ids"].append(exs["id"][idx_ex])

        # ---- Sanity check ----
        assert pixel_values_dict["ex_ids"] == general_dict["ex_ids"]
        nb_combinations = nb_captions * nb_images
        sample_pixel_captions_ids = pixel_values_dict["caption_ids"][:nb_combinations]
        sample_pixel_image_ids = pixel_values_dict["image_ids"][:nb_combinations]
        sample_general_captions_ids = general_dict["caption_ids"][:nb_combinations]
        sample_general_image_ids = general_dict["image_ids"][:nb_combinations]
        self.captions_images_order_per_ex
        for idx in range(nb_combinations):
            expected_caption_idx, expected_image_idx = self.captions_images_order_per_ex[idx]
            assert sample_pixel_captions_ids[idx] == expected_caption_idx
            assert sample_general_captions_ids[idx] == expected_caption_idx
            assert sample_pixel_image_ids[idx] == expected_image_idx
            assert sample_general_image_ids[idx] == expected_image_idx
        # ---- Sanity check ----

        general_dict["ex_ids"] = self._split_array(general_dict["ex_ids"], nb_exs)
        general_dict["caption_ids"] = self._split_array(general_dict["caption_ids"], nb_exs)
        general_dict["image_ids"] = self._split_array(general_dict["image_ids"], nb_exs)
        general_dict["input_ids"] = self._split_array(general_dict["input_ids"], nb_exs)
        pixel_values_dict["pixel_values"] = self._split_array(pixel_values_dict["pixel_values"], nb_exs)
        general_dict["attention_mask"] = self._split_array(general_dict["attention_mask"], nb_exs)

        return {
            "example_ids": general_dict["ex_ids"],
            "caption_ids": general_dict["caption_ids"],
            "image_ids": general_dict["image_ids"],
            "input_ids": general_dict["input_ids"],
            "attention_mask": general_dict["attention_mask"],
            "pixel_values": pixel_values_dict["pixel_values"],
        }

    def _create_example_prompt(self, prompt_template_id, caption):
        if self.bool_instruct_templates:
            prompt_templates_dict = self.prompt_templates_dict_instruct
        else:
            prompt_templates_dict = self.prompt_templates_dict
        prompt_template = prompt_templates_dict[prompt_template_id]["example"]

        template_kwargs = {
            "bos_token": self.tokenizer.bos_token,
            "eos_token": self.tokenizer.eos_token,
            "image_token": self.image_token * self.image_seq_len,
            "token_around_image": self.token_around_image,
            "caption": caption,
        }

        prompt = prompt_template.format(**template_kwargs)
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
        input_ids = torch.stack(list(chain.from_iterable(kwargs["input_ids"]))).to(model.device)
        attention_mask = torch.stack(list(chain.from_iterable(kwargs["attention_mask"]))).to(model.device)

        pv = list(chain.from_iterable(kwargs["pixel_values"]))
        total_batch_size = len(pv)
        max_num_images = max([i.size(0) for i in pv])
        max_height = max([i.size(2) for i in pv])
        max_width = max([i.size(3) for i in pv])
        pixel_values = torch.zeros(total_batch_size, max_num_images, 3, max_height, max_width)
        pixel_attention_mask = torch.zeros(total_batch_size, max_num_images, max_height, max_width, dtype=torch.bool)
        for idx, sample_images in enumerate(pv):
            im_batch_height, im_batch_width = sample_images.size()[2:]
            pixel_values[idx, :, :, :im_batch_height, :im_batch_width] = sample_images
            pixel_attention_mask[idx, :, :im_batch_height, :im_batch_width] = True
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

    def format_model_outputs_to_predictions(self, outputs) -> torch.Tensor:
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

        nb_combinations = len(self.image_column_names) * len(self.caption_column_names)
        nb_exs = batch_size // nb_combinations
        splitted_scores_per_example = self._split_array(score_per_example.tolist(), nb_exs)

        return splitted_scores_per_example

    def add_batch_metric(self, metric, **kwargs):
        outputs = self.predict(**kwargs)
        splitted_scores_per_example = self.format_model_outputs_to_predictions(outputs)
        additional_args = {key: kwargs[key] for key in self.target_keys}
        metric.add_batch(
            splitted_scores_per_example=splitted_scores_per_example,
            **additional_args,
        )
        return metric


class WinogroundVgpt2ImageCaptionMatchingAccWithKLAndEntropy(Vgpt2ImageCaptionMatching):
    dataset_name: str = "facebook/winoground"
    metric_name: str = "ImageCaptionMatchingMetrics"
    metric_kwargs = {
        "metrics": [
            MetricsImageCaptionMatching.TEXT_SCORE,
            MetricsImageCaptionMatching.IMAGE_SCORE,
            MetricsImageCaptionMatching.GROUP_SCORE,
        ]
    }
    # support split names are never used for this dataset
    default_query_split_name: str = "test"
    default_support_split_name: str = "test"
    test_support_split_name: str = "test"
    image_column_names: List[str] = ["image_0", "image_1"]
    id_column_name: str = "id"
    caption_column_names: List[str] = ["caption_0", "caption_1"]
    length_normalize: bool = True
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "{token_around_image}{image_token}{token_around_image}{caption}",
        },
    }
