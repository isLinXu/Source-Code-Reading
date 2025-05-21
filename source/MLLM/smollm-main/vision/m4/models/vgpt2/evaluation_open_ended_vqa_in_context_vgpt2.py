import os
import random
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
from accelerate.utils import extract_model_from_parallel
from datasets import Dataset
from deepspeed.runtime.engine import DeepSpeedEngine
from transformers import AutoTokenizer

from m4.evaluation.config import ShotSelectionMode
from m4.evaluation.custom_metrics.open_ended_vqa_metrics import OEVQAMetrics
from m4.evaluation.tasks import BaseTaskOpenEndedVQA, Predictor
from m4.evaluation.utils import EvaluationVersion
from m4.training.packing import get_splitted_images_and_corresponding_text
from m4.training.utils import (
    FAKE_TOKEN_AROUND_IMAGE_V1,
    FAKE_TOKEN_AROUND_IMAGE_V2,
    IMAGE_TOKEN,
    build_image_transform,
)


class Vgpt2OpenEndedVQAInContext(BaseTaskOpenEndedVQA):
    model_class: str = "VGPT2LMHeadModel"
    predictor_class: Predictor = Predictor.in_contexter
    stop_words = [
        "Question",
        "User",
        "Image",
        "task",
        "What",
        "Who",
        "When",
        "Where",
        "Why",
        "How",
        "<end_of_utterance>",
        "<|im_end|>",
        "<row_",
        "row_1",
        "\u2500lrow_",
        "apiro",
    ]
    buckets_keys: List[str] = []
    tokenizer_max_seq_len = 1024
    prompt_templates_dict = {
        0: {
            "prefix": None,
            "example": "<image>Question:{question} Answer: {answer}\n",
        },
        1: {
            "prefix": "{bos_token}",
            "example": "<image>Question: {question} Answer: {answer}\n",
        },
        2: {
            "prefix": (
                "{bos_token}This is a conversation between a human, User, and an intelligent visual AI, Bot. The"
                " user sends images accompanied by questions, and Bot answers the questions from the user.\n"
            ),
            "example": "User:<image>{question} Bot: {answer}\n",
        },
        3: {
            "prefix": (
                "{bos_token}This is a conversation between a human, User, and an intelligent visual AI, Bot. The"
                " user sends images accompanied by questions, and Bot answers the questions from the user. The bot"
                " should reply as concisely as possible.\n"
            ),
            "example": "User:<image>{question} Bot: {answer}\n",
        },
        4: {
            "prefix": "{bos_token}",
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        5: {
            "prefix": (
                "{bos_token}You are an intelligent assistant, following instructions and answering questions as"
                " accurately as possible.\nHere, the task is to provide a short answer to the question about the"
                " input image.\n"
            ),
            "example": "<image>Question: {question}\nAnswer: {answer}\n",
        },
        6: {
            "prefix": None,
            "example": "{bos_token}<image>Question: {question} Answer: {answer}{eos_token}",
        },
        7: {
            "prefix": (
                "{bos_token}This is an intelligent virtual assistant. Its main purpose is to follow instructions"
                " and answer questions as accurately as possible.\nHere, the task is to provide a short answer to"
                " a question about an input image.\n"
            ),
            "example": "<image>Question: {question}\nAnswer: {answer}\n",
        },
        8: {
            "prefix": "{bos_token}Provide a short answer to the question.\n",
            "example": "Image:<image>\nQuestion: {question}\nAnswer: {answer}\n",
        },
        9: {
            "prefix": "{bos_token}Instruction: provide an answer to the question. Use the image to answer.\n",
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        10: {
            "prefix": "{bos_token}Answer the question using the image provided as a reference.\n",
            "example": "Reference image:<image>\nQuestion: {question}\nAnswer: {answer}\n",
        },
        11: {
            "prefix": "{bos_token}",
            "example": "<image>Question: {question} Answer: {answer}\n",
        },
        12: {
            "prefix": "{bos_token}",
            "example": "Image:<image>Question: {question} Answer: {answer}\n",
        },
        13: {
            "prefix": "",
            "example": "{bos_token}<image>Question: {question}\nAnswer: {answer}\n",
        },
        14: {
            "prefix": "{bos_token}",
            "example": "User: {question} {answer}\n",
        },
        15: {
            "prefix": "{bos_token}",
            "example": "User:<image>{question} {answer}\n",
        },
    }
    prompt_templates_dict_instruct = {}
    bool_instruct_templates = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        print(kwargs)
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
        pass

    def get_data_collator(self, **kwargs):
        def data_collator(batch):
            exs = {key: [ex[key] for ex in batch] for key in batch[0].keys()}
            batch = self.prepare_dataset(exs, **kwargs)
            return batch

        return data_collator

    def prepare_dataset(self, exs: Dict, **kwargs) -> Dict:
        """
        Prepare batch of examples.
        """
        support_dataset: Dataset = kwargs["support_dataset"]
        support_dataset_vision_encoder_embeddings: Optional[np.ndarray] = kwargs.get(
            "support_dataset_vision_encoder_embeddings", None
        )
        num_shots: int = kwargs["num_shots"]
        shot_selection_mode: ShotSelectionMode = kwargs["shot_selection_mode"]
        prompt_template_id: int = kwargs["prompt_template_id"]

        nb_exs = len(exs["id"])
        multiple_images_dataset = isinstance(support_dataset[0][self.image_column_name], list)

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
        elif shot_selection_mode == ShotSelectionMode.first_without_image:
            idx_shots = [list(range(num_shots)) for _ in range(nb_exs)]
        else:
            idx_shots = [
                retrieve_idx_closest_examples(ref_embedding, support_dataset_vision_encoder_embeddings, num_shots)
                for ref_embedding in exs["vision_encoder_embeddings"]
            ]

        # Prepare text shots
        # These are the priming text shots - size: batch_size
        texts_shots = [
            "".join(
                [
                    self._create_example_prompt(
                        prompt_template_id=prompt_template_id,
                        question=support_dataset[idx_shot][self.question_column_name],
                        answer=Counter(support_dataset[idx_shot][self.answers_column_name]).most_common(1)[0][0],
                        image=support_dataset[idx_shot][self.image_column_name],
                        eos_token=self.tokenizer.eos_token,
                        without_image=shot_selection_mode == ShotSelectionMode.first_without_image,
                        multiple_images_dataset=multiple_images_dataset,
                        contexts=(
                            [
                                (context_column_name, support_dataset[context_column_name][idx_shot])
                                for context_column_name in self.context_column_names
                            ]
                            if self.context_column_names
                            else None
                        ),
                    )
                    for idx_shot in idx_shots_ex
                ]
            )
            for idx_shots_ex in idx_shots
        ]

        # These are the tested example - size: batch_size
        tested_exs = [
            self._create_example_prompt(
                prompt_template_id=prompt_template_id,
                question=question,
                image=exs[self.image_column_name][idx_ex],
                eos_token="",
                multiple_images_dataset=multiple_images_dataset,
                contexts=(
                    [
                        (context_column_name, exs[context_column_name][idx_ex])
                        for context_column_name in self.context_column_names
                    ]
                    if self.context_column_names
                    else None
                ),
            ).strip()
            for idx_ex, question in enumerate(exs[self.question_column_name])
        ]
        if self.bool_instruct_templates:
            tested_exs = [ex[: -len("<end_of_utterance>\n")].strip() for ex in tested_exs]

        # These are the concatenation of the priming text shots and tested example - size: batch_size
        tot_texts = [
            self._create_prefix_prompt(prompt_template_id=prompt_template_id) + text_shot + tested_ex
            for text_shot, tested_ex in zip(texts_shots, tested_exs)
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
        input_ids = [tokens.input_ids[idx] for idx in range(len(tot_texts))]
        attention_mask = [tokens.attention_mask[idx] for idx in range(len(tot_texts))]

        # Prepare image shots
        # These are the priming image shots - size: batch_size
        if shot_selection_mode == ShotSelectionMode.first_without_image:
            pixel_values_shots = [[] for _ in range(nb_exs)]
        elif multiple_images_dataset:
            pixel_values_shots = [
                [
                    self.image_transform(sub_image)
                    for idx_shot in idx_shots_ex
                    for img in support_dataset[idx_shot][self.image_column_name]
                    for sub_image in self.simpler_get_splitted_images_and_corresponding_text(image=img)[0]
                ]
                for idx_shots_ex in idx_shots
            ]
        else:
            pixel_values_shots = [
                [
                    self.image_transform(sub_image)
                    for idx_shot in idx_shots_ex
                    for sub_image in self.simpler_get_splitted_images_and_corresponding_text(
                        image=support_dataset[idx_shot][self.image_column_name],
                    )[0]
                ]
                for idx_shots_ex in idx_shots
            ]

        # These are the tested images - size: batch_size
        if multiple_images_dataset:
            tested_pixel_values = [
                [
                    self.image_transform(sub_image)
                    for image in images
                    for sub_image in self.simpler_get_splitted_images_and_corresponding_text(image=image)[0]
                ]
                for images in exs[self.image_column_name]
            ]
        else:
            tested_pixel_values = [
                [
                    self.image_transform(sub_image)
                    for sub_image in self.simpler_get_splitted_images_and_corresponding_text(image=image)[0]
                ]
                for image in exs[self.image_column_name]
            ]

        # These are the concatenation of the priming image shots and tested images - size: batch_size
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

        example_ids: List[int] = exs["id"]
        answers = exs[self.answers_column_name]
        if self.buckets_keys:

            def bucket_infos_to_str(bucket_infos):
                name = []
                for info, info_type in zip(bucket_infos, self.buckets_keys):
                    name.append(f"{info_type}={info}")
                return "/".join(name)

            columns_to_concatenate = [exs[key] for key in self.buckets_keys]
            buckets = [bucket_infos_to_str(bucket_infos) for bucket_infos in zip(*columns_to_concatenate)]
        else:
            buckets = [""] * len(example_ids)
        return {
            "example_ids": example_ids,
            "answers": answers,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_attention_masks": pixel_attention_masks,
            "buckets": buckets,
        }

    def _create_example_prompt(
        self,
        prompt_template_id,
        question,
        image,
        eos_token,
        answer="",
        without_image=False,
        multiple_images_dataset=False,
        contexts=None,
    ):
        if self.bool_instruct_templates:
            prompt_templates_dict = self.prompt_templates_dict_instruct
        else:
            prompt_templates_dict = self.prompt_templates_dict
        prompt_template = prompt_templates_dict[prompt_template_id]["example"]
        prompt_kwargs = {}
        if contexts is not None:
            for context_column_name, additional_context in contexts:
                prompt_kwargs[f"{context_column_name}"] = additional_context
        prompt = prompt_template.format(
            bos_token=self.tokenizer.bos_token,
            eos_token=eos_token,
            # For the `eos_token`, the case is different than `bos_token`: when we include bos/eos in the shots,
            # both of them are always here (thus the usage of tokenizer.bos_token), but for the qeury example,
            # we add a `bos_token`, but not an `eos_token` to let the model continue
            question=question,
            answer=answer,
            **prompt_kwargs,
        )
        if not multiple_images_dataset:
            images = [image]
        else:
            images = image

        if without_image:
            prompt = prompt.replace("<image>", "")
            return prompt

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

    def generate_tokens(self, **kwargs):
        # Flamingo: Beam search with a beam size of 3
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

        num_beams = kwargs["num_beams"]
        no_repeat_ngram_size = kwargs["no_repeat_ngram_size"]
        max_new_tokens = kwargs["max_new_tokens"]

        bad_words = ["\n", "\n\n", self.image_token, self.token_around_image]
        bad_words_ids = self.tokenizer(bad_words, add_special_tokens=False)["input_ids"]

        unwrapped_model = extract_model_from_parallel(model)
        is_deepspeed_model = isinstance(model, DeepSpeedEngine)
        if is_deepspeed_model:
            if model.zero_optimization_partition_weights():
                # Enable automated discovery of external parameters by indicating that
                # we are in a forward pass.
                for module in model.module.modules():
                    module._parameters._in_forward = True
                    pass

        with torch.no_grad():
            generated_tokens = unwrapped_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
                max_new_tokens=max_new_tokens,
                bad_words_ids=bad_words_ids,
                use_cache=True,
                early_stopping=True,
                synced_gpus=is_deepspeed_model,
            )

        generated_tokens = generated_tokens[:, input_ids.shape[1] :]
        return generated_tokens

    def format_tokens_to_texts(self, tokens) -> List[str]:
        texts = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        stop_words_pattern = r"|".join(self.stop_words)
        texts = [re.split(stop_words_pattern, text)[0] for text in texts]
        return texts

    def add_batch_metric(self, metric, **kwargs):
        generated_tokens = self.generate_tokens(**kwargs)
        generated_texts = self.format_tokens_to_texts(generated_tokens)
        kwargs["generated_texts"] = generated_texts
        metric.add_batch(
            **{key: kwargs[key] for key in list(metric.features.keys())},
        )
        return metric


class VQAv2Vgpt2OpenEndedVQAInContextAcc(Vgpt2OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"


class VQAv2SampleVgpt2OpenEndedVQAInContextAcc(VQAv2Vgpt2OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/VQAv2_modif-Sample"


class OKVQAVgpt2OpenEndedVQAInContextAcc(Vgpt2OpenEndedVQAInContext):
    # We are considering the raw answers. In the original paper,
    # they are doing a step of stemming (standardize pluralization and conjugation).
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"


class OKVQASampleVgpt2OpenEndedVQAInContextAcc(OKVQAVgpt2OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/OK-VQA_modif-Sample"


class TextVQAVgpt2OpenEndedVQAInContextAcc(Vgpt2OpenEndedVQAInContext):
    dataset_name: str = "textvqa"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"  # List of strings


class TextVQASampleVgpt2OpenEndedVQAInContextAcc(TextVQAVgpt2OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/textvqa-Sample"


class AdVQAVgpt2OpenEndedVQAInContextAcc(Vgpt2OpenEndedVQAInContext):
    dataset_name: str = "HuggingFaceM4/AdVQA_modif"
    metric_name: str = "OpenEndedVQAMetrics"
    metric_kwargs = {
        "metrics": [
            OEVQAMetrics.FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.OE_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_FIRST_WORD_VQA_ACCURACY,
            OEVQAMetrics.GT_VISION_LAB_OE_VQA_ACCURACY,
        ]
    }
    default_query_split_name: str = "validation"
    default_support_split_name: str = "train"
    image_column_name: str = "image"
    question_column_name: str = "question"
    answers_column_name: str = "answers"


class AdVQASampleVgpt2OpenEndedVQAInContextAcc(AdVQAVgpt2OpenEndedVQAInContextAcc):
    dataset_name: str = "HuggingFaceM4/AdVQA_modif-Sample"
