import dataclasses
import gc
import json
import logging
import math
import signal
from contextlib import contextmanager
from enum import Enum
from functools import partial

import accelerate
import albumentations as alb
import cv2
import numpy as np
import psutil
import pynvml
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from accelerate.state import AcceleratorState
from peft.utils import _get_submodules
from PIL import Image
from transformers import (  # AddedToken is needed for the eval of the tokenizer params # noqa: F401
    AddedToken,
    AutoTokenizer,
)

from m4.training.types import DatasetNames
from m4.utils.check_valid_tokenizer import check_valid_tokenizer


IMAGE_TOKEN = "<image>"
FAKE_TOKEN_AROUND_IMAGE_V2 = "<fake_token_around_image>"
FAKE_TOKEN_AROUND_IMAGE_V1 = "\n\n"
END_OF_UTTERANCE_TOKEN = "<end_of_utterance>"
# Originally taken from the values used in OpenCLIP
IMAGE_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
logger = logging.getLogger(__name__)


class LoggingTypes(Enum):
    """Types of logging to use for the gradient and parameter statistics"""

    JSONL = "jsonl"
    WANDB = "wandb"
    PRINT = "print"


class VisionEncoderTypes(Enum):
    """Types of vision encoders"""

    siglip = "siglip"
    clip = "clip"
    vit = "vit"


class JSONEncoderForDataclasses(json.JSONEncoder):
    """
    Use to serialize dataclass object, like so:
    json.dump(data, fp, indent=2, cls=JSONEncoderForDataclasses)
    """

    def default(self, obj):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


def freeze_model(model, module_exceptions=[]):
    mapping = {
        "LayerNorm": nn.LayerNorm,
        "Linear": nn.Linear,
        "Embedding": nn.Embedding,
    }
    module_exceptions_mapped = [mapping[m] for m in module_exceptions]
    for module in model.modules():
        if module_exceptions and any([isinstance(module, t) for t in module_exceptions_mapped]):
            module.requires_grad_(True)  # Explicitly setting it to true to avoid any mistakes
        else:
            module.requires_grad_(False)
    return model


# Code copied from Nougat repo: https://github.com/facebookresearch/nougat/blob/main/nougat/transforms.py
# Implements image augmentation


def alb_wrapper(transform):
    def f(im):
        return transform(image=np.asarray(im))["image"]

    return f


class Erosion(alb.ImageOnlyTransform):
    """
    Apply erosion operation to an image.

    Erosion is a morphological operation that shrinks the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the erosion kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the erosion kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.erode(img, kernel, iterations=1)
        return img


class Dilation(alb.ImageOnlyTransform):
    """
    Apply dilation operation to an image.

    Dilation is a morphological operation that expands the white regions in a binary image.

    Args:
        scale (int or tuple/list of int): The scale or range for the size of the dilation kernel.
            If an integer is provided, a square kernel of that size will be used.
            If a tuple or list is provided, it should contain two integers representing the minimum
            and maximum sizes for the dilation kernel.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, scale, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        if type(scale) is tuple or type(scale) is list:
            assert len(scale) == 2
            self.scale = scale
        else:
            self.scale = (scale, scale)

    def apply(self, img, **params):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(np.random.randint(self.scale[0], self.scale[1], 2))
        )
        img = cv2.dilate(img, kernel, iterations=1)
        return img


class Bitmap(alb.ImageOnlyTransform):
    """
    Apply a bitmap-style transformation to an image.

    This transformation replaces all pixel values below a certain threshold with a specified value.

    Args:
        value (int, optional): The value to replace pixels below the threshold with. Default is 0.
        lower (int, optional): The threshold value below which pixels will be replaced. Default is 200.
        always_apply (bool, optional): Whether to always apply this transformation. Default is False.
        p (float, optional): The probability of applying this transformation. Default is 0.5.

    Returns:
        numpy.ndarray: The transformed image.
    """

    def __init__(self, value=0, lower=200, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.lower = lower
        self.value = value

    def apply(self, img, **params):
        img = img.copy()
        img[img < self.lower] = self.value
        return img


nougat_transform = alb_wrapper(
    alb.Compose(
        [
            Bitmap(p=0.05),
            alb.OneOf([Erosion((2, 3)), Dilation((2, 3))], p=0.02),
            alb.Affine(shear={"x": (0, 3), "y": (-3, 0)}, cval=(255, 255, 255), p=0.03),
            alb.ShiftScaleRotate(
                shift_limit_x=(0, 0.04),
                shift_limit_y=(0, 0.03),
                scale_limit=(-0.15, 0.03),
                rotate_limit=2,
                border_mode=0,
                interpolation=2,
                value=(255, 255, 255),
                p=0.03,
            ),
            alb.GridDistortion(
                distort_limit=0.05,
                border_mode=0,
                interpolation=2,
                value=(255, 255, 255),
                p=0.04,
            ),
            alb.Compose(
                [
                    alb.Affine(translate_px=(0, 5), always_apply=True, cval=(255, 255, 255)),
                    alb.ElasticTransform(
                        p=1,
                        alpha=50,
                        sigma=120 * 0.1,
                        alpha_affine=120 * 0.01,
                        border_mode=0,
                        value=(255, 255, 255),
                    ),
                ],
                p=0.04,
            ),
            alb.RandomBrightnessContrast(0.1, 0.1, True, p=0.03),
            alb.ImageCompression(95, p=0.07),
            alb.GaussNoise(20, p=0.08),
            alb.GaussianBlur((3, 3), p=0.03),
        ]
    )
)


def _convert_to_rgb(image):
    # `image.convert("RGB")` would only work for .jpg images, as it creates
    # a wrong background for transparent images. The call to `alpha_composite`
    # handles this case
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    alpha_composite = alpha_composite.convert("RGB")
    return alpha_composite


class ConditionalResize(object):
    def __init__(
        self,
        max_image_size,
        min_image_size=378,
        scale_up_frequency=0.7,
        scale_up_max=3.0,
        interpolation=transforms.InterpolationMode.BILINEAR,
        eval=False,
    ):
        self.scale_up_frequency = scale_up_frequency
        self.scale_up_max = scale_up_max
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.interpolation = interpolation
        self.eval = eval

        if not eval and scale_up_frequency is not None and scale_up_max is None:
            raise ValueError(
                f"`scale_up_max` ({scale_up_max}) cannot be None if images are going to be resizes on a"
                f" `scale_up_frequency` ({scale_up_frequency}) frequency basis"
            )

    def __call__(self, image, scale_up_factor=None):
        new_width, new_height = image.size
        aspect_ratio = new_width / new_height
        if scale_up_factor is not None:
            new_width, new_height = int(scale_up_factor * new_width), int(scale_up_factor * new_height)
        elif not self.eval and self.scale_up_frequency is not None and np.random.random() < self.scale_up_frequency:
            # During training, we artificially scale up the images X% of the time to have more representation of large images
            scale_up_factor = np.random.triangular(left=1.5, mode=self.scale_up_max, right=self.scale_up_max)
            new_width, new_height = int(scale_up_factor * new_width), int(scale_up_factor * new_height)

        # Calculate the new size with respect to the aspect ratio
        if new_width >= new_height and new_width > self.max_image_size:
            new_width = self.max_image_size
            new_height = int(new_width / aspect_ratio)
        elif new_height > new_width and new_height > self.max_image_size:
            new_height = self.max_image_size
            new_width = int(new_height * aspect_ratio)

        if new_height > self.max_image_size or new_width > self.max_image_size:
            raise ValueError(
                f"Image size exceeds the maximum size of {self.max_image_size} with the new size of"
                f" ({new_width}, {new_height})"
            )

        # new_width = max(new_width, self.min_image_size)
        # new_height = max(new_height, self.min_image_size)
        new_width = self.max_image_size
        new_height = self.max_image_size

        resized_image = transforms.Resize(
            size=(new_height, new_width),
            interpolation=self.interpolation,
        )(image)
        return resized_image


# TODO(aps): Take parameters from config
def build_image_transform(
    image_size=384,
    min_image_size=378,
    max_image_size=None,
    eval=False,
    vision_encoder_type=VisionEncoderTypes.siglip,
    dataset_name=None,
    scale_up_frequency=None,
    scale_up_max=None,
):
    if image_size is not None and max_image_size is not None:
        raise ValueError(
            f"`image_size` ({image_size}) and `max_image_size` ({max_image_size}) can't be specified at the same."
        )
    if image_size is not None and (scale_up_frequency is not None or scale_up_max is not None):
        raise ValueError(
            "Both `scale_up_frequency` and `scale_up_max` should only be specified when images are systematically"
            " resized to squares."
        )

    if vision_encoder_type == VisionEncoderTypes.clip or vision_encoder_type == VisionEncoderTypes.vit:
        ops_to_compose = [_convert_to_rgb]
        if eval:
            ops_to_compose.append(
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC)
            )
        else:
            if dataset_name in [DatasetNames.OCR, DatasetNames.DOCVQA]:
                ops_to_compose.extend(
                    [
                        transforms.Resize(
                            (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
                        ),
                        nougat_transform,
                    ]
                )
            else:
                ops_to_compose.append(
                    transforms.RandomResizedCrop(
                        (image_size, image_size),
                        scale=(0.9, 1.0),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    )
                )
        ops_to_compose.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGE_DATASET_MEAN, std=IMAGE_DATASET_STD),
            ]
        )
        transform = transforms.Compose(ops_to_compose)

    elif vision_encoder_type == VisionEncoderTypes.siglip:

        def transform(img, scale_up_factor=None):
            conditional_resize = ConditionalResize(
                max_image_size=max_image_size,
                min_image_size=min_image_size,
                scale_up_frequency=scale_up_frequency,
                scale_up_max=scale_up_max,
                interpolation=transforms.InterpolationMode.BILINEAR,
                eval=eval,
            )
            ops_to_compose = [
                _convert_to_rgb,
                (
                    partial(conditional_resize, scale_up_factor=scale_up_factor)
                    if max_image_size is not None
                    else transforms.Resize(
                        (image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR
                    )
                ),
            ]
            # if dataset_name in [DatasetNames.OCR, DatasetNames.DOCVQA] and not eval:
            #     ops_to_compose.append(nougat_transform)
            ops_to_compose.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
            transformed_img = transforms.Compose(ops_to_compose)(img=img)
            return transformed_img

    else:
        raise ValueError("Wrong `vision_encoder_type`.")

    return transform


def image_splitting(
    image,
    vision_encoder_max_image_size,
    max_image_size,
    pre_split_scale_up_max=1.0,
    pre_split_scale_up_frequency=0.0,
    scale_up_factor=None,
):
    """
    Image splitting strategy.
    1) If one side of the original image is larger than `max_image_size`, resize it to `max_image_size` while preserving the aspect ratio.
    2) Divide the resulting image into `ceil(height / vision_encoder_max_image_size)` x `ceil(width / vision_encoder_max_image_size)`
    sub-images of approximately the same size each (up to the fact that `vision_encoder_max_image_size` does not divide `height` or
    `width`).
    3) [Optional] For all the crops and the original image, resize the largest side to `vision_encoder_max_image_size`, while preserving the aspect ratio.
    4) Returns the list of the crops and the original image, in addition to the number of splits for the height and the width.
    """

    width, height = image.size
    aspect_ratio = width / height

    if (scale_up_factor is not None) and (scale_up_factor != 1):
        width, height = int(scale_up_factor * width), int(scale_up_factor * height)

    elif (pre_split_scale_up_frequency is not None) and (np.random.random() < pre_split_scale_up_frequency):
        random_number = np.random.random()
        if max_image_size <= 1092:
            if random_number < 0.5:
                num_sub_images_longest_side = 1
            elif random_number < 0.8:
                num_sub_images_longest_side = 2
            else:
                num_sub_images_longest_side = 3
        else:
            if random_number < 0.4:
                num_sub_images_longest_side = 2
            elif random_number < 0.7:
                num_sub_images_longest_side = 3
            elif random_number < 0.9:
                num_sub_images_longest_side = 4
            else:
                num_sub_images_longest_side = 5
        if width >= height:
            width = num_sub_images_longest_side * vision_encoder_max_image_size
            height = int(width / aspect_ratio)
            height = math.ceil(height / vision_encoder_max_image_size) * vision_encoder_max_image_size
        else:
            height = num_sub_images_longest_side * vision_encoder_max_image_size
            width = int(height * aspect_ratio)
            width = math.ceil(width / vision_encoder_max_image_size) * vision_encoder_max_image_size


    if width >= height:
        if width > max_image_size:
            width = max_image_size
        else:
            width = math.ceil(width / vision_encoder_max_image_size) * vision_encoder_max_image_size
        height = int(width / aspect_ratio)
        height = math.ceil(height / vision_encoder_max_image_size) * vision_encoder_max_image_size
    elif height > width:
        if height > max_image_size:
            height = max_image_size
        else:
            height = math.ceil(height / vision_encoder_max_image_size) * vision_encoder_max_image_size
        width = int(height * aspect_ratio)
        width = math.ceil(width / vision_encoder_max_image_size) * vision_encoder_max_image_size

    if (width == 0) or (height == 0):
        # For some reasons it can happen (rarely) during a training. Don't know the cause.
        width, height = vision_encoder_max_image_size, vision_encoder_max_image_size

    image = image.resize((width, height), Image.LANCZOS)

    frames = []
    if height > vision_encoder_max_image_size or width > vision_encoder_max_image_size:
        # Calculate the number of splits
        num_splits_w = math.ceil(width / vision_encoder_max_image_size)
        num_splits_h = math.ceil(height / vision_encoder_max_image_size)
        # Calculate the optimal width and height for the sub-images
        optimal_width = math.ceil(width / num_splits_w)
        optimal_height = math.ceil(height / num_splits_h)

        # Iterate through each row and column
        for r in range(num_splits_h):
            for c in range(num_splits_w):
                # Calculate the starting point of the crop
                start_x = c * optimal_width
                start_y = r * optimal_height

                # Calculate the ending point of the crop
                end_x = min(start_x + optimal_width, width)
                end_y = min(start_y + optimal_height, height)

                # Crop the image
                crop = image.crop((start_x, start_y, end_x, end_y))
                frames.append(crop)

        # For the global image at the end, we resize it to match the vision_encoder_max_image_size, for cpu memory efficiency
        image = image.resize((vision_encoder_max_image_size, vision_encoder_max_image_size), Image.LANCZOS)

    else:
        num_splits_h, num_splits_w = 0, 0

    frames.append(image)

    return frames, num_splits_h, num_splits_w


def get_tokenizer(
    tokenizer_name: str,
    tokenizer_add_tokens,
    tokenizer_add_special_tokens,
    tokenizer_params,
    additional_vocab_size,
    model_vocab_size=None,
    is_fine_tuning=False,
):
    """
    We artificially separate `tokenizer_add_tokens` and `tokenizer_add_special_tokens` is a dictionary whose keys only takes into account special tokens (eos, pad, cls, etc.).
    On the contrary, `tokenizer_add_tokens` is a list of string of `AddedToken`.
    In practise, we use `tokenizer.add_special_tokens` to add all of these new special tokens or update the existing ones.

    NB: we constraint to tokenizer to be a fast tokenizer because with the slow tokenizer, we can't set the arguments of the added tokens (cf `.add_tokens`) and by default, the separators are stripped.
    """
    tokenizer_params = eval(tokenizer_params)
    assert isinstance(tokenizer_params, dict)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, legacy=False, **tokenizer_params)

    if model_vocab_size is not None:
        if model_vocab_size > len(tokenizer):
            logger.warning(
                f"The model vocabulary size ({model_vocab_size}) is larger than the tokenizer vocabulary size "
                f"({len(tokenizer)}). Updating the tokenizer to match."
            )
            if "additional_special_tokens" in tokenizer_params:
                raise ValueError(
                    "You can't use `additional_special_tokens` in `tokenizer_params` with a model vocab "
                    "size > tokenizer vocab size. We need to adjust tokenizer before adding special "
                    "tokens. Please use `tokenizer_add_tokens` instead."
                )
            # We need to pad the tokenizer vocab with fake tokens
            tokenizer.add_tokens(["<fake_token_{}>".format(i) for i in range(model_vocab_size - len(tokenizer))])

    # This check ensures that the image token and the fake token around it will be in the `DecoupledEmbedding.additional_weight`.
    existing_special_tokens = (
        [*tokenizer.special_tokens_map_extended["additional_special_tokens"]]
        if "additional_special_tokens" in tokenizer.special_tokens_map_extended
        else []
    )
    add_special_tokens_dict = {"additional_special_tokens": existing_special_tokens + eval(tokenizer_add_tokens)}
    if tokenizer_add_special_tokens is not None:
        add_special_tokens_dict.update(eval(tokenizer_add_special_tokens))

    tokenizer.add_special_tokens(add_special_tokens_dict)
    if is_fine_tuning and len(eval(tokenizer_add_tokens)) > 0:
        assert str(eval(tokenizer_add_tokens)[-1]) == END_OF_UTTERANCE_TOKEN
        assert END_OF_UTTERANCE_TOKEN in tokenizer.convert_ids_to_tokens(
            [idx for idx in range(len(tokenizer) - additional_vocab_size, len(tokenizer))]
        )
    elif not is_fine_tuning and len(eval(tokenizer_add_tokens)) > 0:
        assert str(eval(tokenizer_add_tokens)[-1]) == IMAGE_TOKEN
        assert str(eval(tokenizer_add_tokens)[-2]) == FAKE_TOKEN_AROUND_IMAGE_V2
        assert IMAGE_TOKEN in tokenizer.convert_ids_to_tokens(
            [idx for idx in range(len(tokenizer) - additional_vocab_size, len(tokenizer))]
        )
        assert FAKE_TOKEN_AROUND_IMAGE_V2 in tokenizer.convert_ids_to_tokens(
            [idx for idx in range(len(tokenizer) - additional_vocab_size, len(tokenizer))]
        )
    # This verifies that `<image>` was correctly added to the tokenizer vocabulary
    # XXX: opt-1.3b fails here
    # assert tokenizer.is_fast == tokenizer_params.get("use_fast", True)

    check_valid_tokenizer(tokenizer)

    return tokenizer


def pynmvl_handle(accelerator):
    if not torch.cuda.is_available():
        return None

    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(accelerator.local_process_index)


def pynvml_get_total_energy_in_joules(handle):
    if not torch.cuda.is_available():
        return 0
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(handle) / 1000


def compute_tflops_per_batch_per_gpu(
    num_layers,
    batch_size,
    q_seq_len,
    k_seq_len,
    hidden_size,
    kv_in_dim,
    ff_exp_factor=None,
    grad_acc_size=1,
    swiglu=False,
    vocab_size=None,
    count_backward=False,
    use_grad_checkpointing=False,
):
    multiply_add_factor = torch.tensor(2)
    query_transformation = multiply_add_factor * batch_size * q_seq_len * hidden_size**2
    # k_seq_len == v_seq_len
    key_value_transformation = multiply_add_factor * batch_size * k_seq_len * (2 * hidden_size * kv_in_dim)
    attention_matrix_computation = multiply_add_factor * batch_size * q_seq_len * k_seq_len * hidden_size
    attention_softmax = multiply_add_factor * q_seq_len * k_seq_len
    att_over_values_computation = multiply_add_factor * batch_size * q_seq_len * k_seq_len * hidden_size
    post_attention_linear_proj = multiply_add_factor * batch_size * q_seq_len * hidden_size**2

    # There are usually 2 expansion_linear_layers because first one expands, and second one retracts back to hidden_size
    # When using a classic decoder, some blocks don't have those feed-forward layers
    # Swiglu duplicates the first linear layer, so we have to account for 3 of them instead of 2
    if ff_exp_factor and swiglu:
        expansion_linear_layers = 3 * (
            multiply_add_factor * batch_size * q_seq_len * (hidden_size * ff_exp_factor) * hidden_size
        )
    elif ff_exp_factor:
        expansion_linear_layers = 2 * (
            multiply_add_factor * batch_size * q_seq_len * (hidden_size * ff_exp_factor) * hidden_size
        )
    else:
        expansion_linear_layers = torch.tensor(0)

    transformer_block_flops = (
        query_transformation
        + key_value_transformation
        + attention_matrix_computation
        + attention_softmax
        + att_over_values_computation
        + post_attention_linear_proj
        + expansion_linear_layers
    )

    # This computation should only be added if the model has a language head
    if vocab_size:
        language_head_computation = multiply_add_factor * batch_size * q_seq_len * hidden_size * vocab_size
    else:
        language_head_computation = torch.tensor(0)

    forward_fact = 1
    backward_factor = 2 if count_backward else 0
    grad_checkpointing_factor = 1 if use_grad_checkpointing else 0
    model_flops = (forward_fact + backward_factor + grad_checkpointing_factor) * (
        num_layers * transformer_block_flops + language_head_computation
    )
    model_tflops = model_flops / (10**12)

    return model_tflops


def compute_linear_tflops_per_batch_per_gpu(
    batch_size,
    seq_len,
    in_features,
    out_features,
    count_backward=False,
    use_grad_checkpointing=False,
):
    forward_factor = 1
    backward_factor = 2 if count_backward else 0
    grad_checkpointing_factor = 1 if use_grad_checkpointing else 0
    multiply_add_factor = torch.tensor(2)
    linear_forward_flops = multiply_add_factor * batch_size * seq_len * in_features * out_features
    linear_flops = linear_forward_flops * (forward_factor + backward_factor + grad_checkpointing_factor)
    linear_tflops = linear_flops / (10**12)
    return linear_tflops


def compute_perceiver_tflops_per_batch_per_gpu(
    num_layers,
    batch_size,
    q_seq_len,
    vision_embed_seq_len,
    q_k_v_input_dim,
    attention_hidden_size,
    ff_exp_factor=None,
    count_backward=False,
    use_grad_checkpointing=False,
):
    multiply_add_factor = torch.tensor(2)
    query_transformation = multiply_add_factor * batch_size * q_seq_len * q_k_v_input_dim * attention_hidden_size
    # k_seq_len == v_seq_len
    key_value_transformation = (
        multiply_add_factor * batch_size * vision_embed_seq_len * (2 * attention_hidden_size * q_k_v_input_dim)
    )

    k_seq_len = vision_embed_seq_len + q_seq_len
    attention_matrix_computation = multiply_add_factor * batch_size * q_seq_len * k_seq_len * attention_hidden_size
    attention_softmax = multiply_add_factor * q_seq_len * k_seq_len
    att_over_values_computation = multiply_add_factor * batch_size * q_seq_len * k_seq_len * attention_hidden_size
    post_attention_linear_proj = multiply_add_factor * batch_size * q_seq_len * attention_hidden_size * q_k_v_input_dim

    # There are usually 2 expansion_linear_layers because first one expands, and second one retracts back to hidden_size
    # When using a classic decoder, some blocks don't have those feed-forward layers
    if ff_exp_factor:
        expansion_linear_layers = 2 * (
            multiply_add_factor * batch_size * q_seq_len * (q_k_v_input_dim * ff_exp_factor) * q_k_v_input_dim
        )
    else:
        expansion_linear_layers = torch.tensor(0)

    transformer_block_flops = (
        query_transformation
        + key_value_transformation
        + attention_matrix_computation
        + attention_softmax
        + att_over_values_computation
        + post_attention_linear_proj
        + expansion_linear_layers
    )

    forward_fact = 1
    backward_factor = 2 if count_backward else 0
    grad_checkpointing_factor = 1 if use_grad_checkpointing else 0
    model_flops = (forward_fact + backward_factor + grad_checkpointing_factor) * (num_layers * transformer_block_flops)
    model_tflops = model_flops / (10**12)

    return model_tflops


def mem_usage_formatted(logging_type=LoggingTypes.PRINT):
    # adapted from deepspeed's see_memory_usage

    torch.cuda.empty_cache()

    # python doesn't do real-time garbage collection so do it explicitly to get the correct usage reports
    gc.collect()
    vm_stats = psutil.virtual_memory()

    mem = {
        "gpu mem alloc": f"{torch.cuda.memory_allocated()/2**30:0.2f}GB",
        "max alloc": f"{torch.cuda.max_memory_allocated()/2**30:0.2f}GB",
        "reserv": f"{torch.cuda.memory_reserved()/2**30:0.2f}GB",
        "max reserv": f"{torch.cuda.max_memory_reserved()/2**30:0.2f}GB",
        "cpu vm used": f"{(vm_stats.total-vm_stats.available)/2**30:0.2f}GB {vm_stats.percent}%",
    }

    if logging_type == LoggingTypes.PRINT:
        mem = " | ".join([f"{k}: {v}" for k, v in mem.items()]) + " | "

    # get the peak memory to report correct data, so reset the max_memory_allocated counter for the next call
    torch.cuda.reset_peak_memory_stats()

    return mem


def is_deepspeed_used():
    deepspeed_plugin = get_deepspeed_plugin()
    return deepspeed_plugin is not None


def get_deepspeed_stage():
    deepspeed_plugin = get_deepspeed_plugin()
    if deepspeed_plugin is None:
        return 0
    ds_config = deepspeed_plugin.deepspeed_config
    stage = ds_config.get("zero_optimization", {}).get("stage", 0)
    # from accelerate>=0.17.1 can do instead:
    # stage = deepspeed_plugin.zero_stage
    return stage


def is_deepspeed_zero3_used():
    return get_deepspeed_stage() == 3


def accelerate_torch_dtype():
    """
    derive and return `torch_dtype` to be used in `from_pretrained` from either Deepspeed config or if
    Deepspeed isn't used than accelerator state
    """
    if not is_accelerate_initialized():
        return None

    accelerator_state = AcceleratorState()

    if is_deepspeed_used():
        deepspeed_plugin = accelerator_state.deepspeed_plugin
        ds_config = deepspeed_plugin.deepspeed_config
        if ds_config.get("fp16", {}).get("enabled", False):
            torch_dtype = torch.float16
        elif ds_config.get("bf16", {}).get("enabled", False):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None
    else:  # no Deepspeed
        if accelerator_state.mixed_precision == "fp16":
            torch_dtype = torch.float16
        elif accelerator_state.mixed_precision == "bf16":
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None

    return torch_dtype


def is_accelerate_initialized():
    return accelerate.state.is_initialized()


def get_deepspeed_plugin():
    if is_accelerate_initialized():
        return AcceleratorState().deepspeed_plugin
    else:
        return None


def get_deepspeed_engine(accelerator):
    return accelerator.deepspeed_engine_wrapped.engine


def is_deepspeed_zero_init_enabled():
    deepspeed_plugin = get_deepspeed_plugin()
    if deepspeed_plugin is not None:
        return deepspeed_plugin.is_zero3_init_enabled()
    else:
        return False


@contextmanager
def hf_trainer_disable_zero3_init_context_manager():
    # monkey patch hack to emulate a context that has zero_init disabled as it's used in
    # modeling_utils.py in transformers for from_config and from_pretrained.
    import transformers.modeling_utils  # noqa

    orig = transformers.modeling_utils.is_deepspeed_zero3_enabled
    transformers.modeling_utils.is_deepspeed_zero3_enabled = lambda: False
    yield
    transformers.modeling_utils.is_deepspeed_zero3_enabled = orig


def deepspeed_zero_init_disabled_context_manager():
    """
    returns either a context list that includes one that will disable zero.Init or an empty context list
    """
    deepspeed_plugin = get_deepspeed_plugin()
    if deepspeed_plugin is not None:
        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    else:
        return [hf_trainer_disable_zero3_init_context_manager()]


def deepspeed_gathered_parameters_context_manager(params, modify=True):
    """
    Under zero.Init returns a context manager that will gather the sharded param, otherwise returns an empty list

    If `modify` is `True`, gather the shards and once the context exits update the shards with the
    modified data - one wants that when modifying the gathered param. If one wants to just gather
    the shards in order to read the param and no modifications are done to it, use `modify=False` as
    it's more efficient.

    `params` - can be a single parameter, a list, or a tuple of parameters to collect.

    Example:

    from transformers.utils import ContextManagers
    from m4.training.utils import deepspeed_gathered_parameters_context_manager
    with ContextManagers(deepspeed_gathered_parameters_context_manager(module.weight, modify=True)):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


    """
    if is_deepspeed_zero_init_enabled():
        import deepspeed

        # 0 is for updating `params` shards after modifying it, `None` is for read-only (only gather)
        modifier_rank = 0 if modify else None
        return [deepspeed.zero.GatheredParameters(params, modifier_rank=modifier_rank)]
    else:
        return []


# adapted from https://github.com/huggingface/transformers/blob/a081f292ca8479eaf66d7396186021268f128829/src/transformers/modeling_utils.py#L438-L496
# as it appears to be a private function
def load_state_dict_into_model(model_to_load, state_dict, start_prefix):
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: torch.nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if is_deepspeed_zero_init_enabled():
                import deepspeed

                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [named_parameters[k] for k in state_dict.keys() if k in named_parameters]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather, modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs


def get_stats(var, ctx):
    if var is None:
        return {}
    var = var.float()
    abs_var = var.abs()
    return {
        f"{ctx}_var_min": var.min().item(),
        f"{ctx}_var_max": var.max().item(),
        f"{ctx}_var_mean": var.mean().item(),
        f"{ctx}_var_std": var.std().item(),
        f"{ctx}_abs_var_min": abs_var.min().item(),
        f"{ctx}_abs_var_max": abs_var.max().item(),
        f"{ctx}_abs_var_mean": abs_var.mean().item(),
        f"{ctx}_abs_var_std": abs_var.std().item(),
        f"{ctx}_var_norm_2": (var.norm(p=2) / var.numel()).item(),
        f"{ctx}_var_norm_1": (var.norm(p=1) / var.numel()).item(),
        f"{ctx}_nonzero": (var != 0).sum().item(),
    }


def get_stats_format(ctx):
    return {
        f"{ctx}_var_min": "e",
        f"{ctx}_var_max": "e",
        f"{ctx}_var_mean": "e",
        f"{ctx}_var_std": "e",
        f"{ctx}_abs_var_min": "e",
        f"{ctx}_abs_var_max": "e",
        f"{ctx}_abs_var_mean": "e",
        f"{ctx}_abs_var_std": "e",
        f"{ctx}_var_norm_2": "e",
        f"{ctx}_var_norm_1": "e",
        f"{ctx}_nonzero": "",
    }


# Handling SIGTERM signals (in case our job gets pre-empted)
class SigtermListener:
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True


# Mostly copied from https://github.com/huggingface/peft/blob/v0.7.1/src/peft/tuners/mixed/model.py#L103
def lora_replace_module(parent, child_name, new_module, child) -> None:
    setattr(parent, child_name, new_module)
    # It's not necessary to set requires_grad here, as that is handled by
    # _mark_only_adapters_as_trainable

    # child layer wraps the original module, unpack it
    if hasattr(child, "base_layer"):
        child = child.get_base_layer()
    elif hasattr(child, "quant_linear_module"):
        # TODO maybe not necessary to have special treatment?
        child = child.quant_linear_module

    if not hasattr(new_module, "base_layer"):
        new_module.weight = child.weight
        if hasattr(child, "bias"):
            new_module.bias = child.bias

    if getattr(child, "state", None) is not None:
        if hasattr(new_module, "base_layer"):
            new_module.base_layer.state = child.state
        else:
            new_module.state = child.state
        new_module.to(child.weight.device)

    # dispatch to correct device
    for name, module in new_module.named_modules():
        if "lora" in name:
            module.to(child.weight.device)
        if "ranknum" in name:
            module.to(child.weight.device)


# Inspired by _unload_and_optionally_merge https://github.com/huggingface/peft/blob/v0.7.1/src/peft/tuners/mixed/model.py#L256
def lora_unload(model):
    key_list = [key for key, _ in model.named_modules() if "lora" not in key]
    for key in key_list:
        try:
            parent, target, target_name = _get_submodules(model, key)
        except AttributeError:
            continue

        if hasattr(target, "base_layer"):
            lora_replace_module(parent, target_name, target.get_base_layer(), target)
    return model
