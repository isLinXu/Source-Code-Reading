import os

import torch

from .base import BaseTrainingRecipe
from . import register_training_recipe
from ..utils import log
from ..utils import get_state_maybe_zero_3
from ..model import TinyLlavaConfig, TinyLlavaForConditionalGeneration

# 使用register_training_recipe装饰器注册一个名为'common'的训练配方
@register_training_recipe('common')
# 定义一个名为CommonTrainingRecipe的类，继承自BaseTrainingRecipe
class CommonTrainingRecipe(BaseTrainingRecipe):
    ... 
