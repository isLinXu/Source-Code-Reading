# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass  # 从 dataclasses 导入 dataclass 装饰器
from typing import TYPE_CHECKING, Dict, Optional  # 导入类型检查、字典和可选类型

import numpy as np  # 导入 NumPy 库

from ...extras.misc import numpify  # 从 extras.misc 导入 numpify 函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import EvalPrediction  # 从 transformers 导入 EvalPrediction 类型


@dataclass  # 使用 dataclass 装饰器定义类
class ComputeAccuracy:  # 定义计算准确度的类
    r"""
    Computes reward accuracy and supports `batch_eval_metrics`.  # 计算奖励准确度并支持批量评估指标。
    """

    def _dump(self) -> Optional[Dict[str, float]]:  # 导出当前的准确度数据
        result = None  # 初始化结果
        if hasattr(self, "score_dict"):  # 如果存在 score_dict 属性
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}  # 计算每个指标的平均值

        self.score_dict = {"accuracy": []}  # 重置 score_dict
        return result  # 返回结果

    def __post_init__(self):  # 后初始化方法
        self._dump()  # 调用导出方法

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:  # 调用方法
        chosen_scores, rejected_scores = numpify(eval_preds.predictions[0]), numpify(eval_preds.predictions[1])  # 获取选择和拒绝的分数
        if not chosen_scores.shape:  # 如果选择的分数没有形状
            self.score_dict["accuracy"].append(chosen_scores > rejected_scores)  # 计算准确度并添加到字典
        else:
            for i in range(len(chosen_scores)):  # 遍历每个选择的分数
                self.score_dict["accuracy"].append(chosen_scores[i] > rejected_scores[i])  # 计算准确度并添加到字典

        if compute_result:  # 如果计算结果
            return self._dump()  # 导出结果