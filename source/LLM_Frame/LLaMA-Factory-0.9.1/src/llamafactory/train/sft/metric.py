# Copyright 2024 HuggingFace Inc., THUDM, and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library and the THUDM's ChatGLM implementation.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/summarization/run_summarization.py
# https://github.com/THUDM/ChatGLM-6B/blob/main/ptuning/main.py
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
import torch  # 导入 PyTorch 库
from transformers.utils import is_jieba_available, is_nltk_available  # 从 transformers.utils 导入 jieba 和 nltk 可用性检查函数

from ...extras.constants import IGNORE_INDEX  # 从 extras.constants 导入 IGNORE_INDEX
from ...extras.misc import numpify  # 从 extras.misc 导入 numpify 函数
from ...extras.packages import is_rouge_available  # 从 extras.packages 导入 ROUGE 可用性检查函数


if TYPE_CHECKING:  # 如果正在进行类型检查
    from transformers import EvalPrediction, PreTrainedTokenizer  # 导入 EvalPrediction 和 PreTrainedTokenizer 类型


if is_jieba_available():  # 如果 jieba 可用
    import jieba  # type: ignore  # 导入 jieba 库


if is_nltk_available():  # 如果 nltk 可用
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # 从 nltk 导入 BLEU 分数计算相关函数


if is_rouge_available():  # 如果 ROUGE 可用
    from rouge_chinese import Rouge  # 从 rouge_chinese 导入 Rouge


def eval_logit_processor(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":  # 评估 logit 处理器
    r"""
    Computes the token with the largest likelihood to reduce memory footprint.  # 计算具有最大可能性的标记以减少内存占用。
    """
    if isinstance(logits, (list, tuple)):  # 如果 logits 是列表或元组
        if logits[0].dim() == 3:  # (batch_size, seq_len, vocab_size)
            logits = logits[0]  # 取第一个元素
        else:  # moe 模型具有辅助损失
            logits = logits[1]  # 取第二个元素

    if logits.dim() != 3:  # 如果 logits 的维度不是 3
        raise ValueError("Cannot process the logits.")  # 抛出错误

    return torch.argmax(logits, dim=-1)  # 返回最大 logit 的索引


@dataclass  # 使用 dataclass 装饰器定义类
class ComputeAccuracy:  # 定义计算准确度的类
    r"""
    Computes accuracy and supports `batch_eval_metrics`.  # 计算准确度并支持批量评估指标。
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
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)  # 获取预测和标签
        for i in range(len(preds)):  # 遍历每个预测
            pred, label = preds[i, :-1], labels[i, 1:]  # 获取预测和标签
            label_mask = label != IGNORE_INDEX  # 创建标签掩码
            self.score_dict["accuracy"].append(np.mean(pred[label_mask] == label[label_mask]))  # 计算准确度并添加到字典

        if compute_result:  # 如果计算结果
            return self._dump()  # 导出结果


@dataclass  # 使用 dataclass 装饰器定义类
class ComputeSimilarity:  # 定义计算相似度的类
    r"""
    Computes text similarity scores and supports `batch_eval_metrics`.  # 计算文本相似度分数并支持批量评估指标。

    Wraps the tokenizer into metric functions, used in CustomSeq2SeqTrainer.  # 将分词器包装到指标函数中，用于自定义 Seq2SeqTrainer。
    """

    tokenizer: "PreTrainedTokenizer"  # 定义分词器属性

    def _dump(self) -> Optional[Dict[str, float]]:  # 导出当前的相似度数据
        result = None  # 初始化结果
        if hasattr(self, "score_dict"):  # 如果存在 score_dict 属性
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}  # 计算每个指标的平均值

        self.score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}  # 重置 score_dict
        return result  # 返回结果

    def __post_init__(self):  # 后初始化方法
        self._dump()  # 调用导出方法

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:  # 调用方法
        preds, labels = numpify(eval_preds.predictions), numpify(eval_preds.label_ids)  # 获取预测和标签

        preds = np.where(preds != IGNORE_INDEX, preds, self.tokenizer.pad_token_id)  # 处理预测结果
        labels = np.where(labels != IGNORE_INDEX, labels, self.tokenizer.pad_token_id)  # 处理标签

        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)  # 解码预测
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)  # 解码标签

        for pred, label in zip(decoded_preds, decoded_labels):  # 遍历解码后的预测和标签
            hypothesis = list(jieba.cut(pred))  # 使用 jieba 切分预测
            reference = list(jieba.cut(label))  # 使用 jieba 切分标签

            if len(" ".join(hypothesis).split()) == 0 or len(" ".join(reference).split()) == 0:  # 如果任一切分结果为空
                result = {"rouge-1": {"f": 0.0}, "rouge-2": {"f": 0.0}, "rouge-l": {"f": 0.0}}  # 设置默认相似度分数为 0
            else:
                rouge = Rouge()  # 创建 Rouge 实例
                scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))  # 计算 Rouge 分数
                result = scores[0]  # 获取第一个评分结果

            for k, v in result.items():  # 遍历结果
                self.score_dict[k].append(round(v["f"] * 100, 4))  # 将分数添加到字典中

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)  # 计算 BLEU 分数
            self.score_dict["bleu-4"].append(round(bleu_score * 100, 4))  # 将 BLEU 分数添加到字典中

        if compute_result:  # 如果计算结果
            return self._dump()  # 导出结果
