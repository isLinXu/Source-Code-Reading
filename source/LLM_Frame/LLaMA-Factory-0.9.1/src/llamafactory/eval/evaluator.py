# Copyright 2024 the LlamaFactory team.
#
# This code is inspired by the Dan's test library.
# https://github.com/hendrycks/test/blob/master/evaluate_flan.py
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
#
# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import CHOICES, SUBJECTS
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_eval_template


if TYPE_CHECKING:
    from numpy.typing import NDArray


class Evaluator:
    """评估器类，用于模型评估"""
    
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        # 初始化评估器
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)  # 获取各种参数
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]  # 加载分词器
        self.tokenizer.padding_side = "right"  # 设置右侧填充，避免llama2批处理推理中的溢出问题
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)  # 获取并修复模板
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)  # 加载模型
        self.eval_template = get_eval_template(self.eval_args.lang)  # 获取评估模板
        # 获取选项的token ID
        self.choice_inputs = [self.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in CHOICES]

    @torch.inference_mode()  # 推理模式装饰器，禁用梯度计算
    def batch_inference(self, batch_input: Dict[str, "torch.Tensor"]) -> List[str]:
        """批量推理方法"""
        logits = self.model(**batch_input).logits  # 获取模型输出的logits
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)  # 计算每个序列的实际长度
        # 获取每个序列最后一个token的logits
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        # 计算选项的概率分布
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        # 返回概率最高的选项（A、B、C、D）
        return [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]

    def eval(self) -> None:
        """评估方法"""
        # 解析评估任务和分割
        eval_task = self.eval_args.task.split("_")[0]
        eval_split = self.eval_args.task.split("_")[1]

        # 加载映射文件
        mapping = cached_file(
            path_or_repo_id=os.path.join(self.eval_args.task_dir, eval_task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.hf_hub_token,
        )

        # 读取科目分类映射
        with open(mapping, encoding="utf-8") as f:
            categorys: Dict[str, Dict[str, str]] = json.load(f)

        # 初始化每个科目的正确答案数组
        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        results = {}
        
        # 遍历每个科目
        for subject in pbar:
            # 加载数据集
            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, eval_task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token,
                trust_remote_code=True,
            )
            pbar.set_postfix_str(categorys[subject]["name"])  # 更新进度条显示当前科目名称
            
            inputs, outputs, labels = [], [], []  # 初始化输入、输出和标签列表
            # 处理每个样本
            for i in trange(len(dataset[eval_split]), desc="Formatting batches", position=1, leave=False):
                # 获取支持集（few-shot学习的示例）
                support_set = (
                    dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                )
                # 格式化示例
                messages = self.eval_template.format_example(
                    target_data=dataset[eval_split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                )

                # 编码输入
                input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(messages[-1]["content"])

            # 批量处理预测
            for i in trange(
                0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
            ):
                batch_input = self.tokenizer.pad(
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                preds = self.batch_inference(batch_input)
                outputs += preds

            # 计算正确率并保存结果
            corrects = np.array(outputs) == np.array(labels)
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

        pbar.close()
        self._save_results(category_corrects, results)  # 保存评估结果

    def _save_results(self, category_corrects: Dict[str, "NDArray"], results: Dict[str, Dict[int, str]]) -> None:
        """保存评估结果的私有方法"""
        # 生成评估报告
        score_info = "\n".join(
            [
                f"{category_name:>15}: {100 * np.mean(category_correct):.2f}"
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)  # 打印评估结果
        
        # 如果指定了保存目录，则保存详细结果
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            # 保存JSON格式的详细结果
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)
            # 保存评估报告
            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)


def run_eval() -> None:
    """运行评估的入口函数"""
    Evaluator().eval()
