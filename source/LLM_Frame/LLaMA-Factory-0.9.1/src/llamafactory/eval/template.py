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

from dataclasses import dataclass  # 导入数据类装饰器
from typing import Dict, List, Sequence, Tuple  # 导入类型提示

from ..data import Role  # 导入角色枚举
from ..extras.constants import CHOICES  # 导入选项常量


@dataclass
class EvalTemplate:
    """评估模板类，用于格式化评估数据"""
    system: str  # 系统提示语
    choice: str  # 选项格式
    answer: str  # 答案提示语

    def _parse_example(self, example: Dict[str, str]) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"question", "A", "B", "C", "D", "answer"}
        output: a tuple of (prompt, response)
        输入：包含问题、选项和答案的字典
        输出：提示语和回答的元组
        """
        # 格式化所有选项
        candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]
        # 返回完整的问题文本和答案
        return "".join([example["question"]] + candidates + [self.answer]), example["answer"]

    def format_example(
        self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]], subject_name: str
    ) -> List[Dict[str, str]]:
        r"""
        Converts dataset examples to messages.
        将数据集示例转换为消息格式
        """
        messages = []  # 初始化消息列表
        # 处理支持集（few-shot示例）
        for k in range(len(support_set)):
            prompt, response = self._parse_example(support_set[k])  # 解析示例
            messages.append({"role": Role.USER.value, "content": prompt})  # 添加用户消息
            messages.append({"role": Role.ASSISTANT.value, "content": response})  # 添加助手消息

        # 处理目标数据
        prompt, response = self._parse_example(target_data)
        messages.append({"role": Role.USER.value, "content": prompt})  # 添加用户消息
        messages.append({"role": Role.ASSISTANT.value, "content": response})  # 添加助手消息
        # 在第一条消息前添加系统提示语
        messages[0]["content"] = self.system.format(subject=subject_name) + messages[0]["content"]
        return messages


# 存储所有评估模板的字典
eval_templates: Dict[str, "EvalTemplate"] = {}


def _register_eval_template(name: str, system: str, choice: str, answer: str) -> None:
    """注册评估模板的私有函数"""
    eval_templates[name] = EvalTemplate(system=system, choice=choice, answer=answer)


def get_eval_template(name: str) -> "EvalTemplate":
    """获取指定名称的评估模板"""
    eval_template = eval_templates.get(name, None)
    assert eval_template is not None, f"Template {name} does not exist."  # 确保模板存在
    return eval_template


# 注册英文评估模板
_register_eval_template(
    name="en",  # 英文模板名称
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",  # 系统提示语
    choice="\n{choice}. {content}",  # 选项格式
    answer="\nAnswer:",  # 答案提示语
)


# 注册中文评估模板
_register_eval_template(
    name="zh",  # 中文模板名称
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",  # 系统提示语
    choice="\n{choice}. {content}",  # 选项格式
    answer="\n答案：",  # 答案提示语
)
