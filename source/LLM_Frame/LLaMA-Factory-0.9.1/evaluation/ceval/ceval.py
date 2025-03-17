# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

import os  # 文件路径处理
import datasets  # HuggingFace数据集库
import pandas as pd  # 数据处理库


# 引用信息
_CITATION = """\
@article{huang2023ceval,
  title={C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models},
  author={Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and Fu, Yao and Sun, Maosong and He, Junxian},
  journal={arXiv preprint arXiv:2305.08322},
  year={2023}
}
"""

# 数据集描述
_DESCRIPTION = """\
C-Eval is a comprehensive Chinese evaluation suite for foundation models. It consists of 13948 multi-choice questions spanning 52 diverse disciplines and four difficulty levels.
"""

# 数据集主页
_HOMEPAGE = "https://cevalbenchmark.com"

# 许可协议
_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"

# 数据集下载地址
_URL = "ceval.zip"

# 定义所有学科任务列表
task_list = [
    "computer_network",  # 计算机网络
    "operating_system",  # 操作系统
    "computer_architecture",  # 计算机组成
    "college_programming",  # 大学编程
    "college_physics",  # 大学物理
    "college_chemistry",  # 大学化学
    "advanced_mathematics",  # 高等数学
    "probability_and_statistics",  # 概率统计
    "discrete_mathematics",  # 离散数学
    "electrical_engineer",  # 电气工程师
    "metrology_engineer",  # 计量工程师
    "high_school_mathematics",  # 高中数学
    "high_school_physics",  # 高中物理
    "high_school_chemistry",  # 高中化学
    "high_school_biology",  # 高中生物
    "middle_school_mathematics",  # 初中数学
    "middle_school_biology",  # 初中生物
    "middle_school_physics",  # 初中物理
    "middle_school_chemistry",  # 初中化学
    "veterinary_medicine",  # 兽医学
    "college_economics",  # 大学经济学
    "business_administration",  # 工商管理
    "marxism",  # 马克思主义
    "mao_zedong_thought",  # 毛泽东思想
    "education_science",  # 教育科学
    "teacher_qualification",  # 教师资格
    "high_school_politics",  # 高中政治
    "high_school_geography",  # 高中地理
    "middle_school_politics",  # 初中政治
    "middle_school_geography",  # 初中地理
    "modern_chinese_history",  # 中国近现代史
    "ideological_and_moral_cultivation",  # 思想道德修养
    "logic",  # 逻辑学
    "law",  # 法学
    "chinese_language_and_literature",  # 中国语言文学
    "art_studies",  # 艺术学
    "professional_tour_guide",  # 导游专业
    "legal_professional",  # 法律职业
    "high_school_chinese",  # 高中语文
    "high_school_history",  # 高中历史
    "middle_school_history",  # 初中历史
    "civil_servant",  # 公务员
    "sports_science",  # 体育科学
    "plant_protection",  # 植物保护
    "basic_medicine",  # 基础医学
    "clinical_medicine",  # 临床医学
    "urban_and_rural_planner",  # 城乡规划师
    "accountant",  # 会计师
    "fire_engineer",  # 消防工程师
    "environmental_impact_assessment_engineer",  # 环评工程师
    "tax_accountant",  # 税务师
    "physician",  # 医师
]


# 自定义数据集配置类
class CevalConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)  # 设置数据集版本


# 主数据集构建类
class Ceval(datasets.GeneratorBasedBuilder):
    # 生成所有任务的配置
    BUILDER_CONFIGS = [
        CevalConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        """定义数据集特征结构"""
        features = datasets.Features(
            {
                "id": datasets.Value("int32"),  # 题目ID
                "question": datasets.Value("string"),  # 问题描述
                "A": datasets.Value("string"),  # 选项A
                "B": datasets.Value("string"),  # 选项B
                "C": datasets.Value("string"),  # 选项C
                "D": datasets.Value("string"),  # 选项D
                "answer": datasets.Value("string"),  # 正确答案
                "explanation": datasets.Value("string"),  # 答案解析
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,  # 数据集描述
            features=features,  # 特征结构
            homepage=_HOMEPAGE,  # 主页链接
            license=_LICENSE,  # 许可协议
            citation=_CITATION,  # 引用信息
        )

    def _split_generators(self, dl_manager):
        """生成数据集划分"""
        data_dir = dl_manager.download_and_extract(_URL)  # 下载并解压数据集
        task_name = self.config.name  # 当前任务名称
        return [
            # 测试集划分
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test", f"{task_name}_test.csv"),
                },
            ),
            # 验证集划分
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "val", f"{task_name}_val.csv"),
                },
            ),
            # 训练集划分
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev", f"{task_name}_dev.csv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        """生成数据样本"""
        df = pd.read_csv(filepath, encoding="utf-8")  # 读取CSV文件
        for i, instance in enumerate(df.to_dict(orient="records")):  # 转换为字典格式
            # 处理缺失字段
            if "answer" not in instance.keys():
                instance["answer"] = ""  # 填充空答案
            if "explanation" not in instance.keys():
                instance["explanation"] = ""  # 填充空解析
            yield i, instance  # 生成索引和数据实例
