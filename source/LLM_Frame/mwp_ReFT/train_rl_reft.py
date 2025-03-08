# Copyright 2023 Bytedance Ltd.
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 

#     http://www.apache.org/licenses/LICENSE-2.0 

# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
from accelerate import Accelerator, InitProcessGroupKwargs
# 从 accelerate 库导入 Accelerator 和 InitProcessGroupKwargs 类

from accelerate.utils import pad_across_processes, broadcast
# 从 accelerate.utils 导入 pad_across_processes 和 broadcast 函数

from collections import defaultdict
# 从 collections 模块导入 defaultdict 类

from dataclasses import dataclass, field, asdict
# 从 dataclasses 模块导入 dataclass, field 和 asdict 装饰器和函数

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
# 从 datasets 库导入加载数据集的相关函数和类

from datetime import timedelta
# 从 datetime 模块导入 timedelta 类

from functools import partial
# 从 functools 模块导入 partial 函数，用于部分应用

import json
# 导入 json 模块，用于处理 JSON 数据

import os
# 导入 os 模块，用于与操作系统交互

import random
# 导入 random 模块，用于生成随机数

from src.python_engine import run_python_code, process_code
# 从 src.python_engine 导入 run_python_code 和 process_code 函数

from src.utils import set_seed, floatify, compute_ETA, discount_cumsum, do_gather, allgather, allgather_masked_whiten
# 从 src.utils 导入多个实用函数

from tqdm import tqdm
# 从 tqdm 导入 tqdm，用于显示进度条

import torch
# 导入 PyTorch 库

from torch.utils.data import DataLoader
# 从 torch.utils.data 导入 DataLoader 类，用于数据加载

import deepspeed
# 导入 DeepSpeed 库，用于分布式训练

from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
# 从 transformers 库导入自动模型和调度器相关类和函数

from trl import AutoModelForCausalLMWithValueHead
# 从 trl 库导入带值头的自动因果语言模型类

from trl.core import masked_mean, masked_var, masked_whiten, logprobs_from_logits
# 从 trl.core 导入多个用于处理张量的函数

import numpy as np
# 导入 NumPy 库

import wandb
# 导入 Weights & Biases 库，用于实验跟踪

import shutil
# 导入 shutil 模块，用于文件操作

from prettytable import PrettyTable
# 从 prettytable 导入 PrettyTable 类，用于创建美观的表格

tqdm = partial(tqdm, ncols=0, leave=False)
# 使用 partial 函数设置 tqdm 的默认参数，调整进度条的外观

TIMEOUT = 10
# 设置超时时间为 10 秒

instruction=None
# 初始化指令为 None

cot_trigger=None
# 初始化 COT 触发器为 None

answer_trigger=None
# 初始化答案触发器为 None

def setup_cot(src_name):
    # 定义函数 setup_cot，接受一个源名称参数
    assert src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric']
    # 确保源名称在指定的列表中

    global instruction
    global cot_trigger
    global answer_trigger
    # 声明全局变量

    # Complete output is in this form: f'{instruction}{question.strip()}{cot_trigger}{answer_cot.strip()}'
    instruction = 'Question:\n'
    # 设置指令为 'Question:\n'

    cot_trigger = '\nAnswer reasoning:\n'
    # 设置 COT 触发器为 '\nAnswer reasoning:\n'

    answer_trigger = '\nTherefore, the answer is: '
    # 设置答案触发器为 '\nTherefore, the answer is: '

    return 
    # 返回 None

post_process_final_answer_fn_mapper = {
    'gsm8k': lambda x: float(x.replace(',','').strip()),
    # 为 'gsm8k' 定义后处理函数，将字符串转换为浮点数

    'svamp': lambda x: float(x.replace(',','').strip()),
    # 为 'svamp' 定义后处理函数，将字符串转换为浮点数

    'mathqa': lambda x: x.lower().replace('"','').replace("'",'').strip(),
    # 为 'mathqa' 定义后处理函数，处理字符串格式

    'mathqa-numeric': lambda x: float(x),
    # 为 'mathqa-numeric' 定义后处理函数，将字符串转换为浮点数
}

### the answer_cot is a list of answer_cot
post_process_answer_cot_fn_mapper = {
    ('python', 'gsm8k'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    # 为 ('python', 'gsm8k') 定义后处理函数，运行 Python 代码并将结果转换为浮点数

    ('python', 'svamp'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    # 为 ('python', 'svamp') 定义后处理函数，运行 Python 代码并将结果转换为浮点数

    ('python', 'mathqa'): lambda answer_cot: [str(res).lower().replace('"','').replace("'",'').strip() for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    # 为 ('python', 'mathqa') 定义后处理函数，处理字符串格式

    ('python', 'mathqa-numeric'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    # 为 ('python', 'mathqa-numeric') 定义后处理函数，运行 Python 代码并将结果转换为浮点数

    ('nl', 'gsm8k'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    # 为 ('nl', 'gsm8k') 定义后处理函数，处理答案列表

    ('nl', 'svamp'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    # 为 ('nl', 'svamp') 定义后处理函数，处理答案列表

    ('nl', 'mathqa'): lambda answer_cot: [res.split(answer_trigger)[-1].lower().replace('"','').replace("'",'').strip() for res in answer_cot],
    # 为 ('nl', 'mathqa') 定义后处理函数，处理答案列表

    ('nl', 'mathqa-numeric'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    # 为 ('nl', 'mathqa-numeric') 定义后处理函数，处理答案列表
}

compare_answer_fn_mapper = {
    'gsm8k': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    # 为 'gsm8k' 定义答案比较函数，检查提取答案与目标答案的差异

    'svamp': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    # 为 'svamp' 定义答案比较函数，检查提取答案与目标答案的差异

    'mathqa': lambda extracted_ans, target_answer: extracted_ans == target_answer,
    # 为 'mathqa' 定义答案比较函数，检查提取答案与目标答案是否相等

    'mathqa-numeric': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    # 为 'mathqa-numeric' 定义答案比较函数，检查提取答案与目标答案的差异
}

def prepare_deepspeed_ref_model(model):
    # Adopted from: https://github.com/huggingface/trl/blob/02f5c1d8cee73045c837d01d7f1577a57779b035/trl/trainer/ppo_trainer.py#L1399
    # 从 Hugging Face 的 TRL 库中采用的代码

    import deepspeed
    # 导入 DeepSpeed 库

    # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
    # 从 Hugging Face 的 accelerate 库中采用的代码

    deepspeed_plugin = accelerator.state.deepspeed_plugin
    # 获取 DeepSpeed 插件的状态

    config_kwargs = deepspeed_plugin.deepspeed_config
    # 获取 DeepSpeed 配置参数

    if model is not None:
        # 如果模型不为 None
        if hasattr(model, "config"):
            # 如果模型具有 config 属性
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            # 获取隐藏层大小

            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                # 如果隐藏层大小不为 None 且使用 ZeRO 优化的阶段为 3
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        # 更新配置，设置减少桶大小

                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        # 更新配置，设置阶段 3 参数持久性阈值

                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        # 更新配置，设置阶段 3 预取桶大小
                    }
                )

    # If ZeRO-3 is used, we shard both the active and reference model.
    # 如果使用 ZeRO-3，我们将活动模型和参考模型都进行分片

    # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
    # 否则，我们假设参考模型适合内存，并在每个设备上初始化，禁用 ZeRO（阶段 0）

    if config_kwargs["zero_optimization"]["stage"] != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    # 如果不是阶段 3，则将阶段设置为 0

    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    # 使用 DeepSpeed 初始化模型

    model.eval()
    # 将模型设置为评估模式

    return model
    # 返回模型

def prepare_datasets_and_data_loaders(args, tokenizer):
    # 定义函数 prepare_datasets_and_data_loaders，接受参数 args 和 tokenizer
    with accelerator.main_process_first():
        # 确保主进程优先执行

        # make raw dataset
        raw_dataset = DatasetDict({
            # 创建原始数据集
            'train': Dataset.from_list(json.load(open(args['train_file'], 'r'))),
            # 从训练文件加载数据并创建训练数据集
            'test': Dataset.from_list(json.load(open(args['test_file'], 'r'))),
            # 从测试文件加载数据并创建测试数据集
        })
        accelerator.print('Raw data:', raw_dataset)
        # 打印原始数据集

        # make cot related info
        src_name = raw_dataset['train']['item_id'][0].split('_')[0]  # e.g., gsm8k_0, gsm8k_1, gsm8k_2, ...
        # 从训练数据集中获取源名称，例如 'gsm8k'

        setup_cot(src_name)
        # 调用 setup_cot 函数设置 COT 相关信息

        accelerator.print('Using instruction:', instruction)
        # 打印使用的指令
        accelerator.print('Using cot_trigger:', cot_trigger)
        # 打印使用的 COT 触发器
        accelerator.print('Using answer_trigger:', answer_trigger)
        # 打印使用的答案触发器

        def tokenize_fn(batch, args, tokenizer):
            # 定义 tokenize_fn 函数，接受批量数据、参数和 tokenizer
            assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
            # 确保 tokenizer 的 eos_token_id 不为 None

            new_batch = defaultdict(list)
            # 创建一个新的批量字典，用于存储处理后的数据

            all_keys = list(batch.keys())
            # 获取批量数据的所有键

            for item_values in zip(*(batch[k] for k in all_keys)):
                # 遍历批量数据中的每个项
                item = {k: item_values[i] for i, k in enumerate(all_keys)}
                # 创建一个字典，将每个项的值与对应的键关联

                item_id, question, answer_value, answer_cot = \
                        item['item_id'], \
                        item['question'], \
                        item['answer_value'], \
                        item.get('answer_cot', None), \
                # 解包 item 中的相关信息

                question = question.strip()
                # 去除问题的前后空格
                if answer_value is not None:
                    answer_value = answer_value.strip()
                    # 去除答案值的前后空格

                if answer_cot:
                    answer_cot = answer_cot.strip()
                    # 去除答案 COT 的前后空格
                    if args['engine'] == 'nl':
                        answer_cot += f'{answer_trigger}{answer_value}'
                        # 如果引擎类型为 'nl'，则将答案值添加到答案 COT 中

                input = f'{instruction}{question}{cot_trigger}'
                # 构造输入字符串
                output = f'{answer_cot}'
                # 构造输出字符串
                prefix_text = f'{instruction}{question}{cot_trigger}'
                # 构造前缀文本

                # Modify for particular datasets and engine
                if src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric'] and args['engine'] == 'python':
                    prefix_text += f'def solution():\n    """{question}"""\n'
                    # 针对特定数据集和引擎类型进行修改，添加解决方案的定义

                input_encode = tokenizer(input, add_special_tokens=False)
                # 对输入进行编码
                output_encode = tokenizer(output, add_special_tokens=False)
                # 对输出进行编码
                prefix_encode = tokenizer(prefix_text, add_special_tokens=False)
                # 对前缀文本进行编码

                input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
                # 合并输入和输出的 ID，并添加结束符 ID
                labels = [-100] * len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
                # 创建标签，输入部分的标签为 -100，输出部分的标签为对应的 ID
                attention_mask = [1] * len(input_ids)
                # 创建注意力掩码
                prefix = prefix_encode['input_ids']
                # 获取前缀的 ID
                prefix_attention_mask = prefix_encode['attention_mask']
                # 获取前缀的注意力掩码

                # Truncation
                input_ids = input_ids[:args['max_input_length']]
                # 截断输入 ID
                labels = labels[:args['max_input_length']]
                # 截断标签
                attention_mask = attention_mask[:args['max_input_length']]
                # 截断注意力掩码
                prefix = prefix[:args['max_input_length']]
                # 截断前缀
                prefix_attention_mask = prefix_attention_mask[:args['max_input_length']]
                # 截断前缀注意力掩码

                ##
                new_batch['input_ids'].append(input_ids)
                # 将输入 ID 添加到新批量中
                new_batch['labels'].append(labels)
                # 将标签添加到新批量中
                new_batch['attention_mask'].append(attention_mask)
                # 将注意力掩码添加到新批量中
                new_batch['prefix'].append(prefix)
                # 将前缀添加到新批量中
                new_batch['prefix_attention_mask'].append(prefix_attention_mask)
                # 将前缀注意力掩码添加到新批量中
                ##
                new_batch['item_id'].append(item_id)
                # 将项 ID 添加到新批量中
                new_batch['question'].append(question)
                # 将问题添加到新批量中
                new_batch['prefix_text'].append(prefix_text)
                # 将前缀文本添加到新批量中
                new_batch['answer_cot'].append(answer_cot)
                # 将答案 COT 添加到新批量中
                new_batch['answer_value'].append(answer_value)
                # 将答案值添加到新批量中

            return new_batch
            # 返回处理后的新批量

        tokenized_dataset = DatasetDict({
            # 创建一个新的数据集字典
            mode: dataset.map(
                tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer}, batched=True,
                remove_columns=dataset.column_names,
                num_proc=None, load_from_cache_file=True, keep_in_memory=False,
            ) for mode, dataset in raw_dataset.items()})
        # 对原始数据集中的每个数据集应用 tokenize_fn 函数进行处理

        accelerator.print('Processed data:', tokenized_dataset)
        # 打印处理后的数据集

        if accelerator.is_main_process and args['wandb_log']:
            # 如果是主进程并且启用了 wandb 日志
            wandb.config.update({
                "src_name": src_name,
                # 更新 wandb 配置，记录源名称
                "instruction": instruction,
                # 更新 wandb 配置，记录指令
                "cot_trigger": cot_trigger,
                # 更新 wandb 配置，记录 COT 触发器
                "answer_trigger": answer_trigger,
                # 更新 wandb 配置，记录答案触发器
                "raw_dataset": str(raw_dataset),
                # 更新 wandb 配置，记录原始数据集
                "tokenized_dataset": str(tokenized_dataset),
                # 更新 wandb 配置，记录处理后的数据集
                # "train_input_ids_max_length": max(tokenized_dataset['train']['input_ids_max_length']),
                # "test_input_ids_max_length": max(tokenized_dataset['test']['input_ids_max_length']),
            })
            # 记录训练和测试数据集的最大输入长度（注释掉的代码）

    def collate_fn(batch, args, tokenizer):
        # 定义函数 collate_fn，接受批量数据、参数和 tokenizer
        max_input_length = max([len(item['input_ids']) for item in batch])
        # 获取批量中输入 ID 的最大长度
        max_target_length = max([len(item['labels']) for item in batch])
        # 获取批量中标签的最大长度
        max_prefix_length = max([len(item['prefix']) for item in batch])
        # 获取批量中前缀的最大长度

        input_ids, input_ids_left_padded = [], []
        # 初始化输入 ID 列表和左侧填充的输入 ID 列表
        attention_mask, attention_mask_left_padded = [], []
        # 初始化注意力掩码列表和左侧填充的注意力掩码列表
        labels, labels_left_padded = [], []
        # 初始化标签列表和左侧填充的标签列表
        prefix, prefix_left_padded = [], []
        # 初始化前缀列表和左侧填充的前缀列表
        prefix_attention_mask, prefix_attention_mask_left_padded = [], []
        # 初始化前缀注意力掩码列表和左侧填充的前缀注意力掩码列表

        for item in batch:
            # 遍历批量中的每个项
            # input_ids.append(item['input_ids'] + [tokenizer.pad_token_id] * (max_input_length - len(item['input_ids'])))
            # attention_mask.append(item['attention_mask'] + [0] * (max_input_length - len(item['attention_mask'])))
            # labels.append(item['labels'] + [-100] * (max_target_length - len(item['labels'])))
            # 这些行被注释掉，原本是将输入 ID、注意力掩码和标签填充到最大长度

            labels_left_padded.append([-100] * (max_target_length - len(item['labels'])) + item['labels'])
            # 将标签左侧填充，填充部分为 -100
            prefix_left_padded.append([tokenizer.pad_token_id] * (max_prefix_length - len(item['prefix'])) + item['prefix'])
            # 将前缀左侧填充，填充部分为 tokenizer 的填充 ID
            prefix_attention_mask_left_padded.append(
                [0] * (max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask'])
            # 将前缀注意力掩码左侧填充，填充部分为 0

        ppo_forward_kwargs = {
            'query': [item['prefix_text'] for item in batch],
            # 从批量中提取前缀文本
            'query_tensors': torch.LongTensor(prefix_left_padded),
            # 将左侧填充的前缀转换为 LongTensor
            'query_tensors_attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
            # 将左侧填充的前缀注意力掩码转换为 BoolTensor
            # 'answer_values': torch.FloatTensor([float(item['answer_value'].replace(',', '')) for item in batch]),
            'answer_values': [item['answer_value'].replace(',', '') for item in batch],
            # 从批量中提取答案值，并去除逗号
            'item_ids': torch.LongTensor([int(item['item_id'].split('_')[1]) for item in batch]),
            # 从批量中提取项 ID，并转换为 LongTensor
            # 'answer_cot': [item['answer_cot'] for item in batch],
            # 'sft_model_input_ids': torch.LongTensor(input_ids),
            # 'sft_model_attention_mask': torch.BoolTensor(attention_mask),
            # 'sft_model_labels': torch.LongTensor(labels),
        }

        generate_prefix_kwargs = {
            'input_ids': torch.LongTensor(prefix_left_padded),
            # 将左侧填充的前缀转换为 LongTensor
            'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
            # 将左侧填充的前缀注意力掩码转换为 BoolTensor
            'labels': torch.LongTensor(labels_left_padded)
            # 将左侧填充的标签转换为 LongTensor
        }

        return {
            'ppo_forward_kwargs': ppo_forward_kwargs,
            # 返回 PPO 前向传播的参数
            'generate_prefix_kwargs': generate_prefix_kwargs,
            # 返回生成前缀的参数
        }

    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args['batch_size'],
                                num_workers=args['num_workers'], pin_memory=True,
                                collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
    # 创建训练数据加载器，使用 collate_fn 处理批量数据

    test_dataloader = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=args['eval_batch_size'],
                                num_workers=args['num_workers'], pin_memory=True,
                                collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
    # 创建测试数据加载器，使用 collate_fn 处理批量数据

    return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['test'], test_dataloader)
    # 返回训练数据集和训练数据加载器，以及测试数据集和测试数据加载器

def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    # 定义函数 do_checkpoint，接受参数 args、模型、tokenizer、保存路径和最近的检查点路径列表（可选）
    os.makedirs(save_path, exist_ok=True)
    # 创建保存路径，如果已存在则不报错

    unwrapped_model = accelerator.unwrap_model(model)
    # 解包模型，以便进行保存

    unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process,
                                    save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    # 保存模型的预训练状态到指定路径，只有主进程会执行保存操作

    tokenizer.save_pretrained(save_path)
    # 保存 tokenizer 的预训练状态到指定路径

    if accelerator.is_main_process and most_recent_ckpts_paths is not None:
        # 如果是主进程并且最近的检查点路径列表不为 None
        most_recent_ckpts_paths.append(save_path)
        # 将当前保存路径添加到最近的检查点路径列表中
        if args['keep_num_ckpt'] is not None and len(most_recent_ckpts_paths) > args['keep_num_ckpt']:
            # 如果设置了保留的检查点数量并且最近的检查点数量超过了该值
            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
            # 移除最旧的检查点路径
            # os.remove(ckpt_to_be_removed)
            # os.remove(ckpt_to_be_removed)  # 注释掉的代码，用于删除检查点文件
            shutil.rmtree(ckpt_to_be_removed)
            # 删除检查点目录

def rollout(args, model, ref_model, tokenizer, query_tensors, query_tensors_attention_mask, answer_values, src_name):
    # 定义函数 rollout，接受参数 args、模型、参考模型、tokenizer、查询张量、查询注意力掩码、答案值和源名称
    model.eval()
    # 将模型设置为评估模式
    with torch.no_grad():
        # 在不计算梯度的上下文中执行以下代码
        gen_output = accelerator.unwrap_model(model).generate(
            input_ids=query_tensors,
            attention_mask=query_tensors_attention_mask,
            top_k=0.0, top_p=1.0,
            do_sample=True,
            # output_scores=True,
            # return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
            # bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args['max_gen_length'],
        )
        # 使用模型生成输出

        # completed_tensors, logits_per_steps = gen_output[0], gen_output[1]
        completed_tensors = gen_output
        # 将生成的输出张量赋值给 completed_tensors
        completed_tensors = pad_across_processes(completed_tensors, dim=1, pad_index=tokenizer.pad_token_id, pad_first=False)
        # 在进程之间填充生成的张量

    # Evaluate score
    completed_texts = tokenizer.batch_decode(completed_tensors.cpu().numpy().tolist(), skip_special_tokens=True)
    # 解码生成的张量，转换为文本，跳过特殊标记
    programs = [text.strip().split(cot_trigger)[-1].strip() for text in completed_texts]
    # 提取程序部分，去除前后空格
    execute_fn = post_process_answer_cot_fn_mapper[(args['engine'], src_name)]
    # 根据引擎和源名称获取后处理函数
    correctness = []
    for i, extracted_ans in enumerate(execute_fn(programs)):
        # 遍历提取的答案，执行后处理函数
        target_value = post_process_final_answer_fn_mapper[src_name](answer_values[i])
        # 获取目标值
        if extracted_ans is not None:
            if args['engine'] == 'game24' or args['engine'] == 'calcn':
                is_correct = extracted_ans
                # 如果引擎类型为 'game24' 或 'calcn'，则直接将提取的答案视为正确
            else:
                if compare_answer_fn_mapper[src_name](extracted_ans, target_value):
                    is_correct = 1
                    # 如果提取的答案与目标值匹配，则视为正确
                else:
                    is_correct = 0.1
                    # 否则，给予部分奖励
                    # for mathqa, even though it can executed, if the results is not within a,b,c,d,xxx, still zero reward
                    # because we want to give some reward for the prediction that able to select one of the answer
                    # for example, the executed answer is "{}" in mathqa.
                    # THIS PART IS TO BE DECIDED.
                    # if src_name == 'mathqa' and not (len(extracted_ans) == 1 and extracted_ans.isalpha()):
                    #     is_correct = 0
        else:
            is_correct = 0
            # 如果提取的答案为 None，则视为不正确
        correctness.append(is_correct)
        # 将正确性记录到列表中

    model_input_ids = completed_tensors
    # 将生成的张量赋值给模型输入 ID
    model_attention_mask = (completed_tensors != tokenizer.pad_token_id)
    # 创建模型注意力掩码，标记非填充部分
    with torch.no_grad():
        # 在不计算梯度的上下文中执行以下代码
        # Get old logprob and val
        lm_logits, _dummy2, val = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
        # 获取模型的输出 logits 和价值
        old_logprob = logprobs_from_logits(lm_logits[:, :-1, :], labels=model_input_ids[:, 1:])  # (bs, seqlen-1)
        # 计算旧的对数概率

        # Get the ref model logprob
        ref_logprob = None
        if ref_model is not None:
            ref_lm_logits, _dummy2, _dummy3 = ref_model(input_ids=model_input_ids, attention_mask=model_attention_mask)
            # 获取参考模型的输出 logits
            ref_logprob = logprobs_from_logits(ref_lm_logits[:, :-1, :], labels=model_input_ids[:, 1:])  # (bs, seqlen-1)
            # 计算参考模型的对数概率

    # Masking the last prompt token up untils the token before eos_token_id
    prompt_len = query_tensors.size(1)
    # 获取查询张量的长度
    mask = torch.zeros_like(model_input_ids, dtype=torch.bool)  # (bs, seqlen)
    # 创建与模型输入 ID 相同形状的全零掩码
    mask[:, query_tensors.size(1) - 1: -1] = 1
    # 将掩码的最后一个提示标记之前的部分设置为 1
    score_rew = np.zeros(mask.shape)  # (bs, seqlen)
    # 初始化奖励分数为零
    score_rew[:, -2] = np.array(correctness)
    # 将倒数第二个位置的奖励分数设置为正确性

    nonzero = (model_input_ids == tokenizer.eos_token_id).nonzero()
    # 获取模型输入 ID 中所有 eos_token_id 的位置
    for (bidx, tidx) in nonzero:
        mask[bidx][tidx:] = 0
        # 将掩码中 eos_token_id 之后的部分设置为 0
        score_rew[bidx][tidx:] = 0
        # 将奖励分数中 eos_token_id 之后的部分设置为 0
        score_rew[bidx][tidx - 1] = correctness[bidx]
        # 将倒数第二个位置的奖励分数设置为当前的正确性

    # Make the kl reward and the full reward
    kl_rew = None
    rew = score_rew
    if ref_logprob is not None:
        kl = old_logprob - ref_logprob  # (bs, seqlen-1)
        # 计算 KL 奖励
        kl = (kl.float() * mask[:, :-1]).cpu().numpy()
        # 将 KL 奖励转换为 NumPy 数组
        kl_rew = np.zeros(mask.shape)  # (bs, seqlen)
        kl_rew[:, :-1] = -kl  # NOTE the minus sign
        # 将 KL 奖励的负值填充到 KL 奖励数组中

        kl_coef = args["kl_coef"]
        rew = score_rew + kl_coef * kl_rew
        # 计算最终奖励

    # Process val ret adv logprob
    val = (val.float() * mask).cpu().numpy()
    # 将价值乘以掩码并转换为 NumPy 数组
    gamma = args["gamma"]
    lam = args["lam"]
    # 获取折扣因子和 Lambda 值
    # ret = np.zeros_like(rew)
    adv = np.zeros_like(rew)
    # 初始化优势数组
    for i in range(len(rew)):
        cur_rew, cur_val = rew[i], val[i]
        # 获取当前的奖励和价值
        cur_delta = -cur_val[:-1] + cur_rew[:-1] + gamma * cur_val[1:]
        # 计算当前的增量
        cur_adv = discount_cumsum(cur_delta, discount=gamma * lam)
        # 计算当前的优势
        cur_adv[:prompt_len - 1] = 0
        # 将提示长度之前的优势设置为 0
        adv[i][:-1] = cur_adv
        # 将当前的优势赋值到优势数组中

    # lambda_return = GAE + values
    ret = adv + val  # (bs, seqlen)
    # 计算返回值

    rew = torch.tensor(rew, device=mask.device, dtype=old_logprob.dtype) * mask
    # 将奖励转换为张量并乘以掩码
    score_rew = torch.tensor(score_rew, device=mask.device, dtype=old_logprob.dtype) * mask
    # 将奖励分数转换为张量并乘以掩码
    if kl_rew is not None:
        kl_rew = torch.tensor(kl_rew, device=mask.device, dtype=old_logprob.dtype) * mask
    # 如果 KL 奖励不为 None，则将其转换为张量并乘以掩码
    ret = torch.tensor(ret, device=mask.device, dtype=old_logprob.dtype) * mask
    # 将返回值转换为张量并乘以掩码
    val = torch.tensor(val, device=mask.device, dtype=old_logprob.dtype) * mask
    # 将价值转换为张量并乘以掩码
    adv = torch.tensor(adv, device=mask.device, dtype=old_logprob.dtype) * mask
    # 将优势转换为张量并乘以掩码
    old_logprob = old_logprob * mask[:, :-1]
    # 将旧的对数概率乘以掩码

    ## Debug
    # accelerator.print("padded_prompt_len:", prompt_len)
    # accelerator.print("model_input_ids:", tokenizer.batch_decode(model_input_ids[:1].cpu().numpy().tolist()))
    # accelerator.print("model_attention_mask:", model_attention_mask[:1].cpu().float().numpy().tolist())
    # accelerator.print("mask:", mask[:1].cpu().float().numpy().tolist())
    # accelerator.print("rew:", rew[:1].cpu().float().numpy().tolist())
    # accelerator.print("ret:", ret[:1].cpu().float().numpy().tolist())
    # accelerator.print("val:", val[:1].cpu().float().numpy().tolist())
    # accelerator.print("adv:", adv[:1].cpu().float().numpy().tolist())
    # accelerator.print("old_logprob:", old_logprob[:1].cpu().float().numpy().tolist())

    model.train()
    # 将模型设置为训练模式
    return model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv
    # 返回模型输入 ID、模型注意力掩码、掩码、奖励、分数奖励、KL 奖励、返回值、正确性、价值、旧的对数概率、参考对数概率和优势

def train_one_epoch(args, model, ref_model, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    global_step, global_iter_num, test_dataset, test_dataloader,
                    prefix, epoch, best_eval_log_dict, summary_log_dict, most_recent_ckpts_paths):
    # 定义函数 train_one_epoch，接受多个参数，包括训练数据集、数据加载器、优化器、调度器等
    model_dir = args['model_dir']
    # 获取模型保存目录
    clip_grad_norm = args.get('clip_grad_norm', None)
    # 获取梯度裁剪的阈值（如果有的话）
    vf_coef = args['vf_coef']
    # 获取价值函数的系数
    evaluating_step_freq = args.get('evaluating_step_freq', None)
    # 获取评估步骤的频率（如果有的话）
    logging_step_freq = args.get('logging_step_freq', None)
    # 获取日志记录的频率（如果有的话）
    saving_step_freq = args.get('saving_step_freq', None)
    # 获取保存模型的频率（如果有的话）
    model.train()
    # 将模型设置为训练模式
    epoch_result_dict = defaultdict(list)
    # 初始化一个字典，用于存储每个 epoch 的结果

    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_main_process, desc='Train Loop') as t:
        # 使用 tqdm 显示训练进度条，遍历训练数据加载器
        for idx, batch in t:
            # 遍历每个批次
            result_dict = defaultdict(list)
            # 初始化一个字典，用于存储当前批次的结果

            # Do rollout first
            model.eval()
            # 将模型设置为评估模式
            model_input_ids, model_attention_mask, mask, rew, score_rew, kl_rew, ret, correctness, val, old_logprob, ref_logprob, adv = rollout(
                args, model, ref_model, tokenizer,
                query_tensors=batch['ppo_forward_kwargs']['query_tensors'],
                query_tensors_attention_mask=batch['ppo_forward_kwargs']['query_tensors_attention_mask'],
                answer_values=batch['ppo_forward_kwargs']['answer_values'],
                src_name=train_dataset[0]['item_id'].split('_')[0],
            )
            # 调用 rollout 函数，执行前向传播，获取模型输入 ID、注意力掩码、掩码、奖励等信息
            model.train()
            # 将模型重新设置为训练模式
            
            # preprocess
            raw_adv = adv
            # 保存原始优势
            if args['adv_whitening'] == 'global':
                adv = allgather_masked_whiten(adv, mask) # (mini_bs, seqlen)
                # 如果设置为全局优势白化，则进行全局白化处理
            elif args['adv_whitening'] == 'local':
                adv = masked_whiten(adv, mask)
                # 如果设置为局部优势白化，则进行局部白化处理

            batch_size_per_gpu = len(batch['ppo_forward_kwargs']['query'])
            # 获取每个 GPU 的批次大小
            mini_batch_size_per_gpu = args["mini_batch_size"]
            # 获取每个 GPU 的小批次大小
            ppo_epochs = args["ppo_epochs"]
            # 获取 PPO 的训练轮数
            train_stats = {}
            # 初始化训练统计信息
            
            for _ in range(ppo_epochs):
                # 循环进行 PPO 训练轮数
                perms = torch.randperm(batch_size_per_gpu)
                # 随机打乱每个 GPU 的批次大小
                for mini_idx in range(0, len(perms), mini_batch_size_per_gpu):
                    # 遍历每个小批次
                    b_inds = perms[mini_idx: mini_idx + mini_batch_size_per_gpu]
                    # 选择当前小批次的索引
                    # Subset to batch
                    cur_val = val[b_inds].contiguous()  # mini_bs x seqlen
                    # 获取当前小批次的价值
                    cur_old_logprob = old_logprob[b_inds].contiguous()  # mini_bs x seqlen
                    # 获取当前小批次的旧对数概率
                    cur_mask = mask[b_inds].contiguous()  # mini_bs x seqlen
                    # 获取当前小批次的掩码
                    cur_rew = rew[b_inds].contiguous()  # mini_bs x seqlen
                    # 获取当前小批次的奖励
                    cur_score_rew = score_rew[b_inds].contiguous()  # mini_bs x seqlen
                    # 获取当前小批次的分数奖励
                    cur_kl_rew = None if kl_rew is None else kl_rew[b_inds].contiguous()  # mini_bs x seqlen
                    # 获取当前小批次的 KL 奖励（如果存在）
                    cur_ret = ret[b_inds].contiguous()  # mini_bs x seqlen
                    # 获取当前小批次的返回值
                    cur_adv = adv[b_inds].contiguous()  # mini_bs x seqlen
                    # 获取当前小批次的优势
                    cur_raw_adv = raw_adv[b_inds].contiguous()  # mini_bs x seqlen
                    # 获取当前小批次的原始优势
                    cur_model_input_ids = model_input_ids[b_inds].contiguous()  # mini_bs x seqlen
                    # 获取当前小批次的模型输入 ID
                    cur_model_attention_mask = model_attention_mask[b_inds].contiguous()  # mini_bs x seqlen
                    # 获取当前小批次的模型注意力掩码
                    
                    resp_len_per_sample = torch.clamp(torch.sum(cur_mask, dim=1), min=1.0)  # (mini_bs,)
                    # 计算每个样本的响应长度，确保最小值为 1
                    cur_query_mask = torch.logical_xor(cur_mask, cur_model_attention_mask)  # (mini_bs, seqlen)
                    # 计算当前查询的掩码
                    query_len_per_sample = torch.clamp(torch.sum(cur_query_mask, dim=1), min=1.0)  # (mini_bs,)
                    # 计算每个样本的查询长度，确保最小值为 1

                    # Preprocess advantage and get metrics  
                    cur_mask = cur_mask.type(cur_adv.dtype).contiguous()
                    # 将掩码转换为与优势相同的数据类型
                    mean_adv, var_adv = masked_mean(cur_adv, cur_mask), masked_var(cur_adv, cur_mask)
                    # 计算优势的均值和方差

                    # Forward current model
                    model.eval()
                    # 将模型设置为评估模式
                    lm_logits, _, vpreds = model(input_ids=cur_model_input_ids, attention_mask=cur_model_attention_mask)
                    # 获取模型的输出 logits 和价值预测
                    logprob = logprobs_from_logits(lm_logits[:, :-1, :], cur_model_input_ids[:, 1:])  # (mini_bs, seqlen-1)
                    # 计算当前的对数概率

                    # Compute losses
                    loss = 0
                    # 初始化损失为 0

                    # policy gradient loss
                    ratio = torch.exp(logprob - cur_old_logprob)
                    # 计算比率
                    pg_losses = -cur_adv[:, :-1] * ratio
                    # 计算策略梯度损失
                    pg_losses2 = -cur_adv[:, :-1] * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2)
                    # 计算第二种策略梯度损失
                    pg_loss = ((torch.max(pg_losses, pg_losses2) * cur_mask[:, :-1]).sum(dim=-1) / resp_len_per_sample).mean()
                    # 计算最终的策略梯度损失

                    # value loss
                    vpredclipped = torch.max(torch.min(vpreds, cur_val + 0.2), cur_val - 0.2)
                    # 对价值预测进行裁剪
                    vf_losses1 = (vpreds - cur_ret) ** 2
                    # 计算价值损失1
                    vf_losses2 = (vpredclipped - cur_ret) ** 2
                    # 计算价值损失2
                    vf_loss = 0.5 * ((torch.max(vf_losses1, vf_losses2) * cur_mask).sum(dim=-1) / resp_len_per_sample).mean()
                    # 计算最终的价值损失

                    # total loss
                    loss += pg_loss + vf_coef * vf_loss
                    # 计算总损失

                    # model_output = model(input_ids=model_input_ids, attention_mask=model_attention_mask)
                    # logits = model_output[0]
                    # logprob_dist = torch.nn.functional.log_softmax(logits,dim=-1)
                    # logprob = torch.gather(logprob_dist, 2, model_input_ids[:, 1:].unsqueeze(2)).squeeze(-1)
                    # loss_pg = (-logprob * cur_ret[:,:-1]).sum() / torch.maximum(torch.sum(mask[:,:-1]), torch.tensor(1.0))
                    # loss += loss_pg

                    # sft_model_input_ids = batch['ppo_forward_kwargs']['sft_model_input_ids']
                    # sft_model_attention_mask = batch['ppo_forward_kwargs']['sft_model_attention_mask']
                    # sft_model_labels = batch['ppo_forward_kwargs']['sft_model_labels']
                    # loss_sft = model(input_ids=sft_model_input_ids, attention_mask=sft_model_attention_mask, labels=sft_model_labels)[0]
                    # loss += loss_sft

                    # token related metrics
                    mean_query_len = torch.mean(allgather(torch.mean(query_len_per_sample)))
                    # 计算查询长度的均值
                    std_query_len = torch.mean(allgather(torch.std(query_len_per_sample)))
                    # 计算查询长度的标准差
                    mean_resp_len = torch.mean(allgather(torch.mean(resp_len_per_sample)))
                    # 计算响应长度的均值
                    std_resp_len = torch.mean(allgather(torch.std(resp_len_per_sample)))
                    # 计算响应长度的标准差

                    # value related metrics
                    # vf_expl_var_num = torch.var(torch.masked_select(cur_ret - vpreds, cur_mask.bool())) 
                    # vf_expl_var_dem = torch.var(torch.masked_select(cur_ret, cur_mask.bool()))
                    vf_expl_var_num = masked_var(cur_ret - vpreds, cur_mask)
                    # 计算价值解释方差的分子
                    vf_expl_var_dem = masked_var(cur_ret, cur_mask)
                    # 计算价值解释方差的分母
                    vf_expl_var = 1.0 - vf_expl_var_num / (vf_expl_var_dem + 1e-8)
                    # 计算价值解释方差
                    vf_expl_var = max(-1.0, vf_expl_var.item())  # the truncated value suffices
                    # 确保价值解释方差的值不小于 -1
                    mean_vpred = masked_mean(vpreds, cur_mask)
                    # 计算价值预测的均值
                    mean_return = masked_mean(cur_ret, cur_mask)
                    # 计算返回值的均值
                    mean_reward = masked_mean(cur_rew, cur_mask)
                    # 计算奖励的均值
                    mean_score_reward = masked_mean(cur_score_rew, cur_mask)
                    # 计算分数奖励的均值
                    mean_kl_reward = 0.0 if cur_kl_rew is None else masked_mean(cur_kl_rew, cur_mask)
                    # 计算 KL 奖励的均值
                    mean_kcxkl_reward = args["kl_coef"] * mean_kl_reward
                    # 计算 KL 奖励的加权值

                    # policy related metrics
                    mean_ratio = masked_mean(ratio, cur_mask[:, :-1])
                    # 计算比率的均值
                    #mean_adv = masked_mean(cur_adv[:, :-1], cur_mask[:, :-1])
                    mean_logprob = masked_mean(logprob, cur_mask[:, :-1])
                    # 计算对数概率的均值
                    # sequence-level kl
                    mean_seq_kl = -1.0
                    if cur_kl_rew is not None:
                        cur_kl = -cur_kl_rew
                        seq_kl = torch.sum(cur_kl * cur_mask, dim=1)  # (mini_bs,)
                        mean_seq_kl = torch.mean(seq_kl)
                        # 计算序列级别的 KL 奖励

                    # Update
                    epoch_result_dict['loss'].append(loss.item())
                    # 将当前损失添加到 epoch 结果字典中

                    # accelerator.backward(loss)
                    # accelerator.deepspeed_engine_wrapped.backward(loss)
                    # runs backpropagation and handles mixed precision
                    # 执行反向传播并处理混合精度
                    if accelerator.distributed_type == "DEEPSPEED":
                        # 如果使用 DeepSpeed 分布式训练
                        accelerator.deepspeed_engine_wrapped.engine.backward(loss)
                        # 使用 DeepSpeed 的引擎执行反向传播
                        total_grad_norm = 0.0
                        # 初始化总梯度范数为 0
                        for n, p in model.named_parameters():
                            # 遍历模型的每个参数
                            cur_grad = deepspeed.utils.safe_get_full_grad(p).view(-1)
                            # 获取参数的完整梯度并展平
                            cur_grad_norm_sqrt = torch.norm(cur_grad, 2)
                            # 计算当前梯度的 L2 范数
                            if cur_grad_norm_sqrt < 1e-8:
                                accelerator.print(f'{n} grad_norm_sqrt: {cur_grad_norm_sqrt}')
                                # 如果当前梯度范数非常小，则打印参数名称和梯度范数
                            total_grad_norm += cur_grad_norm_sqrt ** 2
                            # 累加当前梯度的平方到总梯度范数
                        total_grad_norm = total_grad_norm ** 0.5
                        # 计算总梯度范数的平方根
                        # Deepspeed's `engine.step` performs the following operations:
                        # - gradient accumulation check
                        # - gradient clipping
                        # - optimizer step
                        # - zero grad
                        # - checking overflow
                        # - lr_scheduler step (only if engine.lr_scheduler is not None)
                        # DeepSpeed 的 `engine.step` 执行以下操作：
                        # - 梯度累积检查
                        # - 梯度裁剪
                        # - 优化器步骤
                        # - 清零梯度
                        # - 检查溢出
                        # - 学习率调度步骤（仅当 engine.lr_scheduler 不为 None 时）
                        accelerator.deepspeed_engine_wrapped.engine.step()
                    else:
                        # 如果不使用 DeepSpeed
                        accelerator.backward(loss)
                        # 执行反向传播
                        total_grad_norm = -1.0
                        # 初始化总梯度范数为 -1
                        if clip_grad_norm is not None:
                            # 如果设置了梯度裁剪阈值
                            total_grad_norm = accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                            # 对模型参数进行梯度裁剪
                    optimizer.step()
                    # 执行优化器步骤
                    model.zero_grad()
                    # 清零模型的梯度
                    optimizer.zero_grad()
                    # 清零优化器的梯度

                    # Update running stats
                    n_correct, total = do_gather([sum(correctness), len(correctness)])
                    # 收集当前批次的正确预测数量和总数量
                    train_stats["acc"] = n_correct / total
                    # 计算并记录训练准确率
                    train_stats["ncor"] = n_correct
                    # 记录当前批次的正确预测数量
                    train_stats["total"] = total
                    # 记录当前批次的总数量
                    train_stats['pg_loss'] = pg_loss.item()
                    # 记录策略梯度损失
                    train_stats['vf_loss'] = vf_loss.item()
                    # 记录价值损失
                    train_stats['vf_expl_var'] = vf_expl_var
                    # 记录价值解释方差

                    for k, v in train_stats.items():
                        result_dict[k].append(v)
                        # 将训练统计信息添加到结果字典中

                    total_param_norm = 0.0
                    # 初始化总参数范数为 0
                    if accelerator.distributed_type == "DEEPSPEED":
                        # 如果使用 DeepSpeed 分布式训练
                        for n, p in model.named_parameters():
                            # 遍历模型的每个参数
                            cur_param = deepspeed.utils.safe_get_full_fp32_param(p).view(-1)
                            # 获取参数的全精度副本并展平
                            total_param_norm += torch.norm(cur_param, 2) ** 2
                            # 累加当前参数的 L2 范数的平方
                        total_param_norm = total_param_norm ** 0.5
                        # 计算总参数范数的平方根
                    else:
                        total_param_norm = torch.norm(
                            torch.cat([p.view(-1) for p in model.parameters()]),
                            p=2  # L2 norm
                        )
                        # 计算模型所有参数的 L2 范数

                    # logging
                    if accelerator.is_main_process and args['wandb_log']:
                        # 如果是主进程并且启用了 wandb 日志
                        wandb.log({
                            "nn/total_grad_norm": total_grad_norm,
                            # 记录总梯度范数
                            "nn/total_param_norm": total_param_norm,
                            # 记录总参数范数
                            "nn/lr": scheduler.get_last_lr()[0],
                            # 记录当前学习率
                        }, step=global_iter_num)
                        wandb.log({
                            "acc/acc": train_stats["acc"],
                            # 记录训练准确率
                            "acc/ncor": train_stats["ncor"],
                            # 记录当前批次的正确预测数量
                            "acc/total": train_stats["total"],
                            # 记录当前批次的总数量
                        }, step=global_iter_num)
                        wandb.log({
                            "loss/loss:": loss,
                            # 记录当前损失
                            "loss/pg_loss": pg_loss,
                            # 记录策略梯度损失
                            "loss/vf_loss": vf_loss,
                            # 记录价值损失
                        }, step=global_iter_num)
                        wandb.log({
                            "tokens/mean_query_len": mean_query_len,
                            # 记录查询长度的均值
                            "tokens/std_query_len": std_query_len,
                            # 记录查询长度的标准差
                            "tokens/mean_resp_len": mean_resp_len,
                            # 记录响应长度的均值
                            "tokens/std_resp_len": std_resp_len,
                            # 记录响应长度的标准差
                        }, step=global_iter_num)
                        wandb.log({
                            "policy/mean_ratio": mean_ratio,
                            # 记录比率的均值
                            "policy/mean_adv": mean_adv,
                            # 记录优势的均值
                            "policy/var_adv": var_adv,
                            # 记录优势的方差
                            "policy/mean_logprob": mean_logprob,
                            # 记录对数概率的均值
                            "policy/mean_seq_kl": mean_seq_kl,
                            # 记录序列级别的 KL 奖励
                        }, step=global_iter_num)
                        wandb.log({
                            "value/vf_expl_var": vf_expl_var,
                            # 记录价值解释方差
                            "value/mean_vpred": mean_vpred,
                            # 记录价值预测的均值
                            "value/mean_return": mean_return,
                            # 记录返回值的均值
                            "value/mean_reward": mean_reward,
                            # 记录奖励的均值
                            "value/mean_score_reward": mean_score_reward,
                            # 记录分数奖励的均值
                            "value/mean_kl_reward": mean_kl_reward,
                            # 记录 KL 奖励的均值
                            "value/mean_kcxkl_reward": mean_kcxkl_reward,
                            # 记录 KL 奖励的加权值
                        }, step=global_iter_num)

                    # Update iter num
                    # torch.distributed.barrier()
                    global_iter_num += 1
                    # 更新全局迭代次数

            scheduler.step()
            # 更新学习率调度器的状态
            global_step += 1
            # 全局步骤数加 1
            # accelerator.empty_cache()
            # 清空缓存（注释掉的代码，可能用于释放内存）
            # Step update metric
            epoch_result_dict['loss'].append(loss.item())
            # 将当前损失添加到 epoch 结果字典中
            for k, v in train_stats.items():
                epoch_result_dict[k].append(v)
                # 将训练统计信息添加到 epoch 结果字典中

            # Step evaluating
            eval_log_dict = {}
            # 初始化评估日志字典
            is_best = False
            # 初始化最佳标志为 False
            if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
                # 如果设置了评估步频且当前全局步骤是评估步频的倍数
                evaluate_result_dict = {f'Eval.Gen.{k}': v for k, v in
                                        evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer).items()}
                # 调用评估生成函数并将结果存入评估结果字典
                eval_log_dict.update(evaluate_result_dict)
                # 更新评估日志字典
                if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', -1):
                    # 如果当前评估的值准确率高于最佳值
                    is_best = True
                    # 设置最佳标志为 True
                    best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                    # 更新最佳评估值准确率
                    if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                        summary_log_dict['Eval.Gen.value_accuracy'] = []
                    # 如果摘要日志字典中没有该键，则初始化为空列表
                    summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])
                    # 将当前评估值准确率添加到摘要日志中

            # Step logging
            train_log_dict = {}
            # 初始化训练日志字典
            if logging_step_freq is not None and global_step % logging_step_freq == 0:
                # 如果设置了日志步频且当前全局步骤是日志步频的倍数
                train_log_dict = {f'T.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in epoch_result_dict.items()}
                # 计算每个指标的平均值并存入训练日志字典

            if eval_log_dict or train_log_dict:
                # 如果评估日志字典或训练日志字典非空
                log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                # 创建日志字典，包含学习率、训练日志、评估日志和最佳评估日志
                if accelerator.is_main_process and args['wandb_log']:
                    # 如果是主进程并且启用了 wandb 日志
                    wandb.log(log_dict, step=global_step)
                    # 记录日志到 wandb
                    log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'], **log_dict}
                    # 添加 wandb 项目和运行名称到日志字典
                log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k,v in log_dict.items()}
                # 格式化日志字典中的浮点数为 5 位有效数字
                accelerator.print(f"{prefix}[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")
                # 打印当前的日志信息

            # Step saving
            if saving_step_freq is not None and global_step % saving_step_freq == 0:
                # 如果设置了保存步频且当前全局步骤是保存步频的倍数
                if is_best:
                    # 如果当前模型是最佳模型
                    save_path = os.path.join(model_dir, f'best')
                    # 设置最佳模型保存路径
                    do_checkpoint(args, model, tokenizer, save_path)
                    # 执行模型检查点保存
                if args['keep_num_ckpt'] > 0:
                    # 如果设置了保留的检查点数量大于 0
                    save_path = os.path.join(model_dir, f'global_step_{str(global_step)}')
                    # 设置当前全局步骤的保存路径
                    do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)
                    # 执行模型检查点保存

            # Keep only max_record items
            for k, v in epoch_result_dict.items():
                if len(v) > 1:
                    epoch_result_dict[k] = v[-1:]
                    # 如果记录的项数量大于 1，则只保留最后一项

            # Metric summary:
            epoch_result_dict = {k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()}
            # 计算每个指标的平均值并更新 epoch 结果字典
            return epoch_result_dict, global_step, global_iter_num
            # 返回 epoch 结果字典、全局步骤和全局迭代次数

def evaluate_generation(args, model, dataset, dataloader, tokenizer):
    # 定义评估生成函数，接受参数、模型、数据集、数据加载器和分词器
    model.eval()
    # 将模型设置为评估模式
    predictions = []
    # 初始化预测结果列表
    targets = []
    # 初始化目标值列表
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process,
                           desc='Evaluation Gen Loop'):
        # 遍历数据加载器中的每个批次，使用 tqdm 显示进度条
        output_ = accelerator.unwrap_model(model).generate(
            # 使用加速器解包模型并生成输出
            **batch['generate_prefix_kwargs'],
            # 从批次中提取生成前缀的关键字参数
            max_length=args['max_gen_length'],
            # 设置生成的最大长度
            output_scores=True,
            # 输出分数
            return_dict_in_generate=True,
            # 返回生成的字典
            num_beams=1,
            # 设置束搜索的束宽
            use_cache=True,
            # 使用缓存
            do_sample=False,
            # 不进行采样
            pad_token_id=tokenizer.pad_token_id,
            # 设置填充标记的 ID
            eos_token_id=tokenizer.eos_token_id,
            # 设置结束标记的 ID
        )
        generated_ids = output_.sequences
        # 获取生成的 ID 序列
        generated_ids = pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        # 在各进程之间填充生成的 ID 序列

        labels = batch['generate_prefix_kwargs']['labels']
        # 从批次中提取标签
        labels = pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        # 在各进程之间填充标签
        labels[labels == -100] = tokenizer.pad_token_id
        # 将标签中值为 -100 的部分替换为填充标记 ID

        generated_ids, labels = accelerator.gather(generated_ids), accelerator.gather(labels)
        # 收集生成的 ID 和标签

        preds = [tokenizer.decode(g.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in
                 generated_ids]
        # 解码生成的 ID 为预测字符串
        predictions.extend(preds)
        # 将预测结果添加到预测列表中
        target = [tokenizer.decode(t.cpu().numpy().tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for t in
                  labels]
        # 解码标签为目标字符串
        targets.extend(target)
        # 将目标结果添加到目标列表中

    predictions = predictions[:len(dataset)]
    # 确保预测结果的长度与数据集相同
    targets = targets[:len(dataset)]
    # 确保目标结果的长度与数据集相同

    if accelerator.is_main_process and accelerator.is_local_main_process:
        # 如果是主进程并且是本地主进程
        results = []
        # 初始化结果列表
        src_name = dataset[0]['item_id'].split('_')[0]
        # 从数据集中获取源名称
        for pred, tar, item in zip(predictions, targets, dataset):
            # 遍历预测、目标和数据集中的每个项
            cur_res = {
                'item_id': item['item_id'],
                # 当前项的 ID
                'answer_value': item['answer_value'],
                # 当前项的答案值
            }
            ## Processing target
            target_cot = tar.strip().split(cot_trigger)[-1].strip()
            # 处理目标，提取目标的 COT 部分
            target_value = post_process_final_answer_fn_mapper[src_name](cur_res['answer_value'])
            # 使用源名称映射处理最终答案
            cur_res['target'] = target
            # 将目标添加到当前结果
            cur_res['target_cot'] = target_cot
            # 将目标的 COT 部分添加到当前结果
            cur_res['target_value'] = target_value
            # 将目标值添加到当前结果
            ## Processing prediction
            prediction_cot = pred.strip().split(cot_trigger)[-1].strip()
            # 处理预测，提取预测的 COT 部分
            cur_res['prediction'] = pred
            # 将预测添加到当前结果
            cur_res['prediction_cot'] = prediction_cot
            # 将预测的 COT 部分添加到当前结果
            cur_res['prediction_value'] = None # Tobe filled
            # 初始化预测值为 None，待填充
            results.append(cur_res)
            # 将当前结果添加到结果列表

        execute_fn = post_process_answer_cot_fn_mapper[(args['engine'], src_name)]
        # 获取后处理函数
        corr_value = 0
        # 初始化正确值计数
        for i, prediction_value in enumerate(execute_fn([item['prediction_cot'] for item in results])):
            # 遍历预测值并执行后处理
            target_value = results[i]['target_value']
            # 获取当前目标值
            is_correct = compare_answer_fn_mapper[src_name](prediction_value, target_value) if prediction_value is not None else False
            # 比较预测值和目标值，判断是否正确
            results[i]['prediction_value'] = prediction_value
            # 将预测值添加到结果中
            results[i]['is_correct'] = is_correct
            # 将正确标志添加到结果中
            corr_value += is_correct
            # 累加正确值计数

        res_path = args['model_dir'].rstrip('/')+ '/' + '_res.json'
        # 设置结果保存路径
        with open(res_path, 'w') as f:
            json.dump(results, f, indent=2)
            # 将结果写入 JSON 文件

        # if args['wandb_log']:
        #     table = wandb.Table(dataframe=pd.DataFrame(results))
        #     wandb.log({"predictions": table})

        value_accuracy = corr_value / len(results) * 100
        # 计算值准确率
        accelerator.print(f"[Eval Info] value_accuracy: {value_accuracy:.5g}%")
        # 打印评估信息
        value_accuracy = torch.FloatTensor([value_accuracy]).to(accelerator.device)
        # 将准确率转换为张量并移动到加速器设备
    else:
        value_accuracy = torch.FloatTensor([-1.0]).to(accelerator.device)
        # 如果不是主进程，设置值准确率为 -1

    value_accuracy = broadcast(value_accuracy).cpu().numpy().tolist()[0]
    # 广播值准确率并转换为列表

    # Metric summary:
    model.train()
    # 将模型设置回训练模式
    return {'value_accuracy': value_accuracy}
    # 返回值准确率

def main(args):
    # 定义主函数，接受参数
    set_seed(args['seed'] + accelerator.process_index)
    # 设置随机种子，确保可重复性

    if accelerator.is_main_process and args['wandb_log']:
        # 如果是主进程并且启用了 wandb 日志
        wandb.init(project=args['wandb_project'], name=args['wandb_run_name'])
        # 初始化 wandb 项目和运行名称
        wandb.config.update(args)
        # 更新 wandb 配置

    tokenizer = AutoTokenizer.from_pretrained(args['tokenizer_name_or_path'], use_fast=True)
    # 从预训练模型加载分词器
    tokenizer.pad_token_id = 1
    # 设置填充标记的 ID
    tokenizer.eos_token_id = 2
    # 设置结束标记的 ID

    (train_dataset, train_dataloader), (test_dataset, test_dataloader) = prepare_datasets_and_data_loaders(args, tokenizer)
    # 准备训练和测试数据集及数据加载器

    MODEL_CLASS = AutoModelForCausalLMWithValueHead
    # 定义模型类
    model = MODEL_CLASS.from_pretrained(args['model_name_or_path'])
    # 从预训练模型加载模型
    # accelerator.print(f'[Vocab size]: {len(tokenizer)}')
    # model.resize_token_embeddings(len(tokenizer))

    # initialize ref model (if any)
    ref_model = None
    # 初始化参考模型为 None
    if args['ref_model_name_or_path']:
        # 如果指定了参考模型路径
        ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(args['ref_model_name_or_path'])
        # 从预训练模型加载参考模型
        # from copy import deepcopy
        # ref_model = deepcopy(model)

    # optimizer
    n_epochs = args['n_epochs']
    # 获取训练的轮数
    num_training_steps = (len(train_dataloader) // accelerator.num_processes * n_epochs)
    # 计算训练步骤总数
    warmup_step = args['warmup_step'] if args['warmup_step'] is not None and args['warmup_step'] >= 0 else int(0.1 * num_training_steps)
    # 设置预热步骤数
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": args['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            "weight_decay": 0.0,
        },
    ]
    # 将模型参数分组，设置权重衰减

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=1e-8)
    # 使用 AdamW 优化器
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps)
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step)
    # 设置学习率调度器为常数调度器

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader,
                                                                              test_dataloader)
    # 准备模型、优化器和数据加载器，适应加速器

    if ref_model is not None:
        # 如果存在参考模型
        if accelerator.distributed_type == "DEEPSPEED":
            ref_model = prepare_deepspeed_ref_model(ref_model)
            # 如果使用 DeepSpeed，准备参考模型
        else:
            ref_model = accelerator.prepare(ref_model)
            # 否则，适应加速器

    global_step = 0
    # 初始化全局步骤
    global_iter_num = 0
    # 初始化全局迭代次数
    evaluating_epoch_freq = args['evaluating_epoch_freq']
    # 获取评估的轮频
    logging_epoch_freq = args['logging_epoch_freq']
    # 获取日志的轮频
    saving_epoch_freq = args['saving_epoch_freq']
    # 获取保存的轮频
    model_dir = args['model_dir']
    # 获取模型保存目录
    best_eval_log_dict = {}
    # 初始化最佳评估日志字典
    summary_log_dict = {}
    # 初始化摘要日志字典
    os.makedirs(model_dir, exist_ok=True)
    # 创建模型保存目录
    most_recent_ckpts_paths = []
    # 初始化最近检查点路径列表
    with tqdm(range(1, n_epochs+1), total=n_epochs, disable=False) as t:
        # 使用 tqdm 显示训练进度
        for epoch in t:
            # 遍历每个训练轮
            kwargs = {
                'args': args,
                'model': model,
                'ref_model': ref_model,
                'train_dataset': train_dataset,
                'train_dataloader': train_dataloader,
                'test_dataset': test_dataset,
                'test_dataloader': test_dataloader,
                'optimizer': optimizer,
                'scheduler': scheduler,
                'global_step': global_step,
                'global_iter_num': global_iter_num,
                'tokenizer': tokenizer,
                'prefix': '',
                'epoch': epoch,
                'best_eval_log_dict': best_eval_log_dict,
                'summary_log_dict': summary_log_dict,
                'most_recent_ckpts_paths': most_recent_ckpts_paths,
            }
            # 准备训练所需的参数
            train_epoch_result_dict, global_step, global_iter_num = train_one_epoch(**kwargs)
            # 训练一个轮次并获取结果

            eval_log_dict = {}
            # 初始化评估日志字典
            is_best = False
            # 初始化最佳标志为 False
            if evaluating_epoch_freq is not None and epoch % evaluating_epoch_freq == 0:
                # 如果设置了评估轮频且当前轮是评估轮频的倍数
                evaluate_result_dict = {f'Eval.Gen.{k}': v for k, v in
                                        evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer).items()}
                # 调用评估生成函数并将结果存入评估结果字典
                eval_log_dict.update(evaluate_result_dict)
                # 更新评估日志字典
                if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', -1):
                    # 如果当前评估的值准确率高于最佳值
                    is_best = True
                    # 设置最佳标志为 True
                    best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                    # 更新最佳评估值准确率
                    if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                        summary_log_dict['Eval.Gen.value_accuracy'] = []
                    # 如果摘要日志字典中没有该键，则初始化为空列表
                    summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])
                    # 将当前评估值准确率添加到摘要日志中

            train_log_dict = {}
            # 初始化训练日志字典
            if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
                # 如果设置了日志轮频且当前轮是日志轮频的倍数
                train_log_dict = {f'T.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in
                                train_epoch_result_dict.items()}
                # 计算每个指标的平均值并存入训练日志字典

            if eval_log_dict or train_log_dict:
                # 如果评估日志字典或训练日志字典非空
                log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                # 创建日志字典，包含学习率、训练日志、评估日志和最佳评估日志
                if accelerator.is_main_process and args['wandb_log']:
                    # 如果是主进程并且启用了 wandb 日志
                    wandb.log(log_dict, step=global_iter_num)
                    # 记录日志到 wandb
                    log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'] + '|' + wandb.run.id, **log_dict}
                    # 添加 wandb 项目和运行名称到日志字典

                log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k, v in log_dict.items()}
                # 格式化日志字典中的浮点数为 5 位有效数字
                accelerator.print(
                    f"[Epoch={epoch}/{args['n_epochs']}, Step={global_step}] {log_dict}")
                # 打印当前的日志信息

            if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
                # 如果设置了保存轮频且当前轮是保存轮频的倍数
                if is_best:
                    # 如果当前模型是最佳模型
                    save_path = os.path.join(model_dir, f'best')
                    # 设置最佳模型保存路径
                    do_checkpoint(args, model, tokenizer, save_path)
                    # 执行模型检查点保存
                #
                if args['keep_num_ckpt'] > 0:
                    # 如果设置了保留的检查点数量大于 0
                    save_path = os.path.join(args['model_dir'], f'global_step_{str(global_step)}_epoch_{epoch}')
                    # 设置当前全局步骤和轮次的保存路径
                    do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)
                    # 执行模型检查点保存

    return 
    # 返回 


if __name__ == '__main__':
    # 如果当前模块是主程序
    from transformers import HfArgumentParser
    # 从 transformers 库导入 HfArgumentParser

    NONE_INT = -100
    # 定义常量 NONE_INT，用于表示无效整数
    NONE_STR = 'None'
    # 定义常量 NONE_STR，用于表示无效字符串

    @dataclass
    class Arguments:
        # 定义 Arguments 数据类，用于存储程序参数
        model_name_or_path: str
        # 模型名称或路径
        tokenizer_name_or_path: str
        # 分词器名称或路径
        model_dir: str
        # 模型保存目录
        train_file: str
        # 训练文件路径
        test_file: str
        # 测试文件路径
        batch_size: int = field(default=8)
        # 批次大小，默认值为 8
        mini_batch_size: int = field(default=8)
        # 小批次大小，默认值为 8
        eval_batch_size: int = field(default=8)
        # 评估批次大小，默认值为 8
        ppo_epochs: int = field(default=1)
        # PPO 训练轮数，默认值为 1
        n_epochs: int = field(default=40)
        # 训练总轮数，默认值为 40
        num_workers: int = field(default=8)
        # 数据加载的工作线程数，默认值为 8
        learning_rate: float = field(default=2e-5)
        # 学习率，默认值为 2e-5
        weight_decay: float = field(default=1e-6)
        # 权重衰减，默认值为 1e-6
        warmup_step: int = field(default=0)
        # 预热步骤数，默认值为 0
        clip_grad_norm: float = field(default=1)
        # 梯度裁剪的阈值，默认值为 1
        vf_coef: float = field(default=1.0)
        # 价值函数的系数，默认值为 1.0
        kl_coef: float = field(default=0.1)
        # KL 散度的系数，默认值为 0.1
        gamma: float = field(default=0.98)
        # 折扣因子，默认值为 0.98
        lam: float = field(default=0.95)
        # GAE 的 λ，默认值为 0.95
        ref_model_name_or_path: str = field(default="")
        # 参考模型名称或路径，默认值为空字符串
        evaluating_epoch_freq: int = field(default=1)
        # 评估的轮频，默认值为 1
        logging_epoch_freq: int = field(default=1)
        # 日志的轮频，默认值为 1
        saving_epoch_freq: int = field(default=1000)
        # 保存的轮频，默认值为 1000
        evaluating_step_freq: int = field(default=NONE_INT)
        # 评估的步频，默认值为 NONE_INT
        logging_step_freq: int = field(default=NONE_INT)
        # 日志的步频，默认值为 NONE_INT
        logging_seq_str_step_freq: int = field(default=NONE_INT)
        # 序列字符串日志的步频，默认值为 NONE_INT
        logging_values_step_freq: int = field(default=NONE_INT)
        # 值日志的步频，默认值为 NONE_INT
        saving_step_freq: int = field(default=NONE_INT)
        # 保存的步频，默认值为 NONE_INT
        seed: int = field(default=42)
        # 随机种子，默认值为 42
        max_input_length: int = field(default=700)
        # 最大输入长度，默认值为 700
        max_gen_length: int = field(default=700)
        # 最大生成长度，默认值为 700
        keep_num_ckpt: int = field(default=5)
        # 保留的检查点数量，默认值为 5
        # wandb stuff
        wandb_log: bool = field(default=False)
        # 是否启用 wandb 日志，默认值为 False
        wandb_project: str = field(default='tmp_anvfupsadfn')
        # wandb 项目名称，默认值为 'tmp_anvfupsadfn'
        wandb_run_name: str = field(default='default_run_name')
        # wandb 运行名称，默认值为 'default_run_name'
        ###
        engine: str = field(default='python')
        # 引擎类型，默认值为 'python'
        adv_whitening: str = field(default='global')
        # 先进白化方法，默认值为 'global'

    parser = HfArgumentParser(Arguments)
    # 创建参数解析器
    (args,) = parser.parse_args_into_dataclasses()
    # 解析命令行参数并转换为数据类
    args = asdict(args)
    # 将数据类转换为字典
    for k, v in args.items():
        if v in [NONE_INT, NONE_STR]:
            args[k] = None
            # 将无效值替换为 None
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])
    # 创建加速器实例，设置超时
    accelerator.print(args)
    # 打印参数
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    # 以 JSON 格式打印参数
    main(args)
    # 调用主函数
