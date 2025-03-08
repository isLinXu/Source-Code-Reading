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
from accelerate.utils import pad_across_processes, broadcast
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
from datetime import timedelta
from functools import partial
import json
import os
import random
from src.python_engine import run_python_code
from src.utils import set_seed, floatify, compute_ETA
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup, AdamW, get_constant_schedule_with_warmup
import wandb
import pandas as pd
import shutil
tqdm = partial(tqdm, ncols=0, leave=False)

TIMEOUT = 10
# 设置超时时间为 10 秒
instruction = None
# 初始化指令为 None
cot_trigger = None
# 初始化 COT 触发器为 None
answer_trigger = None
# 初始化答案触发器为 None

def setup_cot(src_name):
    # 定义设置 COT 的函数，接受源名称作为参数
    assert src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric']
    # 断言源名称必须在指定的选项中
    global instruction
    global cot_trigger
    global answer_trigger
    # 声明全局变量
    # Complete output is in this form: f'{instruction}{question.strip()}{cot_trigger}{answer_cot.strip()}'
    instruction = 'Question:\n'
    # 设置指令文本
    cot_trigger = '\nAnswer reasoning:\n'
    # 设置 COT 触发器文本
    answer_trigger = '\nTherefore, the answer is: '
    # 设置答案触发器文本
    return 

post_process_final_answer_fn_mapper = {
    # 定义处理最终答案的映射
    'gsm8k': lambda x: float(x.replace(',', '').strip()),
    # 对于 gsm8k，去掉逗号并转换为浮点数
    'svamp': lambda x: float(x.replace(',', '').strip()),
    # 对于 svamp，去掉逗号并转换为浮点数
    'mathqa': lambda x: x.lower().replace('"', '').replace("'", '').strip(),
    # 对于 mathqa，将字符串转换为小写，去掉引号并去除空格
    'mathqa-numeric': lambda x: float(x),
    # 对于 mathqa-numeric，直接转换为浮点数
}

### the answer_cot is a list of answer_cot
post_process_answer_cot_fn_mapper = {
    # 定义处理答案 COT 的映射
    ('python', 'gsm8k'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    # 对于 python 和 gsm8k，运行 Python 代码并将结果转换为浮点数
    ('python', 'svamp'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    # 对于 python 和 svamp，运行 Python 代码并将结果转换为浮点数
    ('python', 'mathqa'): lambda answer_cot: [str(res).lower().replace('"', '').replace("'", '').strip() for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    # 对于 python 和 mathqa，运行 Python 代码并处理结果
    ('python', 'mathqa-numeric'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    # 对于 python 和 mathqa-numeric，运行 Python 代码并将结果转换为浮点数
    ('nl', 'gsm8k'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    # 对于 nl 和 gsm8k，处理答案 COT
    ('nl', 'svamp'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    # 对于 nl 和 svamp，处理答案 COT
    ('nl', 'mathqa'): lambda answer_cot: [res.split(answer_trigger)[-1].lower().replace('"', '').replace("'", '').strip() for res in answer_cot],
    # 对于 nl 和 mathqa，处理答案 COT
    ('nl', 'mathqa-numeric'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    # 对于 nl 和 mathqa-numeric，处理答案 COT
}

compare_answer_fn_mapper = {
    # 定义比较答案的映射
    'gsm8k': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    # 对于 gsm8k，判断提取的答案与目标答案的绝对差是否小于等于 0.01
    'svamp': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    # 对于 svamp，判断提取的答案与目标答案的绝对差是否小于等于 0.01
    'mathqa': lambda extracted_ans, target_answer: extracted_ans == target_answer,
    # 对于 mathqa，判断提取的答案是否等于目标答案
    'mathqa-numeric': lambda extracted_ans, target_answer: abs(extracted_ans - target_answer) <= 1e-2,
    # 对于 mathqa-numeric，判断提取的答案与目标答案的绝对差是否小于等于 0.01
}

def prepare_datasets_and_data_loaders(args, tokenizer):
    # 定义准备数据集和数据加载器的函数，接受参数和分词器
    with accelerator.main_process_first():
        # 确保主进程优先执行
        raw_dataset = DatasetDict({
            # 创建数据集字典，包含训练集和测试集
            'train': Dataset.from_list(json.load(open(args['train_file'], 'r'))),
            # 从训练文件加载数据集
            'test': Dataset.from_list(json.load(open(args['test_file'], 'r'))),
            # 从测试文件加载数据集
        })
        accelerator.print('Raw data:', raw_dataset)
        # 打印原始数据集
        src_name = raw_dataset['train'][0]['item_id'].split('_')[0]  # e.g., gsm8k_0, gsm8k_1, gsm8k_2, ...
        # 获取源名称，例如 gsm8k
        setup_cot(src_name)
        # 设置 COT 相关的文本
        accelerator.print('Using instruction:', instruction)
        # 打印使用的指令
        accelerator.print('Using cot_trigger:', cot_trigger)
        # 打印使用的 COT 触发器
        accelerator.print('Using answer_trigger:', answer_trigger)
        # 打印使用的答案触发器

        def tokenize_fn(batch, args, tokenizer):
            # 定义分词函数，接受批次、参数和分词器
            assert tokenizer.eos_token_id is not None, (tokenizer.eos_token_id, tokenizer.eos_token)
            # 确保分词器的结束标记 ID 不为 None
            new_batch = defaultdict(list)
            # 初始化新的批次字典
            all_keys = list(batch.keys())
            # 获取批次中的所有键
            for item_values in zip(*(batch[k] for k in all_keys)):
                # 遍历批次中的每个项
                item = {k: item_values[i] for i, k in enumerate(all_keys)}
                # 创建项字典
                item_id, question, answer_value, answer_cot = \
                        item['item_id'], \
                        item['question'], \
                        item['answer_value'], \
                        item.get('answer_cot', None), \
                        # 解构项字典中的值

                question = question.strip()
                # 去除问题的前后空格
                if answer_value is not None:
                    answer_value = answer_value.strip()
                    # 去除答案值的前后空格

                if answer_cot is not None:
                    answer_cot = answer_cot.strip()
                    # 去除答案 COT 的前后空格
                    if args['engine'] == 'nl':
                        answer_cot += f'{answer_trigger}{answer_value}'
                        # 如果引擎为 nl，则添加答案触发器和答案值

                input = f'{instruction}{question}{cot_trigger}'
                # 构建输入文本
                output = f'{answer_cot}'
                # 构建输出文本
                prefix_text = f'{instruction}{question}{cot_trigger}'
                # 构建前缀文本

                input_encode = tokenizer(input, add_special_tokens=False)
                # 使用分词器对输入文本进行编码
                output_encode = tokenizer(output, add_special_tokens=False)
                # 使用分词器对输出文本进行编码
                prefix_encode = tokenizer(prefix_text, add_special_tokens=False)
                # 使用分词器对前缀文本进行编码

                input_ids = input_encode['input_ids'] + output_encode['input_ids'] + [tokenizer.eos_token_id]
                # 合并输入 ID 和输出 ID，并添加结束标记 ID
                labels = [-100] * len(input_encode['input_ids']) + output_encode['input_ids'] + [tokenizer.eos_token_id]
                # 创建标签，输入部分使用 -100 忽略
                attention_mask = [1] * len(input_ids)
                # 创建注意力掩码
                prefix = prefix_encode['input_ids']
                # 获取前缀的输入 ID
                prefix_attention_mask = prefix_encode['attention_mask']
                # 获取前缀的注意力掩码

                # Truncation
                input_ids_max_length = len(input_ids)
                # 获取输入 ID 的最大长度
                # assert input_ids_max_length <= args['max_input_length'], input_ids_max_length
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
                # 将输入 ID 添加到新批次
                new_batch['labels'].append(labels)
                # 将标签添加到新批次
                new_batch['attention_mask'].append(attention_mask)
                # 将注意力掩码添加到新批次
                new_batch['prefix'].append(prefix)
                # 将前缀添加到新批次
                new_batch['prefix_attention_mask'].append(prefix_attention_mask)
                # 将前缀注意力掩码添加到新批次
                ##
                new_batch['item_id'].append(item_id)
                # 将项 ID 添加到新批次
                new_batch['question'].append(question)
                # 将问题添加到新批次
                new_batch['answer_cot'].append(answer_cot)
                # 将答案 COT 添加到新批次
                new_batch['answer_value'].append(answer_value)
                # 将答案值添加到新批次
                new_batch['input_ids_max_length'].append(input_ids_max_length)
                # 将输入 ID 最大长度添加到新批次
            
            return new_batch
            # 返回新批次

        tokenized_dataset = DatasetDict({
            # 创建数据集字典
            mode: dataset.map(
                tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer}, batched=True, remove_columns=dataset.column_names, 
                num_proc=8, load_from_cache_file=False
                # 对原始数据集进行映射，使用 tokenize_fn 进行分词处理
            ) for mode, dataset in raw_dataset.items()})
        accelerator.print('Processed data:', tokenized_dataset)
        # 打印处理后的数据集
        for mode, dataset in tokenized_dataset.items():
            accelerator.print(mode, f'{mode}_input_ids_max_length', max(dataset['input_ids_max_length']))
            # 打印每个模式的输入 ID 最大长度

        if accelerator.is_main_process and args['wandb_log']:
            # 如果是主进程并且启用了 wandb 日志
            wandb.config.update({
                "src_name": src_name,
                # 记录源名称
                "instruction": instruction,
                # 记录指令
                "cot_trigger": cot_trigger,
                # 记录 COT 触发器
                "answer_trigger": answer_trigger,
                # 记录答案触发器
                "raw_dataset": str(raw_dataset),
                # 记录原始数据集
                "tokenized_dataset": str(tokenized_dataset),
                # 记录处理后的数据集
                "train_input_ids_max_length": max(tokenized_dataset['train']['input_ids_max_length']),
                # 记录训练集的输入 ID 最大长度
                "test_input_ids_max_length": max(tokenized_dataset['test']['input_ids_max_length']),
                # 记录测试集的输入 ID 最大长度
            })

    # def collate_fn(batch, args, tokenizer):
    #     max_input_length = max([len(item['input_ids']) for item in batch])
    #     max_target_length = max([len(item['labels']) for item in batch])
    #     max_prefix_length = max([len(item['prefix']) for item in batch])
    #     input_ids  = []
    #     attention_mask  = []
    #     labels, labels_left_padded  = [], []
    #     prefix_left_padded  = []
    #     prefix_attention_mask_left_padded  = []
    #     for item in batch:
    #         input_ids.append(item['input_ids'] + [tokenizer.pad_token_id]*(max_input_length - len(item['input_ids'])))
    #         attention_mask.append(item['attention_mask'] + [0]*(max_input_length - len(item['attention_mask'])))
    #         labels.append(item['labels'] + [-100]*(max_target_length - len(item['labels'])))

    #         labels_left_padded.append([-100]*(max_target_length - len(item['labels'])) + item['labels'])
    #         prefix_left_padded.append([tokenizer.pad_token_id]*(max_prefix_length - len(item['prefix'])) + item['prefix'])
    #         prefix_attention_mask_left_padded.append([0]*(max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask'])
    #     forward_kwargs = {
    #         'input_ids': torch.LongTensor(input_ids),
    #         'attention_mask': torch.BoolTensor(attention_mask),
    #         'labels': torch.LongTensor(labels)
    #     }
    #     generate_prefix_kwargs = {
    #         'input_ids': torch.LongTensor(prefix_left_padded),
    #         'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
    #         'labels': torch.LongTensor(labels_left_padded)
    #     }
    #     return {
    #         'forward_kwargs': forward_kwargs,
    #         'generate_prefix_kwargs': generate_prefix_kwargs,
    #     }

    # train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True, 
    #                     collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
                        
    # test_dataloader = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=args['eval_batch_size'], num_workers=args['num_workers'], pin_memory=True, 
    #                     collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
                        
    # return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['test'], test_dataloader)


    def collate_fn(batch, args, tokenizer):
        # 定义合并函数，接受批次、参数和分词器
        max_input_length = max([len(item['input_ids']) for item in batch])
        # 获取批次中输入 ID 的最大长度
        max_target_length = max([len(item['labels']) for item in batch])
        # 获取批次中标签的最大长度
        max_prefix_length = max([len(item['prefix']) for item in batch])
        # 获取批次中前缀的最大长度
        input_ids = []
        # 初始化输入 ID 列表
        attention_mask = []
        # 初始化注意力掩码列表
        labels, labels_left_padded = [], []
        # 初始化标签列表和左侧填充标签列表
        prefix_left_padded = []
        # 初始化左侧填充前缀列表
        prefix_attention_mask_left_padded = []
        # 初始化左侧填充前缀注意力掩码列表

        for item in batch:
            # 遍历批次中的每个项
            input_ids.append(item['input_ids'] + [tokenizer.pad_token_id] * (max_input_length - len(item['input_ids'])))
            # 将输入 ID 填充到最大长度
            attention_mask.append(item['attention_mask'] + [0] * (max_input_length - len(item['attention_mask'])))
            # 将注意力掩码填充到最大长度
            labels.append(item['labels'] + [-100] * (max_target_length - len(item['labels'])))
            # 将标签填充到最大长度，输入部分使用 -100 忽略

            labels_left_padded.append([-100] * (max_target_length - len(item['labels'])) + item['labels'])
            # 将标签左侧填充到最大长度
            prefix_left_padded.append([tokenizer.pad_token_id] * (max_prefix_length - len(item['prefix'])) + item['prefix'])
            # 将前缀左侧填充到最大长度
            prefix_attention_mask_left_padded.append([0] * (max_prefix_length - len(item['prefix_attention_mask'])) + item['prefix_attention_mask'])
            # 将前缀注意力掩码左侧填充到最大长度

        forward_kwargs = {
            'input_ids': torch.LongTensor(input_ids),
            # 将输入 ID 转换为 LongTensor
            'attention_mask': torch.BoolTensor(attention_mask),
            # 将注意力掩码转换为 BoolTensor
            'labels': torch.LongTensor(labels)
            # 将标签转换为 LongTensor
        }

        generate_prefix_kwargs = {
            'input_ids': torch.LongTensor(prefix_left_padded),
            # 将左侧填充的前缀输入 ID 转换为 LongTensor
            'attention_mask': torch.BoolTensor(prefix_attention_mask_left_padded),
            # 将左侧填充的前缀注意力掩码转换为 BoolTensor
            'labels': torch.LongTensor(labels_left_padded)
            # 将左侧填充的标签转换为 LongTensor
        }

        return {
            'forward_kwargs': forward_kwargs,
            # 返回前向传播的参数
            'generate_prefix_kwargs': generate_prefix_kwargs,
            # 返回生成前缀的参数
        }

    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True, 
        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
    # 创建训练数据加载器，使用合并函数 collate_fn

    test_dataloader = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=args['eval_batch_size'], num_workers=args['num_workers'], pin_memory=True, 
        collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
    # 创建测试数据加载器，使用合并函数 collate_fn

    return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['test'], test_dataloader)
    # 返回训练集和训练数据加载器，以及测试集和测试数据加载器


def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
    # 定义保存检查点的函数，接受参数、模型、分词器、保存路径和最近检查点路径列表
    os.makedirs(save_path, exist_ok=True)
    # 创建保存路径，如果已存在则不报错
    unwrapped_model = accelerator.unwrap_model(model)
    # 解包模型以获取原始模型
    unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
    # 保存模型到指定路径
    tokenizer.save_pretrained(save_path)
    # 保存分词器到指定路径
    if accelerator.is_main_process and most_recent_ckpts_paths is not None:
        # 如果是主进程并且最近检查点路径列表不为 None
        most_recent_ckpts_paths.append(save_path)
        # 将当前保存路径添加到最近检查点路径列表
        if args['keep_num_ckpt'] is not None and len(most_recent_ckpts_paths) > args['keep_num_ckpt']:
            # 如果设置了保留检查点数量，并且最近检查点数量超过限制
            ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
            # 移除最旧的检查点路径
            # os.remove(ckpt_to_be_removed)
            shutil.rmtree(ckpt_to_be_removed)
            # 删除对应的检查点文件夹

def train_one_epoch(args, model, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    global_step, test_dataset, test_dataloader, 
                    prefix, epoch, best_eval_log_dict, summary_log_dict, most_recent_ckpts_paths):
    # 定义训练一个周期的函数，接受参数、模型、数据集、数据加载器、优化器、调度器、分词器等

    model_dir = args['model_dir']
    # 获取模型保存目录
    clip_grad_norm = args.get('clip_grad_norm', None)
    # 获取梯度裁剪的阈值
    evaluating_step_freq = args.get('evaluating_step_freq', None)
    # 获取评估步频
    logging_step_freq = args.get('logging_step_freq', None)
    # 获取日志记录步频
    saving_step_freq = args.get('saving_step_freq', None)
    # 获取保存步频

    model.train()
    # 将模型设置为训练模式
    epoch_result_dict = defaultdict(list)
    # 初始化一个字典，用于存储每个周期的结果

    with tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_main_process, desc='Train Loop') as t:
        # 使用 tqdm 显示训练进度条
        for idx, batch in t:
            # 遍历训练数据加载器中的每个批次
            with accelerator.accumulate(model):
                # 在加速器上下文中累积梯度
                output = model(**batch['forward_kwargs'])
                # 使用模型进行前向传播，获取输出
                # Get some metrics
                loss = output[0]
                # 获取损失值
                result_dict, extra = {}, None
                # 初始化结果字典和额外信息

                # Update
                accelerator.backward(loss)
                # 反向传播损失
                if accelerator.sync_gradients:
                    # 如果同步梯度
                    if clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
                        # 裁剪模型参数的梯度
                optimizer.step()
                # 更新优化器
                optimizer.zero_grad()
                # 清零梯度
                # model.zero_grad()  # 这行代码被注释掉了
                if accelerator.sync_gradients:
                    scheduler.step()
                    # 更新学习率调度器
                
            if accelerator.sync_gradients:
                global_step += 1
                # 更新全局步数
                # Step update metric
                epoch_result_dict['loss'].append(loss.item()) 
                # 将当前损失添加到周期结果字典中
                for k, v in result_dict.items():
                    epoch_result_dict[k].append(v)
                    # 将其他指标添加到周期结果字典中

                # Step evaluating
                eval_log_dict = {}
                # 初始化评估日志字典
                is_best = False
                if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
                    # 如果设置了评估步频，并且当前全局步数满足评估条件
                    evaluate_result_dict = {f'Eval.Gen.{k}': v for k, v in evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer).items()}
                    # 进行评估并获取评估结果
                    eval_log_dict.update(evaluate_result_dict)
                    # 更新评估日志字典
                    if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', 0):
                        # 如果当前评估准确率超过最佳评估准确率
                        is_best = True
                        best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                        # 更新最佳评估准确率
                    if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                        summary_log_dict['Eval.Gen.value_accuracy'] = []
                    summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])
                    # 将当前评估准确率添加到摘要日志字典中

                # Step logging
                train_log_dict = {}
                if logging_step_freq is not None and global_step % logging_step_freq == 0:
                    # 如果设置了日志记录步频，并且当前全局步数满足记录条件
                    train_log_dict = {f'T.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in epoch_result_dict.items()}
                    # 计算并记录当前周期的平均训练指标
                
                if eval_log_dict or train_log_dict:
                    # 如果有评估日志或训练日志
                    log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                    # 创建日志字典，包含学习率和其他日志信息
                    if accelerator.is_main_process and args['wandb_log']:
                        wandb.log(log_dict, step=global_step)
                        # 如果是主进程并且启用了 wandb 日志，则记录日志
                        log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'], **log_dict}
                    log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k, v in log_dict.items()}
                    # 格式化日志字典中的浮点数
                    accelerator.print(f"{prefix}[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")
                    # 打印当前周期和全局步数的日志信息

                # Step saving
                if saving_step_freq is not None and global_step % saving_step_freq == 0:
                    # 如果设置了保存步频，并且当前全局步数满足保存条件
                    if is_best:
                        save_path = os.path.join(model_dir, f'best')
                        do_checkpoint(args, model, tokenizer, save_path)
                        # 保存最佳模型
                    if args['keep_num_ckpt'] > 0:
                        save_path = os.path.join(model_dir, f'global_step_{str(global_step)}')
                        do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)
                        # 保存当前模型

                # Keep only max_record items
                for k, v in epoch_result_dict.items():
                    if len(v) > 1:
                        epoch_result_dict[k] = v[-1:]
                        # 保留周期结果字典中的最后一个记录

    # Metric summary:
    epoch_result_dict = {k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()}
    # 计算并返回周期结果的平均值
    return epoch_result_dict, global_step

def evaluate_generation(args, model, dataset, dataloader, tokenizer):
    # 定义评估生成模型的函数，接受参数、模型、数据集、数据加载器和分词器

    model.eval()
    # 将模型设置为评估模式
    predictions = []
    # 初始化预测结果列表
    targets = []
    # 初始化目标结果列表

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process, desc='Evaluation Gen Loop'):
        # 使用 tqdm 显示评估进度条，遍历数据加载器中的每个批次
        output_ = model.module.generate(
            # 使用模型生成输出
            **batch['generate_prefix_kwargs'], 
            max_length=args['max_input_length'],
            output_scores=True,
            return_dict_in_generate=True,
            num_beams=1,
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = output_.sequences
        # 获取生成的 ID
        generated_ids = pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        # 在进程间填充生成的 ID

        labels = batch['generate_prefix_kwargs']['labels']
        # 获取标签
        labels = pad_across_processes(labels, dim=1, pad_index=tokenizer.pad_token_id, pad_first=True)
        # 在进程间填充标签
        labels[labels == -100] = tokenizer.pad_token_id
        # 将标签中的 -100 替换为填充标记 ID

        generated_ids, labels = accelerator.gather(generated_ids), accelerator.gather(labels)
        # 收集生成的 ID 和标签

        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for g in generated_ids]
        # 解码生成的 ID 为文本，去除特殊标记和多余空格
        predictions.extend(preds)
        # 将预测结果添加到列表中
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() for t in labels]
        # 解码标签为文本，去除特殊标记和多余空格
        targets.extend(target)
        # 将目标结果添加到列表中

    predictions = predictions[:len(dataset)]
    # 确保预测结果的长度与数据集相同
    targets = targets[:len(dataset)]
    # 确保目标结果的长度与数据集相同

    if accelerator.is_main_process and accelerator.is_local_main_process:
        # 如果是主进程并且是本地主进程
        results = []
        # 初始化结果列表
        src_name = dataset[0]['item_id'].split('_')[0]
        # 获取源名称，例如 gsm8k
        for pred, tar, item in zip(predictions, targets, dataset):
            # 遍历预测结果、目标和数据集中的每个项
            cur_res = {
                'item_id': item['item_id'],
                # 当前项的 ID
                'answer_value': item['answer_value'],
                # 当前项的答案值
            }
            ## Processing target
            target_cot = tar.strip().split(cot_trigger)[-1].strip()
            # 处理目标 COT
            target_value = post_process_final_answer_fn_mapper[src_name](cur_res['answer_value'])
            # 通过源名称处理目标值
            cur_res['target'] = target
            # 将目标添加到当前结果
            cur_res['target_cot'] = target_cot
            # 将目标 COT 添加到当前结果
            cur_res['target_value'] = target_value
            # 将目标值添加到当前结果
            ## Processing prediction
            prediction_cot = pred.strip().split(cot_trigger)[-1].strip()
            # 处理预测 COT
            cur_res['prediction'] = pred
            # 将预测结果添加到当前结果
            cur_res['prediction_cot'] = prediction_cot
            # 将预测 COT 添加到当前结果
            cur_res['prediction_value'] = None  # Tobe filled
            # 初始化预测值为 None，待后续填充
            results.append(cur_res)
            # 将当前结果添加到结果列表

        # save first before execute to trace error.
        res_path = args['model_dir'].rstrip('/') + '/' + '_res.json'
        # 设置结果保存路径
        with open(res_path, 'w') as f:
            json.dump(results, f, indent=2)
            # 将结果写入 JSON 文件

        execute_fn = post_process_answer_cot_fn_mapper[(args['engine'], src_name)]
        # 获取处理答案 COT 的函数
        corr_value = 0
        for i, prediction_value in enumerate(execute_fn([item['prediction_cot'] for item in results])):
            # 遍历结果并执行处理函数
            target_value = results[i]['target_value']
            # 获取当前目标值
            is_correct = compare_answer_fn_mapper[src_name](prediction_value, target_value) if prediction_value is not None else False
            # 比较预测值和目标值，判断是否正确
            results[i]['prediction_value'] = prediction_value
            # 将预测值添加到结果中
            results[i]['is_correct'] = is_correct
            # 将正确性标记添加到结果中
            corr_value += is_correct
            # 统计正确的预测值数量

        res_path = args['model_dir'].rstrip('/') + '/' + '_res.json'
        # 设置结果保存路径
        with open(res_path, 'w') as f:
            json.dump(results, f, indent=2)
            # 将结果写入 JSON 文件

        # if args['wandb_log']:
        #     table = wandb.Table(dataframe=pd.DataFrame(results))
        #     wandb.log({"predictions": table})

        value_accuracy = corr_value / len(results) * 100
        # 计算准确率
        accelerator.print(f"[Eval Info] value_accuracy: {value_accuracy:.5g}%")
        # 打印评估信息
        value_accuracy = torch.FloatTensor([value_accuracy]).to(accelerator.device)
        # 将准确率转换为 FloatTensor
    else:
        value_accuracy = torch.FloatTensor([-1.0]).to(accelerator.device)
        # 如果不是主进程，设置准确率为 -1

    value_accuracy = broadcast(value_accuracy).cpu().numpy().tolist()[0]
    # 广播准确率并转换为列表

    # Metric summary:
    model.train()
    # 将模型设置回训练模式
    return {'value_accuracy': value_accuracy}
    # 返回准确率

def main(args):
    # 定义主函数，接受参数

    set_seed(args['seed'] + accelerator.process_index)
    # 设置随机种子以确保可重复性

    if torch.distributed.get_rank() == 0 and args['wandb_log']:
        # 如果当前进程是主进程并且启用了 wandb 日志
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
    # 准备数据集和数据加载器

    model = AutoModelForCausalLM.from_pretrained(args['model_name_or_path'], low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    # 从预训练模型加载因果语言模型
    accelerator.print(f'[Vocab size]: {len(tokenizer)}')    
    # 打印词汇表大小
    model.resize_token_embeddings(len(tokenizer))
    # 调整模型的嵌入层大小以匹配分词器的词汇表大小

    if accelerator.is_main_process and args['wandb_log']:
        # 如果是主进程并且启用了 wandb 日志
        wandb.run.summary.update({
            'pad_token_id': tokenizer.pad_token_id,
            # 记录填充标记 ID
            'eos_token_id': tokenizer.eos_token_id,
            # 记录结束标记 ID
            'unk_token_id': tokenizer.unk_token_id,
            # 记录未知标记 ID
            'vocab_size': len(tokenizer)
            # 记录词汇表大小
        })

    n_epochs = args['n_epochs']
    # 获取训练周期数
    num_training_steps = (len(train_dataloader) // accelerator.num_processes * n_epochs) // args['gradient_accumulation_steps']
    # 计算总训练步骤数
    warmup_step = args['warmup_step'] if args['warmup_step'] is not None and args['warmup_step'] >= 0 else int(0.1 * num_training_steps)
    # 设置预热步骤数
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            # 获取不包含偏置和 LayerNorm 权重的模型参数
            "weight_decay": args['weight_decay'],
            # 设置权重衰减
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])],
            # 获取包含偏置和 LayerNorm 权重的模型参数
            "weight_decay": 0.0,
            # 不进行权重衰减
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=1e-8)
    # 创建 AdamW 优化器
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=num_training_steps)
    # 创建线性学习率调度器
    # scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step)  # 这行代码被注释掉了

    accelerator.print(
        f"***** Running training *****\n"
        f"  Num examples = {len(train_dataset)}\n"
        f"  Num Epochs = {n_epochs}\n"
        f"  Instantaneous batch size per device = {args['batch_size']}\n"
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args['batch_size'] * accelerator.num_processes * args['gradient_accumulation_steps']}\n"
        f"  Total optimization steps = {num_training_steps}\n"
        f"  Warm up step: {warmup_step}\n"
        f"  Learning rate: {args['learning_rate']}\n"
    )   
    # 打印训练相关信息

    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(model, optimizer, train_dataloader, test_dataloader)
    # 准备模型、优化器和数据加载器以进行加速
    
    global_step = 0
    # 初始化全局步数
    evaluating_epoch_freq = args['evaluating_epoch_freq']
    # 获取评估周期频率
    logging_epoch_freq = args['logging_epoch_freq']
    # 获取日志记录周期频率
    saving_epoch_freq = args['saving_epoch_freq']
    # 获取保存周期频率
    model_dir = args['model_dir']
    # 获取模型保存目录
    best_eval_log_dict = {}
    # 初始化最佳评估日志字典
    summary_log_dict = {}
    # 初始化摘要日志字典
    os.makedirs(model_dir, exist_ok=True)
    # 创建模型保存目录，如果已存在则不报错
    most_recent_ckpts_paths = []
    # 初始化最近检查点路径列表

    with tqdm(range(1, n_epochs + 1), total=n_epochs, disable=False) as t:
        # 使用 tqdm 显示训练周期进度条
        for epoch in t:
            # 遍历每个训练周期
            kwargs = {
                'args': args,
                'model': model, 
                'train_dataset': train_dataset, 
                'train_dataloader': train_dataloader, 
                'test_dataset': test_dataset,
                'test_dataloader': test_dataloader,
                'optimizer': optimizer, 
                'scheduler': scheduler,
                'global_step': global_step, 
                'tokenizer': tokenizer,
                'prefix': '', 
                'epoch': epoch,
                'best_eval_log_dict': best_eval_log_dict,
                'summary_log_dict': summary_log_dict,
                'most_recent_ckpts_paths': most_recent_ckpts_paths,
            }
            # 准备训练周期所需的参数字典
            train_epoch_result_dict, global_step = train_one_epoch(**kwargs)
            # 训练一个周期并获取结果和全局步数

            eval_log_dict = {}
            # 初始化评估日志字典
            is_best = False
            if evaluating_epoch_freq is not None and epoch % evaluating_epoch_freq == 0:
                # 如果设置了评估周期频率，并且当前周期满足评估条件
                evaluate_result_dict = {f'Eval.Gen.{k}': v for k, v in evaluate_generation(args, model, test_dataset, test_dataloader, tokenizer).items()}
                # 进行评估并获取评估结果
                eval_log_dict.update(evaluate_result_dict)
                # 更新评估日志字典
                if eval_log_dict['Eval.Gen.value_accuracy'] > best_eval_log_dict.get('Eval.Gen.value_accuracy_best', 0):
                    # 如果当前评估准确率超过最佳评估准确率
                    is_best = True
                    best_eval_log_dict['Eval.Gen.value_accuracy_best'] = eval_log_dict['Eval.Gen.value_accuracy']
                    # 更新最佳评估准确率
                if 'Eval.Gen.value_accuracy' not in summary_log_dict:
                    summary_log_dict['Eval.Gen.value_accuracy'] = []
                summary_log_dict['Eval.Gen.value_accuracy'].append(eval_log_dict['Eval.Gen.value_accuracy'])
                # 将当前评估准确率添加到摘要日志字典中

            train_log_dict = {}
            # 初始化训练日志字典
            if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
                # 如果设置了日志记录周期频率，并且当前周期满足记录条件
                train_log_dict = {f'T.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in train_epoch_result_dict.items()}
                # 计算并记录当前周期的平均训练指标

            if eval_log_dict or train_log_dict:
                # 如果有评估日志或训练日志
                log_dict = {'lr': scheduler.get_last_lr()[0], **train_log_dict, **eval_log_dict, **best_eval_log_dict}
                # 创建日志字典，包含学习率和其他日志信息
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log(log_dict, step=global_step)
                    # 如果是主进程并且启用了 wandb 日志，则记录日志
                    log_dict = {'wandb': args['wandb_project'] + '|' + args['wandb_run_name'], **log_dict}
                log_dict = {k: f'{v:.5g}' if isinstance(v, float) else v for k, v in log_dict.items()}
                # 格式化日志字典中的浮点数
                accelerator.print(f"[E={epoch}/{args['n_epochs']}, S={global_step}] {log_dict}")
                # 打印当前周期和全局步数的日志信息

            if saving_epoch_freq is not None and epoch % saving_epoch_freq == 0:
                # 如果设置了保存周期频率，并且当前周期满足保存条件
                if is_best:
                    save_path = os.path.join(model_dir, f'best')
                    do_checkpoint(args, model, tokenizer, save_path)
                    # 保存最佳模型
                if args['keep_num_ckpt'] > 0:
                    save_path = os.path.join(args['model_dir'], f'global_step_{str(global_step)}')
                    do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)
                    # 保存当前模型

    return 
    # 返回

if __name__ == '__main__':
    # 如果当前模块是主模块
    from transformers import HfArgumentParser
    # 从 transformers 库导入 HfArgumentParser
    NONE_INT = -100 
    # 定义一个常量，表示无效整数
    NONE_STR = 'None'
    # 定义一个常量，表示无效字符串

    @dataclass
    class Arguments:
        # 定义一个数据类，用于存储参数
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
        batch_size: int = field(default=4)
        # 批次大小，默认值为 4
        eval_batch_size: int = field(default=8)
        # 评估批次大小，默认值为 8
        n_epochs: int = field(default=40)
        # 训练周期数，默认值为 40
        num_workers: int = field(default=8)
        # 数据加载时使用的工作线程数，默认值为 8
        learning_rate: float = field(default=2e-5)
        # 学习率，默认值为 2e-5
        weight_decay: float = field(default=1e-6)
        # 权重衰减，默认值为 1e-6
        warmup_step: int = field(default=0)
        # 预热步骤数，默认值为 0
        clip_grad_norm: float = field(default=1)
        # 梯度裁剪的阈值，默认值为 1
        evaluating_epoch_freq: int = field(default=1)
        # 评估周期频率，默认值为 1
        logging_epoch_freq: int = field(default=1)
        # 日志记录周期频率，默认值为 1
        saving_epoch_freq: int = field(default=1000)
        # 保存周期频率，默认值为 1000
        evaluating_step_freq: int = field(default=NONE_INT)
        # 评估步频，默认值为 NONE_INT
        logging_step_freq: int = field(default=NONE_INT)
        # 日志记录步频，默认值为 NONE_INT
        saving_step_freq: int = field(default=NONE_INT)
        # 保存步频，默认值为 NONE_INT
        seed: int = field(default=42)
        # 随机种子，默认值为 42
        max_input_length: int = field(default=700)
        # 最大输入长度，默认值为 700
        gradient_accumulation_steps: int = field(default=1)
        # 梯度累积步数，默认值为 1
        keep_num_ckpt: int = field(default=1)
        # 保留检查点的数量，默认值为 1
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

    accelerator = Accelerator(gradient_accumulation_steps=args['gradient_accumulation_steps'], kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))]) 
    # 创建加速器实例，设置梯度累积步数，并定义初始化进程组的超时
    accelerator.print(args)
    # 打印参数
    accelerator.print(json.dumps(args, indent=2, ensure_ascii=False))
    # 以 JSON 格式打印参数，确保中文字符正常显示
    main(args)
    # 调用主函数