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
# 从 accelerate 库导入 Accelerator 和 InitProcessGroupKwargs
from collections import defaultdict, Counter
# 从 collections 库导入 defaultdict 和 Counter
from dataclasses import dataclass, field, asdict
# 从 dataclasses 库导入数据类相关功能
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, concatenate_datasets
# 从 datasets 库导入数据集相关功能
from datetime import timedelta
# 从 datetime 库导入 timedelta，用于时间计算
from functools import partial
# 从 functools 库导入 partial，用于部分应用函数
import json
# 导入 json 库，用于处理 JSON 数据
import numpy as np
# 导入 numpy 库，用于数值计算
import os
# 导入 os 库，用于文件和目录操作
from src.utils import set_seed
# 从自定义模块 src.utils 导入 set_seed 函数
from tqdm import tqdm
# 从 tqdm 库导入 tqdm，用于显示进度条
import torch
# 导入 PyTorch 库
from torch.utils.data import DataLoader
# 从 PyTorch 导入 DataLoader 类，用于数据加载
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AdamW
# 从 transformers 库导入相关模型和优化器
import wandb
# 导入 wandb 库，用于实验跟踪和可视化
import shutil
# 导入 shutil 库，用于文件操作

tqdm = partial(tqdm, ncols=0, leave=False)
# 使用 partial 函数设置 tqdm 的默认参数

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
                item_id, question, answer_value, predictions, vote_correctness = \
                        item['item_id'], \
                        item['question'], \
                        item['answer_value'], \
                        item['predictions'], \
                        item['is_correct']
                # 解构项字典中的值

                question, answer_value = question.strip(), answer_value.strip()
                # 去除问题和答案值的前后空格
                unique_ = set()
                # 初始化唯一性集合
                for sample in predictions:
                    # 遍历预测结果
                    prediction_cot, prediction_correctness, prediction_value = sample['completion'], sample['correctness'], sample['solving_res']
                    # 解构预测结果中的值
                    # deduplication
                    if prediction_cot in unique_:
                        continue
                    # 如果预测 COT 已存在，则跳过
                    unique_.add(prediction_cot)
                    # 将预测 COT 添加到唯一性集合

                    input = f'{instruction}{question}{cot_trigger}{prediction_cot}'
                    # 构建输入文本
                    input_encode = tokenizer(input, add_special_tokens=False)
                    # 使用分词器对输入文本进行编码
                    input_ids = input_encode['input_ids']
                    # 获取输入 ID
                    attention_mask = [1] * len(input_ids)
                    # 创建注意力掩码
                    labels = prediction_correctness 
                    # 获取标签

                    # Truncation and filtering 
                    input_ids = input_ids[:args['max_input_length']]
                    # 截断输入 ID
                    attention_mask = attention_mask[:args['max_input_length']]
                    # 截断注意力掩码
                    assert tokenizer.pad_token_id not in input_ids, input_ids
                    # 确保输入 ID 中不包含填充标记 ID
                    ##
                    new_batch['input_ids'].append(input_ids)
                    # 将输入 ID 添加到新批次
                    new_batch['labels'].append(labels)
                    # 将标签添加到新批次
                    new_batch['attention_mask'].append(attention_mask)
                    # 将注意力掩码添加到新批次
                    ##
                    new_batch['item_id'].append(item_id)
                    # 将项 ID 添加到新批次
                    new_batch['question'].append(question)
                    # 将问题添加到新批次
                    new_batch['prediction_cot'].append(prediction_cot)
                    # 将预测 COT 添加到新批次
                    new_batch['prediction_correctness'].append(prediction_correctness)
                    # 将预测正确性添加到新批次
                    new_batch['prediction_value'].append(prediction_value)
                    # 将预测值添加到新批次
                    new_batch['answer_value'].append(answer_value)
                    # 将答案值添加到新批次
                    new_batch['vote_correctness'].append(vote_correctness)
                    # 将投票正确性添加到新批次
                
            return new_batch
            # 返回新批次

        tokenized_dataset = DatasetDict({
            # 创建数据集字典
            mode: dataset.map(
                tokenize_fn, fn_kwargs={'args': args, 'tokenizer': tokenizer}, batched=True, remove_columns=dataset.column_names, 
                num_proc=16, load_from_cache_file=True
            ) for mode, dataset in raw_dataset.items()})
            # 对原始数据集进行映射，使用 tokenize_fn 进行分词处理
        accelerator.print('Processed data:', tokenized_dataset)
        # 打印处理后的数据集


    def collate_fn(batch, args, tokenizer):
        # 定义 collate_fn 函数，接受批次、参数和分词器
        max_input_length = max([len(item['input_ids']) for item in batch])
        # 计算批次中输入 ID 的最大长度
        input_ids  = []
        # 初始化输入 ID 列表
        attention_mask  = []
        # 初始化注意力掩码列表
        labels  = []
        # 初始化标签列表
        for item in batch:
            # 遍历批次中的每个项
            input_ids.append(item['input_ids'] + [tokenizer.pad_token_id] * (max_input_length - len(item['input_ids'])))
            # 将输入 ID 填充到最大长度
            attention_mask.append(item['attention_mask'] + [0] * (max_input_length - len(item['attention_mask'])))
            # 将注意力掩码填充到最大长度
            labels.append(item['labels'])
            # 添加标签

        forward_kwargs = {
            'input_ids': torch.LongTensor(input_ids),
            # 将输入 ID 转换为 LongTensor
            'attention_mask': torch.BoolTensor(attention_mask),
            # 将注意力掩码转换为 BoolTensor
            'labels': torch.LongTensor(labels),
            # 将标签转换为 LongTensor
        }
        return {
            'forward_kwargs': forward_kwargs,
            # 返回包含前向传播参数的字典
        }

    train_dataloader = DataLoader(tokenized_dataset['train'], shuffle=True, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True, 
                            collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
    # 创建训练数据加载器，设置打乱、批次大小、工作线程数和内存固定，并使用 collate_fn 进行数据整理
                            
    test_dataloader = DataLoader(tokenized_dataset['test'], shuffle=False, batch_size=args['batch_size'], num_workers=args['num_workers'], pin_memory=True, 
                            collate_fn=partial(collate_fn, args=args, tokenizer=tokenizer))
    # 创建测试数据加载器，设置不打乱、批次大小、工作线程数和内存固定，并使用 collate_fn 进行数据整理
                            
    return (tokenized_dataset['train'], train_dataloader), (tokenized_dataset['test'], test_dataloader)
    # 返回训练集和训练数据加载器，以及测试集和测试数据加载器的元组

    def do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths=None):
        # 定义保存检查点的函数，接受参数、模型、分词器、保存路径和最近检查点路径列表
        os.makedirs(save_path, exist_ok=True)
        # 创建保存路径，如果已存在则不报错
        unwrapped_model = accelerator.unwrap_model(model)
        # 解包模型以便保存
        unwrapped_model.save_pretrained(save_path, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=accelerator.get_state_dict(model))
        # 保存模型到指定路径
        tokenizer.save_pretrained(save_path)
        # 保存分词器到指定路径
        if accelerator.is_main_process and most_recent_ckpts_paths is not None:
            # 如果是主进程并且最近检查点路径列表不为 None
            most_recent_ckpts_paths.append(save_path)
            # 将当前保存路径添加到最近检查点路径列表
            if args['keep_num_ckpt'] is not None and len(most_recent_ckpts_paths) > args['keep_num_ckpt']:
                # 如果设置了保留的检查点数量，并且最近检查点数量超过限制
                ckpt_to_be_removed = most_recent_ckpts_paths.pop(0)
                # 移除最旧的检查点路径
                # os.remove(ckpt_to_be_removed)  # 注释掉的代码，用于删除文件
                shutil.rmtree(ckpt_to_be_removed)
                # 删除最旧的检查点目录

def train_one_epoch(args, model, train_dataset, train_dataloader, optimizer, scheduler, tokenizer,
                    global_step, test_dataset, test_dataloader, 
                    prefix, epoch, best_eval_log_dict, most_recent_ckpts_paths):
    # 定义训练一个周期的函数，接受参数、模型、训练数据集、训练数据加载器、优化器、调度器、分词器、
    # 全局步数、测试数据集、测试数据加载器、前缀、当前周期、最佳评估日志字典和最近检查点路径列表

    max_epoch = args['n_epochs']
    # 获取最大训练周期数
    model_dir = args['model_dir']
    # 获取模型保存目录
    clip_grad_norm = args.get('clip_grad_norm', None)
    # 获取梯度裁剪的阈值
    evaluating_step_freq = args.get('evaluating_step_freq', None)
    # 获取评估步骤频率
    logging_step_freq = args.get('logging_step_freq', None)
    # 获取日志记录步骤频率
    saving_step_freq = args.get('saving_step_freq', None)
    # 获取保存步骤频率

    model.train()
    # 将模型设置为训练模式
    epoch_result_dict = defaultdict(list)
    # 初始化一个字典，用于存储每个周期的结果

    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), disable=not accelerator.is_main_process, desc='Train Loop'):
        # 遍历训练数据加载器中的每个批次，并显示进度条
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
        if clip_grad_norm is not None:
            accelerator.clip_grad_norm_(model.parameters(), clip_grad_norm)
            # 如果设置了梯度裁剪，则裁剪模型参数的梯度
        optimizer.step()
        # 更新优化器
        scheduler.step()
        # 更新学习率调度器
        model.zero_grad()
        # 将模型的梯度清零
        global_step += 1
        # 增加全局步数

        # Step update metric
        epoch_result_dict['loss'].append(loss.item()) 
        # 将当前损失添加到周期结果字典中
        for k, v in result_dict.items():
            epoch_result_dict[k].append(v)
            # 将其他结果添加到周期结果字典中

        # Step evaluating
        eval_log_dict = {}
        is_best = False
        if evaluating_step_freq is not None and global_step % evaluating_step_freq == 0:
            # 如果设置了评估步骤频率，并且当前全局步数满足评估条件
            evaluate_result_dict = {f'Eval.Rerank.{k}':  v for k, v in evaluate_rerank(args, model, test_dataset, test_dataloader, tokenizer).items()}
            # 进行评估并获取评估结果
            eval_log_dict.update(evaluate_result_dict)
            # 更新评估日志字典
            if eval_log_dict['Eval.Rerank.rerank_acc'] > best_eval_log_dict.get('Eval.Rerank.rerank_acc_best', 0):
                # 如果当前评估准确率超过最佳评估准确率
                is_best = True
                best_eval_log_dict['Eval.Rerank.rerank_acc_best'] = eval_log_dict['Eval.Rerank.rerank_acc']
                # 更新最佳评估准确率

        # Step logging
        train_log_dict = {}
        if logging_step_freq is not None and global_step % logging_step_freq == 0:
            # 如果设置了日志记录步骤频率，并且当前全局步数满足记录条件
            train_log_dict = {f'Train.{k}': sum(v)/len(v) if isinstance(v, list) else v for k, v in epoch_result_dict.items()}
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
            
        # Step saving
        if saving_step_freq is not None and global_step % saving_step_freq == 0:
            # 如果设置了保存步骤频率，并且当前全局步数满足保存条件
            if is_best:
                save_path = os.path.join(model_dir, f'best')
                # 设置最佳模型保存路径
                do_checkpoint(args, model, tokenizer, save_path)
                # 保存最佳模型
            if args['keep_num_ckpt'] > 0:
                save_path = os.path.join(args['model_dir'], f'global_step_{str(global_step)}')
                # 设置当前全局步数的模型保存路径
                do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)
                # 保存当前模型

        # Keep only max_record items
        for k, v in epoch_result_dict.items():
            if len(v) > 1:
                epoch_result_dict[k] = v[-1:]
            # 保留每个指标的最后一个记录

    # Metric summary:
    epoch_result_dict = {k: (sum(v) / len(v) if isinstance(v, list) else v) for k, v in epoch_result_dict.items()}
    # 计算每个指标的平均值
    return epoch_result_dict, global_step
    # 返回周期结果字典和全局步数

def evaluate_rerank(args, model, dataset, dataloader, tokenizer):
    # 定义评估重排序的函数，接受参数、模型、数据集、数据加载器和分词器
    model.eval()
    # 将模型设置为评估模式
    epoch_result_dict = defaultdict(list)
    # 初始化一个字典，用于存储每个周期的结果
    predictions = []
    # 初始化预测列表
    probabilities = []
    # 初始化概率列表
    targets = []
    # 初始化目标列表

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), disable=not accelerator.is_main_process, desc='Evaluation Loop'):
        # 遍历数据加载器中的每个批次，并显示进度条
        output = model(**batch['forward_kwargs'])
        # 使用模型进行前向传播，获取输出
        
        # Get some metrics:
        loss = output[0]
        # 获取损失值

        # Step update metric
        loss = accelerator.gather(loss).mean()
        # 收集损失并计算平均值
        epoch_result_dict['loss'].append(loss.item())
        # 将当前损失添加到周期结果字典中

        # Prediction 
        logits = output[1]
        # 获取模型的输出 logits
        labels = batch['forward_kwargs']['labels']
        # 获取标签

        # Gather
        logits, labels = accelerator.gather(logits), accelerator.gather(labels)
        # 收集 logits 和标签
        probs = torch.softmax(logits, dim=-1)
        # 对 logits 应用 softmax 函数以获取概率分布
        probs, labels = probs.cpu().float().numpy(), labels.cpu().numpy()
        # 将概率和标签转换为 NumPy 数组
        preds = np.argmax(probs, axis=-1)
        # 获取每个样本的预测类别

        predictions.extend(preds.tolist())
        # 将预测结果添加到预测列表
        probabilities.extend(probs.tolist())
        # 将概率结果添加到概率列表
        targets.extend(labels.tolist())
        # 将目标标签添加到目标列表

    # Pred
    predictions = predictions[:len(dataset)]
    # 确保预测结果的长度与数据集相同
    probabilities = probabilities[:len(dataset)]
    # 确保概率结果的长度与数据集相同
    targets = targets[:len(dataset)]
    # 确保目标标签的长度与数据集相同

    cls_acc = (np.array(predictions) == np.array(targets)).mean()
    # 计算分类准确率

    # Gathering from multiple sample
    item_id_to_result = defaultdict(list)
    # 初始化一个字典，将项 ID 映射到结果列表
    for pred, tar, prob, item in zip(predictions, targets, probabilities, dataset):
        # 遍历预测结果、目标、概率和数据集中的每个项
        item_id = item.get('item_id', None)
        # 获取项 ID
        item_id_to_result[item_id].append({
            'item_id': item_id,
            # 'question': item['question'],
            # 'answer_value': item['answer_value'],
            # 'prediction_cot': item['prediction_cot'].split('\n'),
            'prediction_value': item['prediction_value'],
            # 预测值
            'vote_correctness': item['vote_correctness'],
            # 投票正确性
            'prediction_correctness': item['prediction_correctness'],
            # 预测正确性
            'cls_prob_tokens': prob,
            # 类别概率
            # 'cls_tar_tokens': tar,
            # 'cls_pred_tokens': pred,
        })

    src_name = item_id.split('_')[0]
    # 从项 ID 中提取源名称
    rerank_acc = []
    # 初始化重排序准确率列表
    rerank_ub = []
    # 初始化重排序上限列表
    vote_correctness = []
    # 初始化投票正确性列表
    for item_id, group in item_id_to_result.items():
        # 遍历每个项 ID 及其对应的结果组
        # Upper bound:
        upper_bound = 0
        if any([item['prediction_correctness'] for item in group]):
            upper_bound = 1
        # 如果组内有任何预测正确性为真，则上限为 1
        rerank_ub.append(upper_bound)
        # 将上限添加到上限列表

        # Last score among prediction with valid prediction_value
        valid_predictions = []
        for item in group:
            # 遍历组内每个项
            if item['prediction_value'] is not None:
                # 如果预测值不为 None
                if src_name == 'mathqa':
                    if not item['prediction_value'].strip().lower().isalpha():
                        continue
                    # 对于 mathqa，确保预测值是字母
                    if len(item['prediction_value'].strip().lower()) != 1:
                        continue
                    # 确保预测值的长度为 1
                valid_predictions.append(item)
                # 将有效预测添加到有效预测列表

        # Last score and vote acc
        if valid_predictions:
            last_score = [item['cls_prob_tokens'][1] for item in valid_predictions]
            # 获取有效预测的最后得分
            last_score_pred = valid_predictions[int(np.argmax(last_score))]
            # 找到得分最高的预测
            rerank_acc.append(last_score_pred['prediction_correctness'])
            # 将预测正确性添加到重排序准确率列表
            vote_correctness.append(last_score_pred['vote_correctness'])
            # 将投票正确性添加到投票正确性列表
        else:
            rerank_acc.append(0)
            # 如果没有有效预测，重排序准确率为 0
            vote_correctness.append(0)
            # 投票正确性为 0

    model.train()
    # 将模型设置回训练模式
    return {
        'cls_acc': cls_acc, 
        # 返回分类准确率
        'vote_acc': sum(vote_correctness) / len(vote_correctness), 
        # 返回投票准确率
        'rerank_acc': sum(rerank_acc) / len(rerank_acc), 
        # 返回重排序准确率
        'upper_bound': sum(rerank_ub) / len(rerank_ub)
        # 返回重排序上限
    }

def main(args):
    # 定义主函数，接受参数
    set_seed(args['seed'] + accelerator.process_index)
    # 设置随机种子，确保可重复性

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
    model = AutoModelForSequenceClassification.from_pretrained(args['model_name_or_path'], num_labels=2, torch_dtype=torch.bfloat16)
    # 从预训练模型加载序列分类模型
    model.config.pad_token_id = tokenizer.pad_token_id
    # 将模型的填充标记 ID 设置为分词器的填充标记 ID
    model.config.eos_token_id = tokenizer.eos_token_id
    # 将模型的结束标记 ID 设置为分词器的结束标记 ID
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
    most_recent_ckpts_paths = []
    # 初始化最近检查点路径列表
    os.makedirs(model_dir, exist_ok=True)
    # 创建模型保存目录，如果已存在则不报错

    for epoch in range(1, n_epochs + 1):
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
            evaluate_result_dict = {f'Eval.Rerank.{k}': v for k, v in evaluate_rerank(args, model, test_dataset, test_dataloader, tokenizer).items()}
            # 进行评估并获取评估结果
            eval_log_dict.update(evaluate_result_dict)
            # 更新评估日志字典
            if eval_log_dict['Eval.Rerank.rerank_acc'] > best_eval_log_dict.get('Eval.Rerank.rerank_acc_best', 0):
                # 如果当前评估准确率超过最佳评估准确率
                is_best = True
                best_eval_log_dict['Eval.Rerank.rerank_acc_best'] = eval_log_dict['Eval.Rerank.rerank_acc']
                # 更新最佳评估准确率

        train_log_dict = {}
        if logging_epoch_freq is not None and epoch % logging_epoch_freq == 0:
            # 如果设置了日志记录周期频率，并且当前周期满足记录条件
            train_log_dict = {f'Train.{k}': sum(v) / len(v) if isinstance(v, list) else v for k, v in train_epoch_result_dict.items()}
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
                # 设置最佳模型保存路径
                do_checkpoint(args, model, tokenizer, save_path)
                # 保存最佳模型
            if args['keep_num_ckpt'] > 0:
                save_path = os.path.join(args['model_dir'], f'global_step_{str(global_step)}_epoch_{epoch}')
                # 设置当前全局步数和周期的模型保存路径
                do_checkpoint(args, model, tokenizer, save_path, most_recent_ckpts_paths)
                # 保存当前模型
