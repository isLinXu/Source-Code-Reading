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
from collections import Counter
# 从 collections 库导入 Counter，用于计数
from dataclasses import dataclass, field, asdict
# 从 dataclasses 库导入数据类相关功能
from datasets import Dataset
# 从 datasets 库导入 Dataset 类
from datetime import timedelta
# 从 datetime 库导入 timedelta，用于时间计算
import deepspeed
# 导入 deepspeed 库
from functools import partial
# 从 functools 库导入 partial，用于部分应用函数
import json
# 导入 json 库，用于处理 JSON 数据
import os
# 导入 os 库，用于文件和目录操作
from src.python_engine import run_python_code
# 从自定义模块 src.python_engine 导入 run_python_code 函数
from src.utils import set_seed, write_data, floatify
# 从自定义模块 src.utils 导入 set_seed、write_data 和 floatify 函数
import torch
# 导入 PyTorch 库
from torch.utils.data import DataLoader
# 从 PyTorch 导入 DataLoader 类，用于数据加载
from transformers import AutoTokenizer, AutoModelForCausalLM
# 从 transformers 库导入 AutoTokenizer 和 AutoModelForCausalLM
from tqdm import tqdm
# 从 tqdm 库导入 tqdm，用于显示进度条
from typing import Dict
# 从 typing 库导入 Dict 类型
import wandb
# 导入 wandb 库，用于实验跟踪和可视化

tqdm = partial(tqdm, ncols=0, leave=False)
# 使用 partial 函数设置 tqdm 的默认参数

TIMEOUT = 10
# 设置超时时间为 10 秒
instruction=None
# 初始化指令为 None
cot_trigger=None
# 初始化 COT 触发器为 None
answer_trigger=None
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
    'gsm8k': lambda x: float(x.replace(',','').strip()),
    # 对于 gsm8k，去掉逗号并转换为浮点数
    'svamp': lambda x: float(x.replace(',','').strip()),
    # 对于 svamp，去掉逗号并转换为浮点数
    'mathqa': lambda x: x.lower().replace('"','').replace("'",'').strip(),
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
    ('python', 'mathqa'): lambda answer_cot: [str(res).lower().replace('"','').replace("'",'').strip() for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    # 对于 python 和 mathqa，运行 Python 代码并处理结果
    ('python', 'mathqa-numeric'): lambda answer_cot: [floatify(res) for res in run_python_code(programs=answer_cot, TIMEOUT=TIMEOUT)],
    # 对于 python 和 mathqa-numeric，运行 Python 代码并将结果转换为浮点数
    ('nl', 'gsm8k'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    # 对于 nl 和 gsm8k，处理答案 COT
    ('nl', 'svamp'): lambda answer_cot: [floatify(res.split(answer_trigger)[-1].strip()) for res in answer_cot],
    # 对于 nl 和 svamp，处理答案 COT
    ('nl', 'mathqa'): lambda answer_cot: [res.split(answer_trigger)[-1].lower().replace('"','').replace("'",'').strip() for res in answer_cot],
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

def tokenize_fn(examples: Dict, tokenizer, max_length, src_name, engine):
    # 定义 tokenize_fn 函数，接受示例字典、分词器、最大长度、源名称和引擎类型作为参数
    features = {"input_ids": [], "attention_mask": [], "answer_value": [], "answer_cot": [], "question": [], 'item_id': []}
    # 初始化特征字典，包含输入 ID、注意力掩码、答案值、答案 COT、问题和项 ID

    for idx, question in enumerate(examples["question"]):
        # 遍历示例中的每个问题，获取索引和问题文本
        text = f"{instruction}{question}{cot_trigger}"
        # 构建文本，包含指令、问题和 COT 触发器
        if src_name in ['gsm8k', 'mathqa', 'svamp', 'mathqa-numeric'] and engine == 'python':
            # 如果源名称在指定列表中且引擎为 python
            text += f'def solution():\n    """{question}"""\n'
            # 在文本末尾添加 Python 函数定义和问题注释

        source_text_res = tokenizer.encode_plus(text, max_length=max_length, truncation=True, add_special_tokens=False)
        # 使用分词器对文本进行编码，设置最大长度、截断和不添加特殊标记
        features["input_ids"].append(source_text_res["input_ids"])
        # 将输入 ID 添加到特征字典
        features["attention_mask"].append(source_text_res["attention_mask"])
        # 将注意力掩码添加到特征字典
        features["question"].append(question)
        # 将问题添加到特征字典
        features["answer_value"].append(examples["answer_value"][idx])
        # 将对应的答案值添加到特征字典
        features["answer_cot"].append(None if "answer_cot" not in examples else examples["answer_cot"][idx])
        # 如果示例中没有答案 COT，则添加 None，否则添加对应的答案 COT
        features['item_id'].append(examples['item_id'][idx])
        # 将项 ID 添加到特征字典

    return features
    # 返回特征字典

def collate_fn(features, tokenizer):
    # 定义 collate_fn 函数，接受特征列表和分词器作为参数
    batch = {"input_ids": [], "attention_mask": []}
    # 初始化批次字典，包含输入 ID 和注意力掩码
    max_input_length = max(len(x["input_ids"]) for x in features)
    # 计算特征中输入 ID 的最大长度
    for feature in features:
        # 遍历每个特征
        input_ids = feature["input_ids"]
        # 获取输入 ID
        attention_mask = feature["attention_mask"]
        # 获取注意力掩码
        input_ids = [tokenizer.pad_token_id] * (max_input_length - len(input_ids)) + input_ids
        # 在输入 ID 前面填充填充标记 ID
        attention_mask = [0] * (max_input_length - len(attention_mask)) + attention_mask
        # 在注意力掩码前面填充 0
        batch["input_ids"].append(input_ids)
        # 将填充后的输入 ID 添加到批次字典
        batch["attention_mask"].append(attention_mask)
        # 将填充后的注意力掩码添加到批次字典

    batch["input_ids"] = torch.tensor(batch["input_ids"])
    # 将输入 ID 转换为张量
    batch["attention_mask"] = torch.tensor(batch["attention_mask"])
    # 将注意力掩码转换为张量
    return batch
    # 返回批次字典

def main(args):
    # 定义主函数，接受参数
    set_seed(42)
    # 设置随机种子为 42，确保可重复性
    if torch.distributed.get_rank() == 0 and args['wandb_log']:
        # 如果当前进程是主进程并且启用了 wandb 日志
        wandb.init(project=args['wandb_project'], name=args['wandb_run_name'])
        # 初始化 wandb 项目和运行名称
        wandb.config.update(args)
        # 更新 wandb 配置

    # Init parameter
    model_name = args['model_name'] 
    # 获取模型名称
    input_path = args['input_path']
    # 获取输入文件路径
    save_dir = args['save_dir']
    # 获取保存目录
    engine = args['engine']
    # 获取引擎类型
    batch_size = args['batch_size']
    # 获取批次大小
    max_length = args['max_length']
    # 获取最大长度
    num_return_sequences = args['num_return_sequences']
    # 获取返回序列的数量
    temperature = args['temperature']
    # 获取温度参数
    do_sample = args['do_sample']
    # 获取是否进行采样的参数

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # 从预训练模型加载分词器
    tokenizer.pad_token_id = 1
    # 设置填充标记的 ID
    tokenizer.eos_token_id = 2
    # 设置结束标记的 ID

    # loading training data
    raw_dataset = Dataset.from_list(json.load(open(input_path,'r')))
    # 从输入路径加载原始数据集
    accelerator.print('Raw data:', raw_dataset)
    # 打印原始数据集
    src_name = raw_dataset[0]['item_id'].split('_')[0]  # e.g., gsm8k_0, gsm8k_1, gsm8k_2, ...
    # 获取源名称，例如 gsm8k
    setup_cot(src_name)
    # 设置 COT 相关的文本
    accelerator.print('Using instruction:', instruction)
    # 打印使用的指令
    accelerator.print('Using cot_trigger:', cot_trigger)
    # 打印使用的 COT 触发器
    accelerator.print('Using answer_trigger:', answer_trigger)
    # 打印使用的答案触发器
    tokenized_dataset = raw_dataset.map(
        tokenize_fn, fn_kwargs={'tokenizer': tokenizer, 'max_length': max_length, 'src_name': src_name, 'engine': engine},
        batched=True, remove_columns=raw_dataset.column_names, load_from_cache_file=False, num_proc=8
    )
    # 对原始数据集进行映射，使用 tokenize_fn 进行分词处理

    valid_dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    # 创建有效数据加载器

    # loading model
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    # 创建保存目录
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # 从预训练模型加载语言模型，并设置为低内存使用模式
    model = model.eval()
    # 将模型设置为评估模式
    inf_config = {
        "replace_with_kernel_inject": False,
        "dtype": torch.bfloat16,
        "enable_cuda_graph": False,
        "tensor_parallel": {"tp_size": 8},
        'max_out_tokens': 1024,
        'min_out_tokens': 1,
    }
    # 设置推理配置
    model = deepspeed.init_inference(model=model, config=inf_config)
    # 初始化 DeepSpeed 推理
    all_results = []
    # 初始化所有结果列表
    acc = 0
    # 初始化准确率计数器
    for b_idx, batch in tqdm(enumerate(valid_dataloader), desc="Generating", total=len(valid_dataloader)):
        # 遍历有效数据加载器中的每个批次
        with torch.no_grad():
            # 在不计算梯度的情况下执行
            # model.module.reset_cache()
            # model.module.reset_cache()  # 清除缓存（注释掉的代码）
            outputs = model.module.generate(batch["input_ids"].to(torch.cuda.current_device()),
                                    attention_mask=batch["attention_mask"].to(torch.cuda.current_device()),
                                    do_sample=do_sample, 
                                    max_length=max_length,
                                    num_return_sequences=num_return_sequences,
                                    use_cache=True,
                                    temperature=temperature,
                                    pad_token_id=tokenizer.pad_token_id,
                                    eos_token_id=tokenizer.eos_token_id) ## batch_size x num_return_sequence, sequence length
            # 使用模型生成输出
            cur_batch_size = len(outputs)//num_return_sequences
            # 计算当前批次的大小
            if accelerator.is_main_process:
                # 如果是主进程
                decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                # 解码生成的输出
                cur_idx = len(all_results)
                # 获取当前结果的索引
                for b_item_idx in range(cur_batch_size):
                    # 遍历当前批次中的每个项
                    if cur_idx + b_item_idx >= len(tokenized_dataset):
                        break
                    # 如果索引超出范围，则退出循环
                    obj = tokenized_dataset[cur_idx+b_item_idx]
                    # 获取当前项
                    new_obj = {
                        "item_id": obj["item_id"],
                        # 当前项的 ID
                        "question": obj["question"],
                        # 当前项的问题
                        "answer_value": obj["answer_value"],
                        # 当前项的答案值
                        "answer_cot": obj["answer_cot"],
                        # 当前项的答案 COT
                        "predictions": [],
                        # 初始化预测列表
                    }
                    execute_fn = post_process_answer_cot_fn_mapper[(args['engine'], src_name)]
                    # 获取后处理函数
                    target_value = post_process_final_answer_fn_mapper[src_name](obj['answer_value'])
                    # 处理目标值
                    prediction_cots = [pred.strip().split(cot_trigger)[-1] for pred in decoded_output[b_item_idx*num_return_sequences: (b_item_idx+1)*num_return_sequences]]
                    # 提取预测的 COT 部分

                    # Save tmp file to debug
                    write_data(f"{save_dir}.tmp", {**new_obj, 'prediction_cots': prediction_cots})
                    # 将临时文件保存以调试

                    answer_counter = Counter()
                    # 初始化答案计数器
                    for i, prediction_value in enumerate(execute_fn(prediction_cots)):
                        # 遍历预测值并执行后处理
                        if src_name == 'mathqa':
                            if len(prediction_value) != 1:
                                prediction_value = None
                        # 对于 mathqa，确保预测值的长度为 1
                        correctness = compare_answer_fn_mapper[src_name](prediction_value, target_value) if prediction_value is not None else False
                        # 比较预测值和目标值，判断是否正确
                        new_obj["predictions"].append({
                            'completion': prediction_cots[i],
                            "solving_res": prediction_value,
                            "correctness": correctness,
                        })
                        # 将预测结果添加到新对象中
                        if prediction_value is not None:
                            answer_counter[prediction_value] += 1
                            # 更新答案计数器

                    voting_answer = answer_counter.most_common(1)[0][0] if answer_counter else None
                    # 获取投票结果
                    correctness = compare_answer_fn_mapper[src_name](voting_answer, target_value) if voting_answer is not None else False
                    # 比较投票结果与目标值，判断是否正确
                    acc += correctness
                    # 累加正确值计数
                    new_obj["most_voted_answer"] = voting_answer
                    # 将投票结果添加到新对象中
                    new_obj["is_correct"] = correctness
                    # 将正确标志添加到新对象中
                    all_results.append(new_obj)
                    # 将新对象添加到所有结果列表中
                    
                write_data(f"{save_dir}.json", all_results)
                # 将所有结果写入 JSON 文件
                print(f"current_acc: {acc} / {len(all_results)} = {acc/len(all_results)* 100}%")
                # 打印当前准确率
                if accelerator.is_main_process and args['wandb_log']:
                    wandb.log({'acc': acc/len(all_results)* 100}, step=b_idx)
                    # 记录准确率到 wandb

    if accelerator.is_main_process:
        # 如果是主进程
        all_results = all_results[:len(tokenized_dataset)]
        # 确保结果的长度与数据集相同
        write_data(f"{save_dir}.json", all_results)
        # 将所有结果写入 JSON 文件
        print(f"current_acc: {acc} / {len(all_results)} = {acc/len(all_results)* 100}%")
        # 打印当前准确率
        if accelerator.is_main_process and args['wandb_log']:
            wandb.log({'acc': acc/len(all_results)* 100}, step=b_idx+1)
            # 记录准确率到 wandb


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
        model_name: str
        # 模型名称
        input_path: str
        # 输入文件路径
        save_dir: str
        # 保存目录
        engine: str
        # 引擎类型
        batch_size: int = field(default=2)
        # 批次大小，默认值为 2
        max_length: int = field(default=1024)
        # 最大长度，默认值为 1024
        num_return_sequences: int = field(default=1)
        # 返回序列的数量，默认值为 1
        temperature: float = field(default=1.0)
        # 温度参数，默认值为 1.0
        do_sample: bool = field(default=False)
        # 是否进行采样，默认值为 False
        # wandb stuff
        wandb_log: bool = field(default=False)
        # 是否启用 wandb 日志，默认值为 False
        wandb_project: str = field(default='tmp_anvfupsadfn')
        # wandb 项目名称，默认值为 'tmp_anvfupsadfn'
        wandb_run_name: str = field(default='default_run_name')
        # wandb 运行名称，默认值为 'default_run_name'

    parser = HfArgumentParser(Arguments)
    # 创建参数解析器
    (args,) = parser.parse_args_into_dataclasses()
    # 解析命令行参数并转换为数据类
    args = asdict(args)
    # 将数据类转换为字典
    for k, v in args.items():
        # 遍历字典中的每个键值对
        if v in [NONE_INT, NONE_STR]:
            # 如果值为无效整数或无效字符串
            args[k] = None
            # 将无效值替换为 None

    print(args)
    # 打印参数字典
    accelerator = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))]) 
    # 创建加速器实例，设置超时时间为 18000 秒（5 小时）
    main(args)
    # 调用主函数