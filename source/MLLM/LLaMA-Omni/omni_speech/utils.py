# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import sys
import torch
import logging
import logging.handlers
import transformers

from omni_speech.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

# handler = None

# 全局变量，用于存储日志文件处理器
handler = None
# 日志文件目录
LOGDIR = 'logs'

def build_logger(logger_name, logger_filename):
    """
    构建日志记录器

    参数:
        logger_name(str): 日志记录器的名称
        logger_filename(str): 日志文件的名称

    返回:
        logger: 配置好的日志记录器实例
    """
    global handler

    # 配置日志消息格式
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 设置根处理器的格式
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # 重定向标准输出和错误输出到日志记录器
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # 获取日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 为所有日志记录器添加文件处理器
    if handler is None:
        # 确保日志目录存在
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True, encoding='UTF-8')
        handler.setFormatter(formatter)

        # 为所有现有日志记录器添加文件处理器
        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


# class StreamToLogger(object):
#     """
#     Fake file-like stream object that redirects writes to a logger instance.
#     """
#     def __init__(self, logger, log_level=logging.INFO):
#         self.terminal = sys.stdout
#         self.logger = logger
#         self.log_level = log_level
#         self.linebuf = ''
#
#     def __getattr__(self, attr):
#         return getattr(self.terminal, attr)
#
#     def write(self, buf):
#         temp_linebuf = self.linebuf + buf
#         self.linebuf = ''
#         for line in temp_linebuf.splitlines(True):
#             # From the io.TextIOWrapper docs:
#             #   On output, if newline is None, any '\n' characters written
#             #   are translated to the system default line separator.
#             # By default sys.stdout.write() expects '\n' newlines and then
#             # translates them so this is still cross platform.
#             if line[-1] == '\n':
#                 self.logger.log(self.log_level, line.rstrip())
#             else:
#                 self.linebuf += line
#
#     def flush(self):
#         if self.linebuf != '':
#             self.logger.log(self.log_level, self.linebuf.rstrip())
#         self.linebuf = ''


class StreamToLogger(object):
    """
    StreamToLogger 类提供了一个假的文件流对象，它将写入操作重定向到一个日志记录器实例。

    该类的主要用途是允许将打印输出（或其他任何流输出）透明地重定向到日志，
    而无需更改现有代码逻辑。这对于在不需要直接输出到控制台，
    或者希望将输出与日志系统整合时非常有用。

    Attributes:
        terminal: 原始的流对象（如 sys.stdout）。
        logger: 用于记录日志的日志记录器实例。
        log_level: 默认的日志记录级别。
        linebuf: 缓冲区，用于临时存储写入操作直到实际日志记录发生。
    """

    def __init__(self, logger, log_level=logging.INFO):
        """
        初始化 StreamToLogger 实例。

        参数:
            logger: 日志记录器实例。
            log_level: 默认的日志记录级别，默认为 logging.INFO。
        """
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        """
        代理到终端流的属性获取。

        这个方法使得 StreamToLogger 对象可以透明地使用它所代理的流对象的属性，
        从而保持与原始流对象的兼容性。

        参数:
            attr: 要获取的属性名。

        返回:
            从终端流对象获取的属性值。
        """
        return getattr(self.terminal, attr)

    def write(self, buf):
        """
        写入缓冲区并根据内容记录日志。

        这个方法接收字符串数据，将其添加到内部缓冲区，并根据换行符来决定何时记录一行日志。
        它允许将打印输出或其他流输出重定向为日志记录，同时保持对不同平台换行符的兼容性。

        参数:
            buf: 要写入的字符串数据。
        """
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # 从 io.TextIOWrapper 文档：
            #   在输出时，如果 newline 为 None，任何写入的 '\n' 字符都会被翻译成系统默认的行分隔符。
            # 默认情况下 sys.stdout.write() 期望 '\n' 换行符，然后将其翻译，因此这仍然是跨平台的。
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        """
        刷新缓冲区并记录剩余内容为日志。

        这个方法用于强制清空内部缓冲区，并将任何剩余的内容记录为日志条目。
        它确保在缓冲区中有未记录的输出时，调用 flush() 方法可以触发日志记录。
        """
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''

def maybe_zero_3(param, ignore_status=False, name=None):
    """
    根据参数是否使用了ZeRO优化，将其从模型并行GPU中聚合到CPU上。

    参数:
        param (torch.Tensor): 需要处理的参数。
        ignore_status (bool, 可选): 是否忽略参数的状态。默认为False。
        name (str, 可选): 参数的名称，用于日志记录。默认为None。

    返回:
        torch.Tensor: 从GPU聚合到CPU并脱离计算图的参数副本。
    """
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    """
    根据指定的bias策略，从命名参数中获取PEFT模型的状态字典。

    参数:
        named_params (Iterator[Tuple[str, torch.Tensor]]): 模型的命名参数迭代器。
        bias (str): 要包含的bias策略，可以是'none'、'all'或'lora_only'。

    返回:
        Dict[str, torch.Tensor]: 包含符合条件的参数的状态字典。

    提示:
        1. 'none'表示只返回包含'lora_'的参数。
        2. 'all'表示返回所有包含'lora_'或'bias'的参数。
        3. 'lora_only'表示只返回与'lora_'相关的bias参数。
    """
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """
    获取PEFT模型的状态字典，排除LoRA相关的参数，并且根据需要只包含需要梯度的参数。
    对结果应用maybe_zero_3函数，并转移到CPU上。

    参数:
    named_params (iterable): 模型的命名参数迭代器。
    require_grad_only (bool, optional): 是否只包含需要梯度的参数。默认为True。

    返回:
    dict: 处理后的状态字典，不包含LoRA参数，并且根据require_grad_only只包含需要梯度的参数。
    """
    # 过滤掉包含LoRA的参数
    to_return = {k: t for k, t in named_params if "lora_" not in k}

    # 如果require_grad_only为True，则进一步过滤，只保留需要梯度的参数
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}

    # 对结果字典应用maybe_zero_3函数，并转移到CPU上
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}

    return to_return


def get_speech_projector_state_maybe_zero_3(named_params, keys_to_match):
    """
    获取语音投影模型的状态字典，只包含与指定关键字匹配的参数，并对结果应用maybe_zero_3函数，
    转移到CPU上。

    参数:
    named_params (iterable): 模型的命名参数迭代器。
    keys_to_match (list): 用于匹配参数键的关键字列表。

    返回:
    dict: 处理后的状态字典，只包含与关键字匹配的参数，并且经过maybe_zero_3处理，转移到CPU上。
    """
    # 过滤参数，只保留包含任意一个关键字的参数
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}

    # 对结果字典应用maybe_zero_3函数，并转移到CPU上
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}

    return to_return



def find_all_linear_names(model):
    """
    遍历模型的所有模块，找出其中的线性层（全连接层）名称。

    Args:
        model: PyTorch模型，用于遍历其所有模块。

    Returns:
        list: 模型中所有线性层的名称列表。

    """
    # 定义线性层类
    cls = torch.nn.Linear
    # 初始化存储线性层名称的集合
    lora_module_names = set()
    # 定义包含“speech”关键词的列表，用于跳过相关模块
    speech_keywords = ['speech_projector', 'speech_encoder']
    # 遍历模型的所有命名模块
    for name, module in model.named_modules():
        # 如果模块名称包含“speech”关键词，则跳过
        if any(speech_keyword in name for speech_keyword in speech_keywords):
            continue
        # 如果模块是线性层，则处理其名称
        if isinstance(module, cls):
            names = name.split('.')
            # 根据名称的组成部分，添加到集合中
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # 如果集合中包含'lm_head'，则移除它（16-bit精度下的需要）
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    # 返回线性层名称的列表
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """
    安全地保存transformers.Trainer训练得到的模型。

    Args:
        trainer (transformers.Trainer): Trainer对象，用于保存模型。
        output_dir (str): 保存模型的输出目录。

    Returns:
        None

    """
    # Collects the state dict and dump to disk.
    """收集模型的状态字典并保存到磁盘。"""

    # 如果配置了训练语音投影器，则只保存投影器
    if getattr(trainer.args, "tune_speech_projector", False):
        # 只保存projector
        keys_to_match = ['speech_projector']
        # 如果使用了im_start_end，则也保存相应的embedding层
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        # 获取要保存的权重
        weight_to_save = get_speech_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        # 保存配置
        trainer.model.config.save_pretrained(output_dir)

        # 处理保存路径
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        # 只在主进程保存
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                # 如果是checkpoint目录，则保存到speech_projector子目录
                speech_projector_folder = os.path.join(parent_folder, "speech_projector")
                os.makedirs(speech_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(speech_projector_folder, f'{current_folder}.bin'))
            else:
                # 否则直接保存到output_dir
                torch.save(weight_to_save, os.path.join(output_dir, f'speech_projector.bin'))
        return

    # 如果使用了DeepSpeed，则同步GPU后保存模型
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    # 获取模型的状态字典
    state_dict = trainer.model.state_dict()
    # 如果需要保存，则将状态字典转移到CPU并保存
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict  # 释放显存
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def lengths_to_padding_mask(lens):
    """
    根据序列长度生成padding mask。

    参数:
    lens (torch.Tensor): 包含各序列长度的张量。

    返回:
    torch.BoolTensor: bool类型的mask，其中True表示padding位置。
    """
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask



def lengths_to_mask(lens):
    """
    根据序列长度生成非padding mask。

    参数:
    lens (torch.Tensor): 包含各序列长度的张量。

    返回:
    torch.BoolTensor: bool类型的mask，其中False表示padding位置。
    """
    return ~lengths_to_padding_mask(lens)


def disable_torch_init():
    """
    禁用冗余的PyTorch默认初始化，以加速模型创建。
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)




def get_model_name_from_path(model_path):
    """
    从模型路径中提取模型名称。

    参数:
    model_path (str): 模型的路径。

    返回:
    str: 提取到的模型名称。
    """
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def violates_moderation(text):
    """
    检查文本是否违反OpenAI内容审核API。

    参数:
    text (str): 待审核的文本。

    返回:
    bool: 如果文本违反内容审核API，则返回True，否则返回False。
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    """
    以易读格式打印信号量信息。

    参数:
    semaphore (threading.Semaphore): 待打印的信号量对象。

    返回:
    str: 信号量的信息字符串。
    """
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"