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

import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    is_safetensors_available,
)
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead


logger = logging.get_logger(__name__)


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", 
    output_dir: str, 
    safe_serialization: bool
) -> None:
    r"""
    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    修复value head模型检查点的函数，处理三种不同训练场景下的参数保存问题
    """
    # 检查预训练模型类型（需为HuggingFace模型或Peft模型）
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    # 根据安全序列化标志选择加载方式
    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)  # 安全张量格式文件路径
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:  # 安全打开检查点
            state_dict: Dict[str, torch.Tensor] = {key: f.get_tensor(key) for key in f.keys()}  # 加载所有张量
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)  # PyTorch格式文件路径
        state_dict: Dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu")  # 加载检查点

    os.remove(path_to_checkpoint)  # 删除原始检查点文件
    decoder_state_dict, v_head_state_dict = {}, {}  # 初始化解码器和value head参数字典

    # 分离解码器和value head参数
    for name, param in state_dict.items():
        if name.startswith("v_head."):  # 识别value head参数
            v_head_state_dict[name] = param
        else:  # 处理解码器参数（可能包含DeepSpeed Zero3的前缀）
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param  # 移除Zero3前缀

    # 保存解码器部分（使用HuggingFace原生保存方法）
    model.pretrained_model.save_pretrained(
        output_dir, 
        state_dict=decoder_state_dict or None,  # 空字典时传None
        safe_serialization=safe_serialization  # 继承安全序列化设置
    )

    # 保存value head部分（单独保存）
    if safe_serialization:
        save_file(  # 使用safetensors保存
            v_head_state_dict, 
            os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), 
            metadata={"format": "pt"}  # 添加元数据
        )
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))  # 普通PyTorch保存

    logger.info_rank0(f"Value head model saved at: {output_dir}")  # 记录日志


class FixValueHeadModelCallback(TrainerCallback):
    r"""
    A callback for fixing the checkpoint for valuehead models.
    修复value head模型检查点的回调类
    """
    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after a checkpoint save.
        在保存检查点后触发的回调方法
        """
        if args.should_save:  # 检查是否需要保存
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")  # 构建检查点路径
            fix_valuehead_checkpoint(  # 调用修复函数
                model=kwargs.pop("model"), 
                output_dir=output_dir, 
                safe_serialization=args.save_safetensors  # 继承安全序列化设置
            )


class SaveProcessorCallback(TrainerCallback):
    r"""
    A callback for saving the processor.
    保存处理器的回调类（用于多模态模型）
    """
    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor  # 初始化处理器实例

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")  # 检查点路径
            self.processor.save_pretrained(output_dir)  # 保存处理器到检查点目录

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.processor.save_pretrained(args.output_dir)  # 训练结束时保存到最终输出目录


class PissaConvertCallback(TrainerCallback):
    r"""
    A callback for converting the PiSSA adapter to a normal one.
    将PiSSA适配器转换为标准LoRA适配器的回调
    """
    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of training.
        训练开始时保存初始PiSSA适配器
        """
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")  # 初始适配器保存路径
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                # 临时修改初始化配置
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")  # 保存原始配置
                setattr(model.peft_config["default"], "init_lora_weights", True)  # 强制使用标准初始化
                model.save_pretrained(pissa_init_dir, safe_serialization=args.save_safetensors)  # 保存初始适配器
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)  # 恢复原始配置

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")  # 初始适配器路径
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")  # 备份路径
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")  # 转换后路径
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            
            # 四步转换流程
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")  # 获取当前配置
                
                # 步骤1：保存带标准初始化的备份
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                
                # 步骤2：保存转换后的LoRA适配器
                model.save_pretrained(
                    pissa_convert_dir, 
                    safe_serialization=args.save_safetensors, 
                    convert_pissa_to_lora=pissa_init_dir  # 执行PiSSA到LoRA的转换
                )
                
                # 步骤3：重新加载备份适配器
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                
                # 步骤4：清理旧配置（兼容旧版peft）
                if "pissa_init" in model.peft_config.keys():  # 检查旧版适配器名称
                    model.delete_adapter("pissa_init")  # 删除旧适配器
                
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)  # 恢复原始配置


class LogCallback(TrainerCallback):
    r"""
    A callback for logging training and evaluation status.
    训练日志记录回调，处理进度跟踪、资源监控和Web UI集成
    """
    def __init__(self) -> None:
        # 训练进度跟踪
        self.start_time = 0  # 训练开始时间戳
        self.cur_steps = 0  # 当前步数
        self.max_steps = 0  # 最大步数
        self.elapsed_time = ""  # 已用时间
        self.remaining_time = ""  # 剩余时间预估
        self.thread_pool: Optional["ThreadPoolExecutor"] = None  # 异步日志线程池
        
        # 训练状态
        self.aborted = False  # 是否被中止
        self.do_train = False  # 是否处于训练模式
        
        # Web UI集成
        self.webui_mode = os.environ.get("LLAMABOARD_ENABLED", "0").lower() in ["true", "1"]  # 是否启用Web UI
        if self.webui_mode:
            signal.signal(signal.SIGABRT, self._set_abort)  # 注册中止信号处理
            self.logger_handler = logging.LoggerHandler(os.environ.get("LLAMABOARD_WORKDIR"))  # 自定义日志处理器
            logging.add_handler(self.logger_handler)  # 添加日志处理器
            transformers.logging.add_handler(self.logger_handler)  # 集成transformers日志

    # 信号处理函数
    def _set_abort(self, signum, frame) -> None:
        self.aborted = True  # 设置中止标志

    # 重置计时器
    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    # 计算时间统计
    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time  # 已用时间（秒）
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0  # 平均每步耗时
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step  # 预估剩余时间
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))  # 格式化为HH:MM:SS
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    # 异步写入日志文件
    def _write_log(self, output_dir: str, logs: Dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")  # 追加JSON格式日志

    # 创建日志线程池
    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
        self.thread_pool = ThreadPoolExecutor(max_workers=1)  # 单线程池避免竞争

    # 关闭线程池
    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)  # 等待任务完成
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        # 初始化完成后清理旧日志
        if args.should_save and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG)) and args.overwrite_output_dir:
            logger.warning_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))  # 删除旧日志文件

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        # 训练开始时初始化
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)  # 设置最大步数
            self._create_thread_pool(output_dir=args.output_dir)  # 创建日志线程池

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()  # 训练结束关闭线程池

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        # 子步骤结束时检查中止标志
        if self.aborted:
            control.should_epoch_stop = True  # 停止当前epoch
            control.should_training_stop = True  # 停止整个训练

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        # 训练步骤结束时检查中止标志
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        # 日志记录主方法
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)  # 更新计时
        
        # 构建日志字典
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),  # 当前损失
            eval_loss=state.log_history[-1].get("eval_loss"),  # 评估损失
            lr=state.log_history[-1].get("learning_rate"),  # 学习率
            epoch=state.log_history[-1].get("epoch"),  # 当前epoch
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,  # 进度百分比
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )

        # 吞吐量统计
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)  # tokens/秒
            logs["total_tokens"] = state.num_input_tokens_seen  # 总处理tokens

        # VRAM监控
        if os.environ.get("RECORD_VRAM", "0").lower() in ["true", "1"]:
            vram_allocated, vram_reserved = get_peak_memory()  # 获取显存峰值
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)  # 转换为GB
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        # Web UI格式优化
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):  # 扩展指标
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"
            logger.info_rank0("{" + log_str + "}")  # 结构化日志输出

        # 提交异步写入任务
        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)  # 异步写入日志文件

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        """
        预测步骤回调方法，用于评估/预测阶段的进度跟踪
        """
        if self.do_train:  # 检查是否处于训练模式
            return  # 训练模式下不处理预测步骤

        if self.aborted:  # 检查是否收到中止信号
            sys.exit(0)  # 立即退出程序

        if not args.should_save:  # 检查是否需要保存日志
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)  # 获取评估数据加载器
        if has_length(eval_dataloader):  # 检查数据加载器是否有长度
            if self.max_steps == 0:  # 首次进入预测步骤时初始化
                self._reset(max_steps=len(eval_dataloader))  # 根据数据加载器长度重置计时器
                self._create_thread_pool(output_dir=args.output_dir)  # 创建日志线程池

            self._timing(cur_steps=self.cur_steps + 1)  # 更新当前步骤数并计算时间
            
            # 每5步记录一次日志（平衡日志频率和性能）
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,  # 当前已处理样本数
                    total_steps=self.max_steps,  # 总样本数
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,  # 完成百分比
                    elapsed_time=self.elapsed_time,  # 已用时间
                    remaining_time=self.remaining_time,  # 预估剩余时间
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)  # 异步写入日志
