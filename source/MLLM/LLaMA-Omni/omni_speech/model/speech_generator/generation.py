import copy
import torch
import inspect
import warnings
import numpy as np
import torch.nn as nn
from typing import Optional, Union, List, Callable
import torch.distributed as dist

from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerationConfig,
    GenerationMode,
    LogitsProcessorList,
    StoppingCriteriaList,
    GenerateOutput, 
    GenerationMixin,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
    is_deepspeed_zero3_enabled,
    is_torchdynamo_compiling,
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
    is_hqq_available,
    QuantizedCacheConfig,
    is_quanto_available,
    DynamicCache,
    EncoderDecoderCache,
    logging
)
# from transformers.generation.stopping_criteria import validate_stopping_criteria

logger = logging.get_logger(__name__)


class GenerationWithCTC(GenerationMixin):

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        streamer_unit: Optional["BaseStreamer"] = None,
        streaming_unit_gen = False,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        """
        生成文本的主要函数。

        参数:
            inputs: 输入张量，用于生成文本。
            generation_config: 生成配置对象。
            logits_processor: 用于处理logits的处理器列表。
            stopping_criteria: 停止生成的条件列表。
            prefix_allowed_tokens_fn: 允许的前缀token函数。
            synced_gpus: 是否同步GPU。
            assistant_model: 辅助模型。
            streamer: 流处理器。
            streamer_unit: 流处理器单元。
            streaming_unit_gen: 是否使用流式生成。
            negative_prompt_ids: 负面提示ID。
            negative_prompt_attention_mask: 负面提示注意力掩码。
            **kwargs: 其他关键字参数。

        返回:
            GenerateOutput或torch.LongTensor: 生成的文本输出。
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        # 1. 处理`generation_config`和可能更新它的kwargs，并验证`.generate()`调用
        # 验证模型类是否有效
        self._validate_model_class()
        # 提取tokenizer参数，主要用于停止条件
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria # 首先提取这个，我们只用它来停止条件
        # 准备生成配置和模型参数，并进行验证
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())
        # 验证辅助模型
        self._validate_assistant(assistant_model)

        # 2. Set generation parameters if not already defined
        # 2. 如果未定义，则设置生成参数
        # 根据是否启用DeepSpeed Zero3和分布式训练环境，决定同步GPU的使用
        if synced_gpus is None:
            if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
                synced_gpus = True
            else:
                synced_gpus = False
        # 初始化logits处理器和停止条件
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        # 检查模型输入是否需要attention_mask
        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        # 准备模型输入张量和相关参数
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        # 获取批量大小
        batch_size = inputs_tensor.shape[0]
        # 获取设备信息
        device = inputs_tensor.device
        # 准备特殊token
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # 对于仅解码器模型，必须使用左填充进行批量生成，否则会导致生成结果不正确
        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder and not is_torchdynamo_compiling():
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            # 检查输入序列是否使用了右填充，并发出警告
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        # 如果模型不是编码器-解码器，并且模型输入名称是"inputs_embeds"，则设置use_cache为True
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            model_kwargs["use_cache"] = True
        else:
            # 否则，使用generation_config中的use_cache值
            model_kwargs["use_cache"] = generation_config.use_cache

        # 如果kwargs中没有attention_mask，但是需要attention_mask并且模型接受attention_mask
        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            # 准备attention_mask
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor
            )

        # 如果模型是编码器-解码器，并且model_kwargs中没有encoder_outputs
        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            # 准备编码器-解码器的kwargs
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        # 准备用于自回归生成的input_ids
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            # 如果模型输入名称是"input_ids"，则使用inputs_tensor，否则从model_kwargs中移除"input_ids"
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        # 如果启用了token_healing，则修复tokens
        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        # 如果streamer不为None，则将input_ids放入streamer
        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        # 获取输入ID的长度
        input_ids_length = input_ids.shape[-1]
        # 检查是否使用默认的最大长度和最小长度
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        # 准备生成长度配置
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )
        # 默认不使用动态缓存
        use_dynamic_cache_by_default = False
        # 根据模型类名设置缓存名称
        if "mamba" in self.__class__.__name__.lower():
            cache_name = "cache_params"
        else:
            cache_name = "past_key_values"
        # 检查是否同时传递了cache_implementation和cache对象，这是不支持的情况
        if generation_config.cache_implementation is not None and (model_kwargs.get(cache_name) is not None):
            raise ValueError(
                f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                "Cache object) is unsupported. Please use only one of the two."
            )
        elif generation_config.cache_implementation is not None:
            # 如果指定了cache_implementation，则根据实现类型设置缓存
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                # 检查是否支持静态缓存
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                # 获取并设置缓存
                model_kwargs[cache_name] = self._get_cache(
                    generation_config.cache_implementation,
                    getattr(generation_config, "num_beams", 1) * batch_size,
                    generation_config.max_length,
                    model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                # 检查是否支持量化缓存
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue."
                    )
                # 设置量化缓存的配置
                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                # 获取量化缓存的类
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]
                # 检查是否安装了相应的量化后端库
                if cache_config.backend == "quanto" and not is_quanto_available():
                    raise ImportError(
                        "You need to install `quanto` in order to use KV cache quantization with quanto backend. "
                        "Please install it via  with `pip install quanto`"
                    )
                elif cache_config.backend == "HQQ" and not is_hqq_available():
                    raise ImportError(
                        "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                        "Please install it via  with `pip install hqq`"
                    )
                # 设置量化缓存
                model_kwargs[cache_name] = cache_class(cache_config)
        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        # 使用默认的DynamicCache()实例。这将避免来回从遗留格式复制缓存，从而使用更多的内存
        elif generation_config.cache_implementation is None and self._supports_default_dynamic_cache():
            past = model_kwargs.get(cache_name, None)
            requires_cross_attention_cache = (
                self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
            )
            # 如果past为None，则根据是否需要跨注意力缓存来初始化缓存
            if past is None:
                model_kwargs[cache_name] = (
                    DynamicCache()
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache(DynamicCache(), DynamicCache())
                )
                use_dynamic_cache_by_default = True

            # 如果past是tuple类型，则从遗留缓存中创建相应的缓存实例
            elif isinstance(past, tuple):
                model_kwargs[cache_name] = (
                    DynamicCache.from_legacy_cache(past)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(past)
                )
                use_dynamic_cache_by_default = True

        # 验证生成长度是否符合配置要求
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. determine generation mode
        # 7. 确定生成模式
        generation_mode = generation_config.get_generation_mode(assistant_model)

        # 如果使用了streamer或streamer_unit，并且num_beams大于1，则抛出异常
        if (streamer is not None or streamer_unit is not None) and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        # 如果输入的input_ids设备类型与模型的设备类型不同，则发出警告
        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 8. prepare distribution pre_processing samplers
        # 8. 准备 logits 处理器
        # 该函数用于获取一个 logits 处理器，它将根据提供的参数来处理生成过程中的 logits
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,                                # 生成配置
            input_ids_seq_length=input_ids_length,                              # 输入 ID 序列长度
            encoder_input_ids=inputs_tensor,                                    # 编码器输入 ID 张量
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,                  # 前缀允许的 tokens 函数
            logits_processor=logits_processor,                                  # logits 处理器
            device=inputs_tensor.device,                                        # 设备
            model_kwargs=model_kwargs,                                          # 模型参数
            negative_prompt_ids=negative_prompt_ids,                            # 负面提示 ID
            negative_prompt_attention_mask=negative_prompt_attention_mask,      # 负面提示注意力掩码
        )

        # 9. prepare stopping criteria
        # 9. 准备停止标准
        # 该函数用于获取停止标准，它将根据提供的参数来决定何时停止生成过程
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config,                                # 生成配置
            stopping_criteria=stopping_criteria,                                # 停止标准
            tokenizer=tokenizer,                                                # 分词器
            **kwargs                                                            # 其他关键字参数
        )
        # 10. go into different generation modes
        # 10. 进入不同的生成模式
        # 判断生成模式是否为采样或贪婪搜索
        if generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. prepare logits warper
            # 11. 准备 logits 调整器
            # 如果配置中启用了采样，则准备 logits 调整器，否则为 None
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample
                else None
            )

            # 12. expand input_ids with `num_return_sequences` additional sequences per batch
            # 12. 扩展 input_ids，每个批次增加 `num_return_sequences` 个序列
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,                                            # 输入 ID
                expand_size=generation_config.num_return_sequences,             # 扩展大小
                is_encoder_decoder=self.config.is_encoder_decoder,              # 是否为编码器-解码器模型
                **model_kwargs,                                                 # 其他模型参数
            )

            # 13. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            # 13. 运行采样（当 `generation_config.do_sample=False` 时退化为贪婪搜索）
            # 如果是流式单元生成，则调用 _sample_streaming_unit 方法
            if streaming_unit_gen:
                return self._sample_streaming_unit(
                    input_ids,                                                  # 输入 ID
                    logits_processor=prepared_logits_processor,                 # 准备好的 logits 处理器
                    logits_warper=prepared_logits_warper,                       # 准备好的 logits 调整器
                    stopping_criteria=prepared_stopping_criteria,               # 准备好的停止标准
                    generation_config=generation_config,                        # 生成配置
                    synced_gpus=synced_gpus,                                    # 同步 GPU
                    streamer=streamer,                                          # 流处理器
                    streamer_unit=streamer_unit,                                # 流单元
                    **model_kwargs,                                             # 其他模型参数
                )
            else:
                # 否则调用 _sample 方法
                return self._sample(
                    input_ids,                                                  # 输入 ID
                    logits_processor=prepared_logits_processor,                 # 准备好的 logits 处理器
                    logits_warper=prepared_logits_warper,                       # 准备好的 logits 调整器
                    stopping_criteria=prepared_stopping_criteria,               # 准备好的停止标准
                    generation_config=generation_config,                        # 生成配置
                    synced_gpus=synced_gpus,                                    # 同步 GPU
                    streamer=streamer,                                          # 流处理器
                    **model_kwargs,                                             # 其他模型参数
                )
        else:
            raise NotImplementedError

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        streamer: Optional["BaseStreamer"],
        logits_warper: Optional[LogitsProcessorList],
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        """
        生成模型的样本。

        Args:
            input_ids (torch.LongTensor): 输入的token ID序列。
            logits_processor (LogitsProcessorList): 用于处理模型输出的logits的处理器列表。
            stopping_criteria (StoppingCriteriaList): 停止生成的条件列表。
            generation_config (GenerationConfig): 生成配置。
            synced_gpus (bool): 是否同步GPU。
            streamer (Optional["BaseStreamer"]): 用于流式输出结果的streamer。
            logits_warper (Optional[LogitsProcessorList]): 用于对logits进行扭曲处理的处理器列表。
            **model_kwargs: 其他传递给模型的参数。

        Returns:
            Union[GenerateNonBeamOutput, torch.LongTensor]: 生成的token ID序列或包含更多信息的结构化输出。
        """
        # 初始化值
        # init values
        # 从配置中获取填充token的ID
        pad_token_id = generation_config._pad_token_tensor
        # 获取是否输出注意力矩阵的配置
        output_attentions = generation_config.output_attentions
        # 获取是否输出隐藏状态的配置
        output_hidden_states = generation_config.output_hidden_states
        # 获取是否输出分数的配置
        output_scores = generation_config.output_scores
        # 获取是否输出logits的配置
        output_logits = generation_config.output_logits
        # 获取在生成过程中是否返回字典的配置
        return_dict_in_generate = generation_config.return_dict_in_generate
        # 检查是否有以EOS标记为停止标准的条件
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        # 获取是否进行采样的配置
        do_sample = generation_config.do_sample

        # 检查是否需要采样以及logits_warper是否为LogitsProcessorList实例
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        # 初始化各种输出控制变量，如果 return_dict_in_generate 为 True 并且相应的输出标志为 True，则初始化为空元组，否则为 None
        # 如果生成过程中需要返回分数，并且配置了输出分数，则初始化一个空元组来稍后填充分数数据。
        scores = () if (return_dict_in_generate and output_scores) else None
        # raw_logits 通常指的是模型输出的原始对数概率，它通常是一个二维张量，其中第一维是序列长度，第二维是token ID的个数。
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        # 初始化解码器的注意力权重
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        # 初始化解码器的交叉注意力权重
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        # 初始化解码器的隐藏状态
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # 如果模型是编码器-解码器结构，获取编码器的注意力权重和隐藏状态
        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # 跟踪哪些序列已经完成
        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # 当存在未完成的序列时，继续循环
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            # 准备模型输入
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            # 准备可变的输出控制（注意：某些模型可能不接受所有输出控制）
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            # 前向传播以获取下一个 token
            outputs = self(**model_inputs, return_dict=True)

            # 如果模型运行在同步 GPU 上且当前 peer 已完成，则跳过不必要的计算
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need # 不浪费资源运行我们不需要的代码

            # 克隆 next_token_logits 以避免保持对可能非常大的 outputs.logits 的引用
            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()

            # pre-process distribution
            # 预处理分布
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            # 当需要时存储分数、注意力权重和隐藏状态
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            # 选择下一个 token
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            
            # finished sentences should have their next token be a padding token
            # 已完成的句子的下一个 token 应该是填充 token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            # 更新生成的 ids、模型输入和下一步的长度
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            # 更新未完成序列的状态
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # 删除 outputs.logits 的引用，以避免内存泄漏
            del outputs

        # 如果使用了 streamer，则结束 streamer
        if streamer is not None:
            streamer.end()

        # 如果return_dict_in_generate为True，则根据配置返回不同的输出对象
        if return_dict_in_generate:
            # 如果配置是编码器-解码器结构
            if self.config.is_encoder_decoder:
                # 返回编码器-解码器结构的生成输出对象
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,  # 生成的序列
                    scores=scores,  # 生成序列的分数
                    logits=raw_logits,  # 原始logits
                    encoder_attentions=encoder_attentions,  # 编码器的注意力权重
                    encoder_hidden_states=encoder_hidden_states,  # 编码器的隐藏状态
                    decoder_attentions=decoder_attentions,  # 解码器的注意力权重
                    cross_attentions=cross_attentions,  # 跨注意力权重
                    decoder_hidden_states=decoder_hidden_states,  # 解码器的隐藏状态
                    past_key_values=model_kwargs.get("past_key_values"),  # 过去的键值对
                )
            else:
                # 如果配置只有解码器结构，返回解码器结构的生成输出对象
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,  # 生成的序列
                    scores=scores,  # 生成序列的分数
                    logits=raw_logits,  # 原始logits
                    attentions=decoder_attentions,  # 解码器的注意力权重
                    hidden_states=decoder_hidden_states,  # 解码器的隐藏状态
                    past_key_values=model_kwargs.get("past_key_values"),  # 过去的键值对
                )
        else:
            # 如果return_dict_in_generate为False，直接返回生成的序列
            return input_ids

    def _sample_streaming_unit(
        self,
        input_ids: torch.LongTensor,                        # 输入的token id序列
        logits_processor: LogitsProcessorList,              # 处理logits的处理器列表
        stopping_criteria: StoppingCriteriaList,            # 停止生成的条件列表
        generation_config: GenerationConfig,                # 生成配置
        synced_gpus: bool,                                  # 是否同步GPU
        streamer: Optional["BaseStreamer"],                 # 流处理器
        streamer_unit: Optional["BaseStreamer"],            # 流处理器单元
        logits_warper: Optional[LogitsProcessorList],       # 处理logits的包装器
        **model_kwargs,                                     # 其他模型参数
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:    # 返回生成的结果或token id序列
        # init values
        # 初始化各种控制变量和状态
        # 从 generation_config 中获取配置参数
        pad_token_id = generation_config._pad_token_tensor                  # 获取填充token的ID
        output_attentions = generation_config.output_attentions             # 是否输出注意力矩阵
        output_hidden_states = generation_config.output_hidden_states       # 是否输出隐藏状态
        output_scores = generation_config.output_scores                     # 是否输出分数
        output_logits = generation_config.output_logits                     # 是否输出logits
        return_dict_in_generate = generation_config.return_dict_in_generate # 生成时是否返回字典
        # 检查是否有基于EOS的停止标准
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        # 获取采样设置
        do_sample = generation_config.do_sample

        # 如果设置了采样，但logits_warper不是LogitsProcessorList实例，则抛出异常
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        # 初始化变量，根据条件决定是否返回分数、logits、注意力权重和隐藏状态
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        # 如果模型是编码器-解码器结构，获取编码器的注意力权重和隐藏状态
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        # 跟踪哪些序列已经完成
        # batch_size 为输入ID的批量大小
        batch_size = input_ids.shape[0]
        # 初始化此对等端是否完成标志为False
        this_peer_finished = False
        # 初始化未完成序列张量，大小为批量大小，数据类型为torch.long，设备与输入ID相同
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        # 获取模型初始缓存位置
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        # 初始化生成单元张量为空
        generated_units = torch.tensor([])
        # 当存在未完成序列时，执行循环
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            # 准备模型输入
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # prepare variable output controls (note: some models won't accept all output controls)
            # 准备可变输出控制（注意：某些模型可能不接受所有输出控制）
            model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
            model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

            # forward pass to get next token
            # 前向传播以获取下一个令牌
            outputs = self(**model_inputs, return_dict=True)

            # 如果同步GPU且此对等端已完成，则跳过不必要的代码执行
            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            # 克隆是必要的，以避免保持对可能非常大的outputs.logits的悬空引用（克隆本身始终很小）
            next_token_logits = outputs.logits[:, -1, :].clone()

            # pre-process distribution
            # 预处理分布
            next_token_scores = logits_processor(input_ids, next_token_logits)
            # 如果需要采样，则对分数进行扭曲
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # 当需要时存储分数、注意力和隐藏状态
            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    # 根据配置是编码器-解码器还是仅解码器，添加相应的注意力
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    # 根据配置是编码器-解码器还是仅解码器，添加相应的隐藏状态
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            # 根据是否采用随机采样来决定下一步的生成策略
            if do_sample:
                # 对下一个令牌的分数应用softmax，转换为概率分布
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # 从概率分布中随机采样，得到下一个令牌
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 如果不采用随机采样，则选择分数最高的令牌作为下一个令牌
                next_tokens = torch.argmax(next_token_scores, dim=-1)
            
            # speechgen
            # 语音生成部分
            # 按照特定维度拼接隐藏状态，然后通过语音生成器预测
            hidden_states = torch.cat([decoder_hidden_states[0][-1][:, -1:, :]] + [decoder_hidden_states[i][-1] for i in range(1, len(decoder_hidden_states))], dim=1)
            ctc_pred = self.speech_generator.predict(hidden_states.squeeze(0))
            # 对预测结果进行后处理，得到当前的单位序列
            cur_units = ctc_postprocess(ctc_pred, blank=self.model.config.unit_vocab_size)
            
            # finished sentences should have their next token be a padding token
            # 对已完成的句子，将其下一个令牌设置为填充令牌
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            # 更新生成的ID、模型输入和长度，为下一步生成做准备
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            if streamer_unit is not None:
                for i in range(len(generated_units), len(cur_units)):
                    streamer_unit.put(cur_units[i].unsqueeze(0))
            generated_units = cur_units
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            # 更新未完成序列的标志，并检查停止条件
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # 删除中间结果，释放内存
            del outputs

        # 如果使用了流式传输器，则结束流式传输
        if streamer is not None:
            streamer.end()

        # 根据条件返回相应的生成器输出
        if return_dict_in_generate:
            # 如果模型是编码解码器结构
            if self.config.is_encoder_decoder:
                # 返回编码解码器生成器输出，包含序列、分数、原始逻辑、注意力和隐藏状态等信息
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,  # 生成的序列
                    scores=scores,  # 每个生成序列的分数
                    logits=raw_logits,  # 原始逻辑值
                    encoder_attentions=encoder_attentions,  # 编码器注意力权重
                    encoder_hidden_states=encoder_hidden_states,  # 编码器隐藏状态
                    decoder_attentions=decoder_attentions,  # 解码器注意力权重
                    cross_attentions=cross_attentions,  # 交叉注意力权重
                    decoder_hidden_states=decoder_hidden_states,  # 解码器隐藏状态
                    past_key_values=model_kwargs.get("past_key_values"),  # 之前的键值对
                )
            else:
                # 如果模型不是编码解码器结构，则返回解码器生成器输出
                # 返回解码器生成器输出，包含序列、分数、原始逻辑、注意力和隐藏状态等信息
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,  # 生成的序列
                    scores=scores,  # 每个生成序列的分数
                    logits=raw_logits,  # 原始逻辑值
                    attentions=decoder_attentions,  # 解码器注意力权重
                    hidden_states=decoder_hidden_states,  # 解码器隐藏状态
                    past_key_values=model_kwargs.get("past_key_values"),  # 之前的键值对
                )
        # 如果不需要以字典形式返回生成器输出
        else:
            # 直接返回生成的序列
            return input_ids


def ctc_postprocess(tokens, blank):
    """
    对CTC解码后的令牌进行后处理，去除重复的令牌和空白令牌。

    参数:
    tokens (torch.Tensor): CTC解码后的令牌序列，形状为(1, T)。
    blank (int): 空白令牌的索引。

    返回:
    torch.Tensor: 去除重复和空白令牌后的令牌序列。
    """
    # 将tokens张量中的第一个维度去除并转换为列表
    _toks = tokens.squeeze(0).tolist()

    # 去除连续重复的令牌
    deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]

    # 去除空白令牌
    hyp = torch.tensor([v for v in deduplicated_toks if v != blank])

    # 返回处理后的令牌序列
    return hyp
