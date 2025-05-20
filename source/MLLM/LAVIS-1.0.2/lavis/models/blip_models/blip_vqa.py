"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch # 导入 PyTorch 库
import torch.nn.functional as F # 导入 PyTorch 的函数式 API，通常用于激活函数、损失函数等
from lavis.common.registry import registry # 从 lavis.common.registry 导入 registry，用于注册模型
from lavis.models.base_model import tile # 从 lavis.models.base_model 导入 tile 函数
from lavis.models.blip_models.blip import BlipBase # 从 lavis.models.blip_models.blip 导入 BlipBase 基类
from lavis.models.blip_models.blip_outputs import ( # 从 lavis.models.blip_models.blip_outputs 导入 BLIP 模型的输出类
    BlipOutput, # BLIP 模型的主要输出类
    BlipIntermediateOutput, # BLIP 模型的中间输出类
)
from lavis.models.med import XBertEncoder, XBertLMHeadDecoder # 从 lavis.models.med 导入 XBertEncoder 和 XBertLMHeadDecoder
from lavis.models.vit import VisionTransformerEncoder # 从 lavis.models.vit 导入 VisionTransformerEncoder


# 使用 registry.register_model 装饰器注册模型，模型名称为 "blip_vqa"
@registry.register_model("blip_vqa")
# 定义 BlipVQA 类，继承自 BlipBase
class BlipVQA(BlipBase):
    """
    BLIP VQA models. # BLIP VQA 模型。

    Supported model types: # 支持的模型类型：
        - base: vqa model initialized with pre-trained BLIP base model on 115M image-text pairs after CapFilt; not fine-tuned. # base: 使用在 CapFilt 后 1.15 亿图像-文本对上预训练的 BLIP base 模型初始化的 VQA 模型；未微调。
        - vqav2: fine-tuned BLIP base model on VQA v2.0 dataset. # vqav2: 在 VQA v2.0 数据集上微调的 BLIP base 模型。

    Usage: # 用法：
        >>> from lavis.models import load_model # 从 lavis.models 导入 load_model
        >>> model = load_model("blip_vqa", "vqav2") # 加载 vqav2 类型的 blip_vqa 模型
        >>> model = load_model("blip_vqa", "okvqa") # 加载 okvqa 类型的 blip_vqa 模型
        >>> model = load_model("blip_vqa", "aokvqa") # 加载 aokvqa 类型的 blip_vqa 模型
    """

    # 定义预训练模型配置字典
    PRETRAINED_MODEL_CONFIG_DICT = {
        "vqav2": "configs/models/blip_vqav2.yaml", # vqav2 模型的配置文件路径
        "okvqa": "configs/models/blip_vqa_okvqa.yaml", # okvqa 模型的配置文件路径
        "aokvqa": "configs/models/blip_vqa_aokvqa.yaml", # aokvqa 模型的配置文件路径
    }

    # 类的初始化方法
    def __init__(self, image_encoder, text_encoder, text_decoder, max_txt_len=35):
        super().__init__() # 调用父类的初始化方法
        self.tokenizer = self.init_tokenizer() # 初始化分词器

        self.visual_encoder = image_encoder # 设置视觉编码器

        self.text_encoder = text_encoder # 设置文本编码器
        self.text_decoder = text_decoder # 设置文本解码器

        self.max_txt_len = max_txt_len # 设置最大文本长度

    # 模型的前向传播方法
    def forward(self, samples):
        """
        Args: # 参数：
            samples (dict): A dictionary containing the following keys: # 一个字典，包含以下键：
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480. # 形状为 (batch_size, 3, H, W) 的张量。输入图像。默认 H=480, W=480。
                - text_input (list): A list of strings, each string is a question # 字符串列表，每个字符串是一个问题
                - answer (list): A list of strings, each string is an answer # 字符串列表，每个字符串是一个答案
                - weight (torch.Tensor): A tensor used to weigh each answer in the loss computation. # 用于在损失计算中衡量每个答案权重的张量。
                   The shape of the tensor is (sum(n_answers),) # 张量的形状是 (sum(n_answers),)
                - n_answers (torch.Tensor): A tensor shape (batch_size,) containing the number of answers # 形状为 (batch_size,) 的张量，包含 batch 中每个问题的答案数量。
                     for each question in the batch. # 用于 batch 中的每个问题。

        Returns: # 返回值：
            A BlipOutput object containing loss and intermediate outputs, # 一个 BlipOutput 对象，包含损失和中间输出，
            see :class:`lavis.models.blip_outputs.BlipOutput` for more details. # 更多详情请参阅 :class:`lavis.models.blip_models.blip_outputs.BlipOutput`。

        Examples: # 示例：
        ```python
            >>> import torch # 导入 torch
            >>> from lavis.models import load_model # 从 lavis.models 导入 load_model
            >>> model = load_model("blip_vqa") # 加载 blip_vqa 模型
            >>> samples = { # 创建 samples 字典
            ...     "image": torch.rand(2, 3, 480, 480), # 随机图像张量
            ...     "text_input": ["What is this?", "What is that?"], # 文本输入列表 (问题)
            ...     "answer": ["cat", "cat", "dog"], # 答案列表
            ...     "weight": torch.tensor([1.0, 1.0, 1.0]), # 权重张量
            ...     "n_answers": torch.tensor([2, 1]), # 每个问题的答案数量张量
            ... }
            >>> output = model(samples) # 调用模型进行前向传播
            >>> output.keys() # 查看输出的键
            odict_keys(['intermediate_output', 'loss']) # 输出的键
            >>> output.intermediate_output.keys() # 查看中间输出的键
            odict_keys(['image_embeds', 'encoder_output', 'decoder_output', 'decoder_labels']) # 中间输出的键
        ```
        """
        # 前向传播编码器，获取编码器输出和图像嵌入
        encoder_output, image_embeds = self.forward_encoder(samples)
        # 前向传播解码器，获取损失、解码器输出和解码器目标
        loss, decoder_output, decoder_targets = self.forward_decoder(
            samples=samples, encoder_out=encoder_output # 传入 samples 和编码器输出
        )

        # 返回 BlipOutput 对象
        return BlipOutput(
            loss=loss, # 损失
            intermediate_output=BlipIntermediateOutput( # 中间输出信息
                image_embeds=image_embeds, # 图像嵌入
                encoder_output=encoder_output, # 编码器输出
                decoder_output=decoder_output, # 解码器输出
                decoder_labels=decoder_targets, # 解码器目标
            ),
        )

    # 前向传播编码器方法
    def forward_encoder(self, samples):
        questions = samples["text_input"] # 获取问题列表
        # 使用分词器处理问题
        questions = self.tokenizer(
            questions, # 问题列表
            padding="longest", # 填充到最长序列
            truncation=True, # 截断超过最大长度的文本
            max_length=self.max_txt_len, # 最大文本长度
            return_tensors="pt", # 返回 PyTorch 张量
        ).to(self.device) # 将张量移动到设备
        # 将第一个 token (通常是 [CLS]) 替换为编码器 token ID
        questions.input_ids[:, 0] = self.tokenizer.enc_token_id
        # 更新 samples 字典，添加 tokenized_text
        samples.update({"tokenized_text": questions})

        # 使用视觉编码器提取图像特征
        image_embeds = self.visual_encoder.forward_features(samples["image"])
        # 使用文本编码器进行自动掩码前向传播，结合 tokenized_text 和 visual_embeds
        encoder_output = self.text_encoder.forward_automask(
            tokenized_text=samples["tokenized_text"], visual_embeds=image_embeds
        )

        return encoder_output, image_embeds # 返回编码器输出和图像嵌入

    # 前向传播解码器方法
    def forward_decoder(self, samples, encoder_out, **kwargs):
        # 使用分词器处理答案
        answers = self.tokenizer(
            samples["answer"], padding="longest", return_tensors="pt" # 填充到最长序列，返回 PyTorch 张量
        ).to(self.device) # 将张量移动到设备
        # 将第一个 token (通常是 [CLS]) 替换为 BOS token ID
        answers.input_ids[:, 0] = self.tokenizer.bos_token_id
        # 创建答案目标，将填充 token 替换为 -100 (在损失计算中忽略)
        answer_targets = answers.input_ids.masked_fill(
            answers.input_ids == self.tokenizer.pad_token_id, -100
        )

        question_states = [] # 存储问题状态的列表
        question_atts = [] # 存储问题注意力掩码的列表

        question = samples["tokenized_text"] # 获取 tokenized 问题
        question_output = encoder_out # 获取编码器输出 (问题编码结果)

        # 根据每个问题的答案数量复制问题状态和注意力掩码
        for b, n in enumerate(samples["n_answers"]):
            question_states += [question_output.last_hidden_state[b]] * n # 复制问题隐藏状态 n 次
            question_atts += [question.attention_mask[b]] * n # 复制问题注意力掩码 n 次

        question_states = torch.stack(question_states, dim=0) # 将列表堆叠成张量
        question_atts = torch.stack(question_atts, dim=0) # 将列表堆叠成张量

        # 使用文本解码器处理答案
        answer_output = self.text_decoder(
            answers.input_ids, # 答案输入 ID
            attention_mask=answers.attention_mask, # 答案注意力掩码
            encoder_hidden_states=question_states, # 编码器隐藏状态 (问题状态)
            encoder_attention_mask=question_atts, # 编码器注意力掩码 (问题注意力)
            labels=answer_targets, # 答案目标
            return_dict=True, # 返回字典格式的输出
            reduction="none", # 不对损失进行 reduction
        )

        # 计算加权损失
        loss = samples["weight"] * answer_output.loss
        bsz = samples["image"].size(0) # 获取 batch size

        loss = loss.sum() / bsz # 计算平均损失 (按 batch size 平均)

        return loss, answer_output, answer_targets # 返回损失、解码器输出和解码器目标

    # 预测答案方法
    def predict_answers(
        self,
        samples,
        num_beams=3, # beam search 的 beam 数量
        inference_method="rank", # 推理方法："rank" 或 "generate"
        max_len=10, # 生成答案的最大长度
        min_len=1, # 生成答案的最小长度
        num_ans_candidates=128, # 答案候选数量 (用于 ranking)
        answer_list=None, # 答案列表 (用于 ranking)
        **kwargs
    ):
        """
        Args: # 参数：
            samples (dict): A dictionary containing the following keys: # 一个字典，包含以下键：
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480. # 形状为 (batch_size, 3, H, W) 的张量。输入图像。默认 H=480, W=480。
                - text_input (str or [str]): String or a list of strings, each string is a question. # 字符串或字符串列表，每个字符串是一个问题。
                                             The number of questions must be equal to the batch size. If a single string, will be converted to a list of string, with length 1 first. # 问题数量必须等于 batch size。如果是一个字符串，将首先转换为长度为 1 的字符串列表。
            num_beams (int): Number of beams for beam search. 1 means no beam search. # beam search 的 beam 数量。1 表示没有 beam search。
            inference_method (str): Inference method. One of "rank", "generate". # 推理方法。可以是 "rank" 或 "generate"。
                - If "rank", the model will return answers with the highest probability from the answer list. # 如果是 "rank"，模型将从答案列表中返回概率最高的答案。
                - If "generate", the model will generate answers. # 如果是 "generate"，模型将生成答案。
            max_len (int): Maximum length of generated answers. # 生成答案的最大长度。
            min_len (int): Minimum length of generated answers. # 生成答案的最小长度。
            num_ans_candidates (int): Number of answer candidates, used to filter out answers with low probability. # 答案候选数量，用于过滤掉低概率的答案。
            answer_list (list): A list of strings, each string is an answer. # 字符串列表，每个字符串是一个答案。

        Returns: # 返回值：
            List: A list of strings, each string is an answer. # 字符串列表，每个字符串是一个答案。

        Examples: # 示例：
        ```python
            >>> from PIL import Image # 从 PIL 导入 Image
            >>> from lavis.models import load_model_and_preprocess # 从 lavis.models 导入 load_model_and_preprocess
            >>> model, vis_processors, txt_processors = load_model_and_preprocess("blip_vqa", "vqav2") # 加载模型和预处理工具
            >>> raw_image = Image.open("docs/data/merlion.png").convert("RGB") # 打开图像并转换为 RGB
            >>> question = "Which city is this photo taken?" # 定义问题
            >>> image = vis_processors["eval"](raw_image).unsqueeze(0) # 预处理图像并添加 batch 维度
            >>> question = txt_processors["eval"](question) # 预处理问题文本
            >>> samples = {"image": image, "text_input": [question]} # 创建 samples 字典
            >>> answers = model.predict_answers(samples) # 预测答案
            >>> answers # 打印答案
            ['singapore'] # 输出答案
            >>> answer_list = ["Singapore", "London", "Palo Alto", "Tokyo"] # 定义答案列表
            >>> answers = model.predict_answers(samples, answer_list=answer_list) # 使用答案列表预测答案
            >>> answers # 打印答案
            ['Singapore'] # 输出答案
        ```
        """
        # 检查推理方法是否有效
        assert inference_method in [
            "rank",
            "generate",
        ], "Inference method must be one of 'rank' or 'generate', got {}.".format(
            inference_method
        )

        # 如果文本输入是单个字符串，转换为列表
        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        # 检查问题数量是否等于 batch size
        assert len(samples["text_input"]) == samples["image"].size(
            0
        ), "The number of questions must be equal to the batch size."

        # 根据推理方法调用相应的函数
        if inference_method == "generate":
            return self._generate_answers(
                samples, num_beams=num_beams, max_length=max_len, min_length=min_len # 调用生成答案函数
            )
        elif inference_method == "rank":
            # 如果是 ranking 方法，必须提供答案列表
            assert answer_list is not None, "answer_list must be provided for ranking"

            # 限制答案候选数量不超过答案列表长度
            num_ans_candidates = min(num_ans_candidates, len(answer_list))

            return self._rank_answers(
                samples, answer_list=answer_list, num_ans_candidates=num_ans_candidates # 调用 ranking 答案函数
            )

    # 生成答案方法 (使用 beam search)
    def _generate_answers(self, samples, num_beams=3, max_length=10, min_length=1):
        # 前向传播编码器，获取编码器输出
        encoder_out, _ = self.forward_encoder(samples)

        question_output = encoder_out # 获取问题编码结果

        # 重复问题状态和注意力掩码以匹配 beam 数量
        question_states = question_output.last_hidden_state.repeat_interleave(
            num_beams, dim=0
        )
        question_atts = torch.ones(question_states.size()[:-1], dtype=torch.long).to(
            self.device
        )

        # 构建模型 kwargs 用于生成
        model_kwargs = {
            "encoder_hidden_states": question_states, # 编码器隐藏状态
            "encoder_attention_mask": question_atts, # 编码器注意力掩码
        }

        bsz = samples["image"].size(0) # 获取 batch size
        # 创建 BOS token ID 的初始输入
        bos_ids = torch.full(
            (bsz, 1), fill_value=self.tokenizer.bos_token_id, device=self.device
        )

        # 使用文本解码器生成答案
        outputs = self.text_decoder.generate(
            input_ids=bos_ids, # 初始输入 ID
            max_length=max_length, # 最大生成长度
            min_length=min_length, # 最小生成长度
            num_beams=num_beams, # beam 数量
            eos_token_id=self.tokenizer.sep_token_id, # EOS token ID
            pad_token_id=self.tokenizer.pad_token_id, # PAD token ID
            **model_kwargs # 其他模型参数
        )

        # 收集生成的答案
        answers = [] # 存储答案的列表
        for output in outputs: # 遍历生成的序列
            answer = self.tokenizer.decode(output, skip_special_tokens=True) # 解码序列为文本，跳过特殊 token
            answers.append(answer) # 将答案添加到列表

        return answers # 返回答案列表

    # Ranking 答案方法
    def _rank_answers(self, samples, answer_list, num_ans_candidates):
        """
        Generate the first token of answers using decoder and select ${num_ans_candidates}
        most probable ones. Then select answers from answer list, which start with the probable tokens.
        Lastly, use the selected answers as the ground-truth labels for decoding and calculating LM loss.
        Return the answers that minimize the losses as result.

        使用解码器生成答案的第一个 token，并选择 ${num_ans_candidates} 个最可能的 token。
        然后从答案列表中选择以这些可能 token 开头的答案。
        最后，使用选定的答案作为解码的 ground-truth 标签并计算 LM 损失。
        返回使损失最小的答案作为结果。
        """
        # 使用分词器处理答案列表
        answer_candidates = self.tokenizer(
            answer_list, padding="longest", return_tensors="pt" # 填充到最长序列，返回 PyTorch 张量
        ).to(self.device) # 将张量移动到设备
        # 将第一个 token (通常是 [CLS]) 替换为 BOS token ID
        answer_candidates.input_ids[:, 0] = self.tokenizer.bos_token_id

        answer_ids = answer_candidates.input_ids # 获取答案 ID
        answer_atts = answer_candidates.attention_mask # 获取答案注意力掩码

        # 前向传播编码器，获取问题编码结果
        question_output, _ = self.forward_encoder(samples)
        question_states = question_output.last_hidden_state # 获取问题隐藏状态

        tokenized_question = samples["tokenized_text"] # 获取 tokenized 问题
        question_atts = tokenized_question.attention_mask # 获取问题注意力掩码

        num_ques = question_states.size(0) # 获取问题数量
        # 创建初始输入 (BOS token)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        # 使用文本解码器处理初始输入，获取第一个 token 的 logits
        start_output = self.text_decoder(
            start_ids, # 初始输入 ID
            encoder_hidden_states=question_states, # 编码器隐藏状态 (问题状态)
            encoder_attention_mask=question_atts, # 编码器注意力掩码 (问题注意力)
            return_dict=True, # 返回字典格式的输出
            reduction="none", # 不对损失进行 reduction
        )
        logits = start_output.logits[:, 0, :]  # 第一个 token 的 logits

        # topk_probs: top-k 概率
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1] # 获取答案的第一个 token ID (跳过 BOS)
        # 计算第一个 token 的概率，并选择答案列表中第一个 token 的概率
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token
        )
        # 选择 top-k 概率和对应的 ID
        topk_probs, topk_ids = prob_first_token.topk(num_ans_candidates, dim=1)

        # answer input: [num_question*k, answer_len] # 答案输入：[问题数量*k, 答案长度]
        input_ids = [] # 存储输入 ID 的列表
        input_atts = [] # 存储输入注意力掩码的列表
        # 根据 top-k ID 选择对应的答案 ID 和注意力掩码
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id)) # 选择对应的答案 ID
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id)) # 选择对应的答案注意力掩码
        input_ids = torch.cat(input_ids, dim=0) # 将列表连接成张量
        input_atts = torch.cat(input_atts, dim=0) # 将列表连接成张量

        # 创建目标 ID，将填充 token 替换为 -100
        targets_ids = input_ids.masked_fill(
            input_ids == self.tokenizer.pad_token_id, -100
        )

        # repeat encoder's output for top-k answers # 重复编码器输出以匹配 top-k 答案
        question_states = tile(question_states, 0, num_ans_candidates) # 复制问题状态 k 次
        question_atts = tile(question_atts, 0, num_ans_candidates) # 复制问题注意力掩码 k 次

        # 使用文本解码器处理选定的答案候选
        output = self.text_decoder(
            input_ids, # 输入 ID (选定的答案候选)
            attention_mask=input_atts, # 注意力掩码
            encoder_hidden_states=question_states, # 编码器隐藏状态 (重复的问题状态)
            encoder_attention_mask=question_atts, # 编码器注意力掩码 (重复的问题注意力)
            labels=targets_ids, # 目标 ID
            return_dict=True, # 返回字典格式的输出
            reduction="none", # 不对损失进行 reduction
        )

        log_probs_sum = -output.loss # 计算 log 概率之和 (损失的负值)
        log_probs_sum = log_probs_sum.view(num_ques, num_ans_candidates) # 将结果 reshape 为 [问题数量, 候选数量]

        max_topk_ids = log_probs_sum.argmax(dim=1) # 选择 log 概率之和最大的候选的索引
        max_ids = topk_ids[max_topk_ids >= 0, max_topk_ids] # 根据索引获取对应的答案列表中的 ID

        # 根据 ID 选择最终的答案
        answers = [answer_list[max_id] for max_id in max_ids]

        return answers # 返回最终答案列表

    # 从配置字典创建模型
    @classmethod
    def from_config(cls, cfg=None):
        # 从配置创建视觉编码器
        image_encoder = VisionTransformerEncoder.from_config(cfg)

        # text encoder + multimodal encoder # 文本编码器 + 多模态编码器
        # 从配置创建文本编码器
        text_encoder = XBertEncoder.from_config(cfg)
        # 从配置创建文本解码器
        text_decoder = XBertLMHeadDecoder.from_config(cfg)

        # 从配置获取最大文本长度，默认为 35
        max_txt_len = cfg.get("max_txt_len", 35)

        # 创建 BlipVQA 模型实例
        model = cls(
            image_encoder=image_encoder, # 视觉编码器
            text_encoder=text_encoder, # 文本编码器
            text_decoder=text_decoder, # 文本解码器
            max_txt_len=max_txt_len, # 最大文本长度
        )

        # 从配置加载检查点
        model.load_checkpoint_from_config(cfg)

        return model # 返回创建的模型实例
