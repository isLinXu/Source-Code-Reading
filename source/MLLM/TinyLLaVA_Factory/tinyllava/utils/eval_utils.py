import os
from PIL import Image
from io import BytesIO
import base64
from transformers import AutoTokenizer
import torch
from transformers import StoppingCriteria, PhiForCausalLM

def disable_torch_init():
    """
    禁用torch的冗余默认初始化以加速模型创建。
    Disable the redundant torch default initialization to accelerate model creation.
    """
    # 设置torch.nn.Linear的reset_parameters方法为空操作，以禁用其默认初始化
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    # 设置torch.nn.LayerNorm的reset_parameters方法为空操作，以禁用其默认初始化
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
 
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        """
        初始化KeywordsStoppingCriteria类。

        :param keywords: 关键词列表
        :param tokenizer: 分词器对象
        :param input_ids: 输入ID张量
        """
        self.keywords = keywords                                                            # 关键词列表
        self.keyword_ids = []                                                               # 存储关键词ID的列表
        self.max_keyword_len = 0                                                            # 最大关键词长度

        # 遍历关键词列表，获取每个关键词的ID，并更新最大关键词长度
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            # 如果关键词ID列表长度大于1且第一个元素是bos_token_id，则去掉第一个元素
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            # 更新最大关键词长度
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            # 将当前关键词ID添加到列表中
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer                                                           # 分词器对象
        self.start_len = input_ids.shape[1]                                                  # 输入ID张量的第二个维度大小

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        对一批输出ID进行评估，判断是否包含关键词。

        :param output_ids: 一个包含输出ID的张量，形状为(batch_size, sequence_length)
        :param scores: 一个包含分数的张量，形状为(batch_size, sequence_length)
        :param kwargs: 其他可能需要的参数
        :return: 如果输出中包含任何关键词，则返回True，否则返回False
        """
        # 计算可以检查的最大关键词长度，即输出ID张量的第二个维度大小减去输入ID张量的第二个维度大小
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)

        # 确保所有关键词ID都在相同的设备上（CPU或GPU）
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]

        # 检查输出的最后一部分是否与任何关键词ID完全匹配
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True

        # 解码输出的最后一小部分，跳过特殊标记
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]

        # 检查解码后的输出中是否包含任何关键词
        for keyword in self.keywords:
            if keyword in outputs:
                return True

        # 如果没有找到任何关键词，则返回False
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        对于给定的输出ID和分数，调用此方法将逐个批次地处理它们，并返回所有批次处理结果的逻辑与。

        :param output_ids: 一个包含输出ID的张量，形状为 (N, ...)，其中 N 是批次大小。
        :param scores: 一个包含分数的张量，形状应与 output_ids 相同。
        :param kwargs: 其他可能传递给 call_for_batch 方法的关键字参数。
        :return: 如果所有批次的处理结果都为真，则返回 True，否则返回 False。
        """
        outputs = []                                                                    # 初始化一个空列表来存储每个批次的处理结果
        for i in range(output_ids.shape[0]):                                            # 遍历每个批次
            # 对于每个批次，调用 call_for_batch 方法并将结果添加到 outputs 列表中
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)                                                             # 如果 outputs 列表中的所有元素都为真，则返回 True，否则返回 False
    
def load_image_from_base64(image):
    """
    从base64编码的字符串中加载图片。

    参数:
    image (str): 包含图片数据的base64编码字符串。

    返回:
    Image: PIL库中的Image对象，代表解码后的图片。
    """
    # 使用BytesIO将base64解码后的字节流转换为文件对象
    # 使用PIL库的Image.open方法从文件对象中加载图片
    return Image.open(BytesIO(base64.b64decode(image)))
