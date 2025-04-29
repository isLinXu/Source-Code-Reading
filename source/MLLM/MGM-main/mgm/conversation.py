import dataclasses
from enum import auto, Enum
from typing import List, Tuple
import base64
from io import BytesIO
from PIL import Image


class SeparatorStyle(Enum):
    """Different separator style."""  # 英文注释保留：定义不同的对话分隔符风格
    SINGLE = auto()    # 单分隔符风格（全程使用相同分隔符）
    TWO = auto()       # 双分隔符交替风格（用户和助手使用不同分隔符）
    MPT = auto()       # MPT模型专用格式（使用<|im_start|>标签）
    PLAIN = auto()     # 无格式纯文本（仅用换行符分隔）
    LLAMA_2 = auto()   # LLaMA-2风格（使用[INST]指令标签）
    LLAMA_3 = auto()   # LLaMA-3风格（使用<|eot_id|>结束标记）
    GEMMA = auto()     # Google Gemma风格（使用<start_of_turn>标签）


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""  # 英文注释保留：对话历史管理类
    system: str         # 系统级提示（定义助手行为准则）
    roles: List[str]    # 对话角色列表（如["USER", "ASSISTANT"]）
    messages: List[List[str]]  # 消息存储结构，格式为[[角色, 内容], ...]
    offset: int         # 消息偏移量（用于增量加载）
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE  # 分隔符风格，默认单分隔符
    sep: str = "###"    # 主分隔符（默认###）
    sep2: str = None    # 辅助分隔符（双分隔符风格时使用）
    version: str = "Unknown"  # 模板版本标识（用于兼容不同版本）
    skip_next: bool = False  # 跳过标记（用于流式处理控制）

    def get_prompt(self):
        """构建完整的对话提示文本"""
        # 处理首条消息包含图像的特殊情况
        if len(self.messages) > 0 and isinstance(self.messages[0][1], tuple):
            messages = self.messages.copy()
            init_role, init_msg = messages[0]
            # 清理图像占位符并重构消息格式
            init_msg = init_msg[0].replace("<image>", "").strip()
            
            if 'mmtag' in self.version:  # 多模态标签版本处理
                messages[0] = (init_role, init_msg)
                # 插入图像标签和确认消息
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, f"<image>\n{init_msg}")

        # 单分隔符风格处理
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep  # 以系统提示开头
            for role, message in messages:
                if message:
                    if isinstance(message, tuple):  # 处理含图像的消息
                        message = message[0]  # 提取文本部分
                    ret += f"{role}: {message}{self.sep}"  # 添加角色和消息
                else:
                    ret += f"{role}:"  # 无消息时保留角色提示

        # 双分隔符交替风格处理
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]  # 分隔符交替列表
            ret = self.system + seps[0]  # 系统提示+第一个分隔符
            for i, (role, message) in enumerate(messages):
                if message:
                    if isinstance(message, tuple):
                        message = message[0]
                    # 交替使用分隔符（用户用sep，助手用sep2）
                    ret += f"{role}: {message}{seps[i % 2]}"
                else:
                    ret += f"{role}:"

        # MPT模型风格处理
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep  # 系统提示开头
            for role, message in messages:
                if message:
                    if isinstance(message, tuple):
                        message = message[0]
                    # MPT格式：角色直接拼接消息（无冒号）
                    ret += f"{role}{message}{self.sep}"
                else:
                    ret += role  # 无消息时仅保留角色

        # LLaMA-2风格处理
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            # 定义系统消息包装器（当系统提示非空时添加<<SYS>>标签）
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if msg else ""
            # 定义指令包装器（用户消息用[INST]包裹）
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            
            ret = ""
            for i, (role, message) in enumerate(messages):
                if i == 0:  # 首条消息验证
                    assert message, "首条消息不能为空"
                    assert role == self.roles[0], "首条消息必须来自用户"
                
                if message:
                    if isinstance(message, tuple):  # 处理多模态消息
                        message, _, _ = message  # 提取文本内容
                    if i == 0:  # 首条消息合并系统提示
                        message = wrap_sys(self.system) + message
                    # 用户消息添加[INST]标签，助手消息添加后缀
                    if i % 2 == 0:
                        ret += f"{self.sep}{wrap_inst(message)}"
                    else:
                        ret += f" {message} {self.sep2}"
            
            ret = ret.lstrip(self.sep)  # 去除开头的冗余分隔符

        # LLaMA-3风格处理
        elif self.sep_style == SeparatorStyle.LLAMA_3:
            ret = self.system + self.sep  # 系统提示开头
            for role, message in messages:
                if message:
                    if isinstance(message, tuple):
                        message = message[0]
                    # 直接拼接角色和消息（使用特殊标记）
                    ret += f"{role}{message}{self.sep}"
                else:
                    ret += role  # 无消息时保留角色标记

        # Gemma模型风格处理
        elif self.sep_style == SeparatorStyle.GEMMA:
            seps = [self.sep, self.sep2]  # 分隔符列表
            ret = self.system + seps[0]  # 系统提示开头
            for i, (role, message) in enumerate(messages):
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    # 添加对话转折标记
                    ret += f"<start_of_turn>{role}\n{message}<end_of_turn>\n{seps[i % 2]}"
                else:
                    ret += f"<start_of_turn>{role}\n"  # 无消息时保留开始标记

        # 纯文本风格处理
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]  # 分隔符（通常为换行符）
            ret = self.system  # 无系统提示前缀
            for i, (role, message) in enumerate(messages):
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    # 直接拼接消息和分隔符（忽略角色）
                    ret += f"{message}{seps[i % 2]}"
                else:
                    ret += ""  # 空消息不处理

        else:
            raise ValueError(f"无效的分隔风格: {self.sep_style}")

        return ret  # 返回构建完成的提示文本

    def append_message(self, role, message):
        """向对话历史追加新消息
        Args:
            role: str - 发送者角色（如 'user'/'assistant'）
            message: str/tuple - 消息内容（文本或含图像的元组）
        """
        self.messages.append([role, message])  # 将消息添加到消息列表

    def process_image(self, image, image_process_mode, return_pil=False, image_format='PNG', max_len=1344, min_len=672):
        """图像预处理方法
        Args:
            image: PIL.Image - 输入图像
            image_process_mode: str - 处理模式（Pad/Default/Crop/Resize）
            return_pil: bool - 是否返回PIL对象（否则返回Base64字符串）
            image_format: str - 输出格式（默认PNG）
            max_len: int - 最大边长限制（默认1344像素）
            min_len: int - 最小边长限制（默认672像素）
        """
        # 填充模式处理
        if image_process_mode == "Pad":
            def expand2square(pil_img, background_color=(122, 116, 104)):
                """将图像扩展为正方形
                Args:
                    pil_img: PIL.Image - 输入图像
                    background_color: tuple - 填充背景色 (R, G, B)
                Returns:
                    PIL.Image - 正方形图像
                """
                width, height = pil_img.size
                if width == height:  # 已经是正方形
                    return pil_img
                elif width > height:  # 横向扩展
                    result = Image.new(pil_img.mode, (width, width), background_color)  # 创建正方形画布
                    result.paste(pil_img, (0, (width - height) // 2))  # 垂直居中粘贴
                    return result
                else:  # 纵向扩展
                    result = Image.new(pil_img.mode, (height, height), background_color)  # 创建正方形画布
                    result.paste(pil_img, ((height - width) // 2, 0))  # 水平居中粘贴
                    return result
            image = expand2square(image)  # 应用填充处理
        elif image_process_mode in ["Default", "Crop"]:  # 默认或裁剪模式
            pass  # 不进行特殊处理
        elif image_process_mode == "Resize":  # 调整尺寸模式
            image = image.resize((336, 336))  # 固定尺寸调整到336x336
        else:
            raise ValueError(f"Invalid image_process_mode: {image_process_mode}")  # 无效模式报错

        # 动态尺寸调整（保持宽高比）
        if max(image.size) > max_len:  # 如果最长边超过限制
            max_hw, min_hw = max(image.size), min(image.size)  # 获取最大和最小边长
            aspect_ratio = max_hw / min_hw  # 计算宽高比
            shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))  # 计算最短边
            longest_edge = int(shortest_edge * aspect_ratio)  # 计算最长边
            W, H = image.size
            if H > W:  # 竖图
                H, W = longest_edge, shortest_edge
            else:  # 横图或方图
                H, W = shortest_edge, longest_edge
            image = image.resize((W, H))  # 调整尺寸

        # 返回处理结果
        if return_pil:  # 如果返回PIL对象
            return image
        else:  # 返回Base64字符串
            buffered = BytesIO()  # 创建内存缓冲区
            image.save(buffered, format=image_format)  # 图像编码
            img_b64_str = base64.b64encode(buffered.getvalue()).decode()  # Base64编码并解码为字符串
            return img_b64_str

    def get_images(self, return_pil=False):
        """从对话历史中提取并处理所有图像
        Args:
            return_pil: bool - 是否返回PIL对象列表
        Returns:
            List[PIL.Image/base64_str] - 处理后的图像列表
        """
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):  # 遍历消息
            if i % 2 == 0:  # 仅处理用户消息（假设偶数索引为用户消息）
                if type(msg) is tuple:  # 含图像的消息元组
                    msg, image, image_process_mode = msg  # 解包元组
                    image = self.process_image(image, image_process_mode, return_pil=return_pil)  # 处理图像
                    images.append(image)  # 添加到列表
        return images

    def to_gradio_chatbot(self):
        """将对话历史转换为Gradio聊天机器人兼容格式
        Returns:
            List[List[str]] - Gradio聊天机器人格式的对话历史
        """
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):  # 遍历消息
            if i % 2 == 0:  # 用户消息
                if type(msg) is tuple:  # 含图像的消息元组
                    msg, image, image_process_mode = msg
                    img_b64_str = self.process_image(
                        image, "Default", return_pil=False,
                        image_format='JPEG')  # 处理图像为JPEG格式
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'  # 生成HTML图像标签
                    msg = img_str + msg.replace('<image>', '').strip()  # 拼接图像和文本
                    ret.append([msg, None])  # 添加用户消息，助手消息为空
                else:
                    ret.append([msg, None])  # 添加纯文本用户消息
            else:  # 助手消息
                if type(msg) is tuple and len(msg) == 2:  # 含图像的消息元组
                    msg, img_b64_str = msg
                    img_str = f'<img src="data:image/jpeg;base64,{img_b64_str}" alt="user upload image" />'  # 生成HTML图像标签
                    msg = msg.strip() + img_str  # 拼接文本和图像
                ret[-1][-1] = msg  # 更新最后一条消息的助手回复
        return ret

    def copy(self):
        """创建对话对象的深拷贝
        Returns:
            Conversation - 新的对话对象
        """
        return Conversation(
            system=self.system,  # 复制系统提示
            roles=self.roles,    # 复制角色定义
            messages=[[x, y] for x, y in self.messages],  # 深拷贝消息列表
            offset=self.offset,  # 复制消息偏移量
            sep_style=self.sep_style,  # 复制分隔符风格
            sep=self.sep,        # 复制主分隔符
            sep2=self.sep2,      # 复制辅助分隔符
            version=self.version  # 复制版本标识
        )

    def dict(self):
        """将对话对象转换为字典格式
        Returns:
            dict - 包含对话信息的字典
        """
        if len(self.get_images()) > 0:  # 如果对话包含图像
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],  # 提取消息文本
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {  # 无图像时的简化格式
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

# Vicuna v0 对话模板
conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",  # 系统提示：定义助手行为准则
    roles=("Human", "Assistant"),  # 角色定义：人类和助手
    messages=(  # 示例对话
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,  # 消息偏移量
    sep_style=SeparatorStyle.SINGLE,  # 单分隔符风格
    sep="###",  # 分隔符
)

# Vicuna v1 对话模板
conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",  # 系统提示：定义助手行为准则
    roles=("USER", "ASSISTANT"),  # 角色定义：用户和助手
    version="v1",  # 版本标识
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.TWO,  # 双分隔符风格
    sep=" ",  # 主分隔符
    sep2="</s>",  # 辅助分隔符
)

# LLaMA-2 对话模板
conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",  # 系统提示：强调安全性和诚实性
    roles=("USER", "ASSISTANT"),  # 角色定义：用户和助手
    version="llama_v2",  # 版本标识
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.LLAMA_2,  # LLaMA-2 风格
    sep="<s>",  # 开始标记
    sep2="</s>",  # 结束标记
)

# LLaVA LLaMA-2 对话模板
conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",  # 系统提示：定义多模态助手行为准则
    roles=("USER", "ASSISTANT"),  # 角色定义：用户和助手
    version="llama_v2",  # 版本标识
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.LLAMA_2,  # LLaMA-2 风格
    sep="<s>",  # 开始标记
    sep2="</s>",  # 结束标记
)

# LLaMA-3 对话模板
conv_llama_3 = Conversation(
    system="<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",  # 系统提示：定义多模态助手行为准则
    roles=("<|start_header_id|>user<|end_header_id|>\n\n", 
           "<|start_header_id|>system<|end_header_id|>\n\n"),  # 角色定义：使用特殊标记
    version="llama_v3",  # 版本标识
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.LLAMA_3,  # LLaMA-3 风格
    sep="<|eot_id|>",  # 对话结束标记
)

# MPT 对话模板
conv_mpt = Conversation(
    system="""<|im_start|>system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",  # 系统提示：定义助手行为准则
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),  # 角色定义：使用特殊标记
    version="mpt",  # 版本标识
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.MPT,  # MPT 风格
    sep="<|im_end|>",  # 对话结束标记
)

# LLaVA 纯文本对话模板
conv_llava_plain = Conversation(
    system="",  # 无系统提示
    roles=("", ""),  # 无角色定义
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.PLAIN,  # 纯文本风格
    sep="\n",  # 分隔符为换行符
)

# LLaVA v0 对话模板
conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",  # 系统提示：定义助手行为准则
    roles=("Human", "Assistant"),  # 角色定义：人类和助手
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.SINGLE,  # 单分隔符风格
    sep="###",  # 分隔符
)

# LLaVA v0 多模态标签版本
conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",  # 系统提示：定义多模态助手行为准则，包含图像格式说明
    roles=("Human", "Assistant"),  # 角色定义：人类和助手
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.SINGLE,  # 单分隔符风格
    sep="###",  # 分隔符
    version="v0_mmtag",  # 版本标识：v0 多模态标签版本
)

# LLaVA v1 对话模板
conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",  # 系统提示：定义助手行为准则
    roles=("USER", "ASSISTANT"),  # 角色定义：用户和助手
    version="v1",  # 版本标识
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.TWO,  # 双分隔符风格
    sep=" ",  # 主分隔符
    sep2="</s>",  # 辅助分隔符
)

# Vicuna 图像特殊处理 v1 版本
conv_vicuna_imgsp_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",  # 系统提示：定义助手行为准则
    roles=("USER", "ASSISTANT"),  # 角色定义：用户和助手
    version="imgsp_v1",  # 版本标识：图像特殊处理 v1 版本
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.TWO,  # 双分隔符风格
    sep=" ",  # 主分隔符
    sep2="</s>",  # 辅助分隔符
)

# LLaVA v1 多模态标签版本
conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
           "The visual content will be provided with the following format: <Image>visual content</Image>.",  # 系统提示：定义多模态助手行为准则，包含图像格式说明
    roles=("USER", "ASSISTANT"),  # 角色定义：用户和助手
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.TWO,  # 双分隔符风格
    sep=" ",  # 主分隔符
    sep2="</s>",  # 辅助分隔符
    version="v1_mmtag",  # 版本标识：v1 多模态标签版本
)

# Phi-2 对话模板
conv_phi_2 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",  # 系统提示：定义助手行为准则
    roles=("USER", "ASSISTANT"),  # 角色定义：用户和助手
    version="phi2",  # 版本标识：Phi-2 模型
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.TWO,  # 双分隔符风格
    sep=" ",  # 主分隔符
    sep2="<|endoftext|>",  # 辅助分隔符：文本结束标记
)

# Mistral Instruct 对话模板
conv_mistral_instruct = Conversation(
    system="",  # 无系统提示
    roles=("USER", "ASSISTANT"),  # 角色定义：用户和助手
    version="llama_v2",  # 版本标识：LLaMA v2 风格
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.LLAMA_2,  # LLaMA-2 风格
    sep="<s>",  # 开始标记
    sep2="</s>",  # 结束标记
)

# Gemma 对话模板
conv_gemma = Conversation(
    system="",  # 无系统提示
    roles=("user", "model"),  # 角色定义：用户和模型
    version="gemma",  # 版本标识：Gemma 模型
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.GEMMA,  # Gemma 风格
    sep="",  # 主分隔符为空
    sep2="<eos>",  # 辅助分隔符：序列结束标记
)

# ChatML 直接对话模板
conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",  # 系统提示：直接回答问题
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),  # 角色定义：使用特殊标记
    version="mpt",  # 版本标识：MPT 模型
    messages=(),  # 初始空消息列表
    offset=0,  # 消息偏移量
    sep_style=SeparatorStyle.MPT,  # MPT 风格
    sep="<|im_end|>",  # 对话结束标记
)

# 默认对话模板
default_conversation = conv_vicuna_v1  # 默认使用 Vicuna v1 对话模板

# 对话模板字典
conv_templates = {
    "default": conv_vicuna_v0,  # 默认模板：Vicuna v0
    "v0": conv_vicuna_v0,  # Vicuna v0 模板
    "v1": conv_vicuna_v1,  # Vicuna v1 模板
    "vicuna_v1": conv_vicuna_v1,  # Vicuna v1 模板（别名）
    "phi_2": conv_phi_2,  # Phi-2 模板
    "gemma": conv_gemma,  # Gemma 模板
    "llama_2": conv_llama_2,  # LLaMA-2 模板
    "llama_3": conv_llama_3,  # LLaMA-3 模板
    "imgsp_v1": conv_vicuna_imgsp_v1,  # Vicuna 图像特殊处理 v1 模板
    "mistral_instruct": conv_mistral_instruct,  # Mistral Instruct 模板
    "chatml_direct": conv_chatml_direct,  # ChatML 直接对话模板
    "mistral_direct": conv_chatml_direct,  # Mistral 直接对话模板（别名）
    "plain": conv_llava_plain,  # 纯文本模板
    "v0_plain": conv_llava_plain,  # 纯文本模板（别名）
    "llava_v0": conv_llava_v0,  # LLaVA v0 模板
    "v0_mmtag": conv_llava_v0_mmtag,  # LLaVA v0 多模态标签版本
    "llava_v1": conv_llava_v1,  # LLaVA v1 模板
    "v1_mmtag": conv_llava_v1_mmtag,  # LLaVA v1 多模态标签版本
    "llava_llama_2": conv_llava_llama_2,  # LLaVA LLaMA-2 模板
    "mpt": conv_mpt,  # MPT 模板
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())  # 打印默认对话模板的提示文本