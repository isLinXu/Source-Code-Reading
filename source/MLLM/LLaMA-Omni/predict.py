# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import subprocess
import json
import soundfile as sf
import torch

from cog import BasePredictor, Input, Path, BaseModel
from fairseq import utils as fairseq_utils
from fairseq.models.text_to_speech.vocoder import CodeHiFiGANVocoder

from omni_speech.model.builder import load_pretrained_model
from omni_speech.utils import disable_torch_init
from omni_speech.infer.infer import create_data_loader, ctc_postprocess

# 定义模型缓存目录
MODEL_CACHE = "models"
# 定义模型下载URL格式化字符串
MODEL_URL = (
    f"https://weights.replicate.delivery/default/ictnlp/LLaMA-Omni/{MODEL_CACHE}.tar"
)
# 设置环境变量，禁用数据集在线更新
os.environ["HF_DATASETS_OFFLINE"] = "1"
# 设置环境变量，禁用Transformers在线更新
os.environ["TRANSFORMERS_OFFLINE"] = "1"
# 设置环境变量，指定HuggingFace模型缓存目录
os.environ["HF_HOME"] = MODEL_CACHE
# 设置环境变量，指定PyTorch模型缓存目录
os.environ["TORCH_HOME"] = MODEL_CACHE
# 设置环境变量，指定HuggingFace数据集缓存目录
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
# 设置环境变量，指定Transformers缓存目录
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
# 设置环境变量，指定HuggingFace Hub缓存目录
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE

# 定义模型输出类，包含音频文件路径和识别文本
class ModelOutput(BaseModel):
    audio: Path
    text: str

# 定义下载权重函数
def download_weights(url, dest):
    # 记录下载开始时间
    start = time.time()
    # 打印下载URL
    print("downloading url: ", url)
    # 打印下载目标路径
    print("downloading to: ", dest)
    # 使用pget命令下载文件，支持断点续传
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    # 计算并打印下载耗时
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        """
        加载模型到内存中，以提高运行多个预测的效率。
        首先检查模型缓存文件是否存在，如果不存在，则从指定的URL下载模型权重。
        然后，加载预训练的模型和分词器，以及用于语音合成的vocoder模型。
        """
        # 检查模型缓存文件是否存在，如果不存在，则下载模型权重
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)

        # Model
        # 禁用PyTorch的初始化检查，加快加载速度
        disable_torch_init()

        # 加载预训练的模型和分词器
        self.tokenizer, self.model, _ = load_pretrained_model(
            f"{MODEL_CACHE}/Llama-3.1-8B-Omni", model_base=None, s2s=True
        )

        # 读取vocoder的配置文件
        with open(f"{MODEL_CACHE}/vocoder/config.json") as f:
            vocoder_cfg = json.load(f)

        # 初始化vocoder模型，用于将编码转换为音频信号
        self.vocoder = CodeHiFiGANVocoder(
            f"{MODEL_CACHE}/vocoder/g_00500000", vocoder_cfg
        ).cuda()

    def predict(
        self,                                                                           # 当前类的实例
        input_audio: Path = Input(description="Input audio"),                           # 输入音频文件路径
        prompt: str = Input(                                                            # 提示文本，默认值为"Please directly answer the questions in the user's speech"
            default="Please directly answer the questions in the user's speech"
        ),
        temperature: float = Input(                                                     # 控制随机性的参数，范围在0到1之间，默认值为0.0
            description="Controls randomness. Lower values make the model more deterministic, higher values make it more random.",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        top_p: float = Input(                                                            # 控制输出多样性的参数，当temperature > 0时有效，默认值为0.0
            description="Controls diversity of the output. Valid when temperature > 0. Lower values make the output more focused, higher values make it more diverse.",
            default=0.0,
            ge=0.0,
            le=1.0,
        ),
        max_new_tokens: int = Input(                                                      # 生成的最大token数，默认值为256
            description="Maximum number of tokens to generate", default=256, ge=1
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        """运行模型进行单次预测"""  # 函数文档字符串

        questions = [                           # 构建输入数据
            {
                "speech": str(input_audio),
                "conversations": [{"from": "human", "value": f"<speech>\n{prompt}"}],
            }
        ]

        data_loader = create_data_loader(       # 创建数据加载器
            questions,
            self.tokenizer,
            self.model.config,
            input_type="mel",
            mel_size=128,
            conv_mode="llama_3",
        )

        (input_ids, speech_tensor, speech_length) = next(iter(data_loader)) # 获取数据加载器中的数据

        # 将数据移动到GPU
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        speech_tensor = speech_tensor.to(
            dtype=torch.float16, device="cuda", non_blocking=True
        )
        speech_length = speech_length.to(device="cuda", non_blocking=True)

        with torch.inference_mode():                                        # 使用推理模式
            output_ids, output_units = self.model.generate(                 # 生成输出
                input_ids,
                speech=speech_tensor,
                speech_lengths=speech_length,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p if temperature > 0 else None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=128004,
                streaming_unit_gen=False,
            )

        prediction = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()                                                           # 解码生成的token

        output_units = ctc_postprocess(                                     # 后处理输出单元
            output_units, blank=self.model.config.unit_vocab_size
        )

        print(prediction)                                                   # 打印预测文本

        print(f"output_units: {output_units}")                              # 打印输出单元
        print(type(output_units))                                           # 打印输出单元类型

        output_units = [(list(map(int, output_units.strip().split())))]

        x = {
            "code": torch.LongTensor(output_units[0]).view(1, -1),
        }

        x = fairseq_utils.move_to_cuda(x)                                   # 将数据移动到GPU
        wav = self.vocoder(x, True)                                         # 生成音频波形

        out_path = "/tmp/out.wav"                                           # 输出音频文件路径

        sf.write(                                                           # 保存音频文件
            out_path,
            wav.detach().cpu().numpy(),
            16000,
        )

        return ModelOutput(audio=Path(out_path), text=prediction)           # 返回预测结果
