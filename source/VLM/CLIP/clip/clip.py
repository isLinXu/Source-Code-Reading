# 导入hashlib库，用于计算文件的SHA256哈希值
import hashlib
# 导入os库，用于与操作系统交互
import os
# 导入urllib库，用于处理URL
import urllib
# 导入warnings库，用于发出警告
import warnings
# 从packaging库导入version模块，用于版本比较
from packaging import version
# 导入Union和List类型提示
from typing import Union, List

# 导入torch库
import torch
# 从PIL库导入Image类，用于图像处理
from PIL import Image
# 从torchvision.transforms导入多个图像变换函数
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# 导入tqdm库，用于显示进度条
from tqdm import tqdm

# 从当前模块导入build_model函数
from .model import build_model
# 从简单分词器模块导入SimpleTokenizer类并重命名为_Tokenizer
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    # 尝试从torchvision.transforms导入插值模式
    from torchvision.transforms import InterpolationMode
    # 设置插值模式为BICUBIC
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    # 如果导入失败，则使用PIL库的BICUBIC
    BICUBIC = Image.BICUBIC

# 检查PyTorch版本是否低于1.7.1
if version.parse(torch.__version__) < version.parse("1.7.1"):
    # 如果是，发出警告，建议使用1.7.1或更高版本
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

# 指定该模块公开的接口，包含available_models、load和tokenize函数
__all__ = ["available_models", "load", "tokenize"]
# 初始化简单分词器
_tokenizer = _Tokenizer()

# 定义可用模型的字典，包含模型名称和对应的下载链接
_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    # RN50模型的下载链接
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    # RN101模型的下载链接
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    # RN50x4模型的下载链接
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    # RN50x16模型的下载链接
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    # RN50x64模型的下载链接
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    # ViT-B/32模型的下载链接
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    # ViT-B/16模型的下载链接
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    # ViT-L/14模型的下载链接
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    # ViT-L/14@336px模型的下载链接
}

# 定义下载函数，接收URL和下载根目录
def _download(url: str, root: str):
    # 创建下载目录，如果已存在则不报错
    os.makedirs(root, exist_ok=True)
    # 提取文件名
    filename = os.path.basename(url)

    # 从URL中提取预期的SHA256校验和
    expected_sha256 = url.split("/")[-2]
    # 构建下载目标路径
    download_target = os.path.join(root, filename)

    # 如果目标路径存在但不是文件，抛出错误
    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    # 如果文件已存在且SHA256校验和匹配，返回文件路径
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            # 如果SHA256校验和不匹配，发出警告并重新下载文件
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    # 打开URL源和目标文件
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        # 使用tqdm显示下载进度条
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            # 每次读取8192字节
            while True:
                buffer = source.read(8192)
                # 如果没有更多数据，退出循环
                if not buffer:
                    break

                # 将读取的数据写入目标文件
                output.write(buffer)
                # 更新进度条
                loop.update(len(buffer))

    # 下载完成后再次检查SHA256校验和，如果不匹配，抛出错误
    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target
    # 返回下载目标路径

# 将图像转换为RGB格式
def _convert_image_to_rgb(image):
    return image.convert("RGB")

# 定义图像预处理的变换函数
def _transform(n_px):
    return Compose([
        # 调整图像大小，使用双三次插值
        Resize(n_px, interpolation=BICUBIC),
        # 中心裁剪图像
        CenterCrop(n_px),
        # 转换图像为RGB
        _convert_image_to_rgb,
        # 将图像转换为张量
        ToTensor(),
        # 对图像进行归一化处理
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

# 返回可用的CLIP模型名称
def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

# 加载CLIP模型
def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    # 如果模型名称在可用模型中，下载该模型
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    # 如果提供的是文件路径，直接使用该路径
    elif os.path.isfile(name):
        model_path = name
    # 如果模型未找到，抛出错误并列出可用模型
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    # 打开模型文件
    with open(model_path, 'rb') as opened_file:
        try:
            # 尝试加载JIT模型
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # 如果加载失败，尝试加载状态字典
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    # 如果不是JIT模式，构建模型并转移到指定设备
    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        # 如果设备是CPU，将模型转换为float类型
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)
        # 返回模型和预处理函数

    # 修补设备名称
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    # 获取节点的属性
    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a node which is polymorphic over return type.
        
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    # 修补设备节点
    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # 修补浮点类型
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        # 修补浮点类型
        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())
    # 返回模型和预处理函数


# 返回tokenized结果
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    # 如果输入是字符串，将其转换为列表
    if isinstance(texts, str):
        texts = [texts]

    # 获取开始和结束标记的token
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # 对每个文本进行编码，并添加开始和结束标记
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    # 如果torch版本低于1.8.0，返回LongTensor
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        # 否则返回IntTensor
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    # 将tokens填充到结果张量中
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
                # 如果tokens长度超过context_length且允许截断，截断并替换最后一个token为结束标记
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
                # 如果不允许截断，抛出错误

        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
    # 返回tokenized结果